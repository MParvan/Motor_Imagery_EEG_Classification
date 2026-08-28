import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.eeg_bci.data import (
    DATASET_SESSION_CONFIGS,
    DATASET_VALIDATION_GROUP_BY_DEFAULTS,
    EEGTrials,
    SplitIndices,
    ValidationSplitConfig,
    _metadata_to_dict,
    get_development_and_test_indices,
    get_loso_subject_splits,
    get_subject_specific_splits,
    split_development_train_validation,
)
from src.eeg_bci.classical import (
    CLASSICAL_MODEL_CHOICES,
    DEFAULT_CSP_COMPONENTS,
    _effective_csp_components,
    build_classical_estimator,
    fit_predict_classical_split,
)
from src.eeg_bci.train import (
    MOTOR_IMAGERY_FMAX_HZ,
    MOTOR_IMAGERY_FMIN_HZ,
    _load_trials,
    create_arg_parser,
    summarize_accuracy,
)
from src.eeg_bci.utils import ZScoreScalerTorch


def make_2a_like_trials():
    subjects = []
    sessions = []
    runs = []
    labels = []
    rows = []
    value = 0.0
    for subject in [f"S{i}" for i in range(1, 10)]:
        for session in ["0train", "1test"]:
            for run in [f"run_{i}" for i in range(1, 7)]:
                subjects.append(subject)
                sessions.append(session)
                runs.append(run)
                labels.append(len(labels) % 2)
                rows.append(
                    np.array(
                        [
                            [value + 0.0, value + 1.0, value + 2.0, value + 3.0],
                            [value + 10.0, value + 11.0, value + 12.0, value + 13.0],
                        ],
                        dtype=np.float32,
                    )
                )
                value += 1.0
    return EEGTrials(
        X=np.stack(rows, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        metadata={
            "subject": np.asarray(subjects),
            "session": np.asarray(sessions),
            "run": np.asarray(runs),
        },
    ).validate()


def make_2b_like_trials():
    subjects = []
    sessions = []
    runs = []
    labels = []
    rows = []
    value = 100.0
    for subject in [f"S{i}" for i in range(1, 10)]:
        for session in ["0train", "1train", "2train", "3test", "4test"]:
            subjects.append(subject)
            sessions.append(session)
            runs.append(f"{session}_run")
            labels.append(len(labels) % 2)
            rows.append(
                np.array(
                    [
                        [value + 0.0, value + 1.0, value + 2.0, value + 3.0],
                        [value + 10.0, value + 11.0, value + 12.0, value + 13.0],
                    ],
                    dtype=np.float32,
                )
            )
            value += 1.0
    return EEGTrials(
        X=np.stack(rows, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        metadata={
            "subject": np.asarray(subjects),
            "session": np.asarray(sessions),
            "run": np.asarray(runs),
        },
    ).validate()


def make_classical_trials(n_subjects=3, n_classes=4, trials_per_subject=12, n_channels=8, n_times=32):
    rng = np.random.default_rng(123)
    subjects = []
    sessions = []
    runs = []
    labels = []
    rows = []
    for subject_index, subject in enumerate([f"S{i}" for i in range(1, n_subjects + 1)]):
        for trial_index in range(trials_per_subject):
            label = trial_index % n_classes
            subjects.append(subject)
            sessions.append(f"session_{trial_index % 2}")
            runs.append(f"run_{trial_index % 3}")
            labels.append(label)
            signal = rng.normal(loc=subject_index + label * 0.25, scale=0.5, size=(n_channels, n_times)).astype(np.float32)
            rows.append(signal)
    return EEGTrials(
        X=np.stack(rows, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        metadata={
            "subject": np.asarray(subjects),
            "session": np.asarray(sessions),
            "run": np.asarray(runs),
        },
    ).validate()


class TrialAlignmentTests(unittest.TestCase):
    def test_alignment_validation_passes(self):
        trials = make_2a_like_trials()
        self.assertEqual(trials.n_trials, 108)
        self.assertEqual(trials.subjects[0], "S1")

    def test_alignment_validation_fails_on_metadata_length_mismatch(self):
        with self.assertRaises(ValueError):
            EEGTrials(
                X=np.zeros((2, 3, 4), dtype=np.float32),
                y=np.zeros(2, dtype=np.int64),
                metadata={
                    "subject": np.asarray(["S1"]),
                    "session": np.asarray(["A", "B"]),
                    "run": np.asarray(["R1", "R2"]),
                },
            ).validate()

    def test_metadata_uses_positional_alignment_with_non_contiguous_pandas_index(self):
        meta = pd.DataFrame(
            {
                "subject": ["S1", "S2", "S3"],
                "session": ["0train", "0train", "1test"],
                "run": ["run_2", "run_1", "run_3"],
            },
            index=[10, 20, 40],
        )
        metadata = _metadata_to_dict(meta)
        trials = EEGTrials(
            X=np.arange(24, dtype=np.float32).reshape(3, 2, 4),
            y=np.array([0, 1, 0], dtype=np.int64),
            metadata=metadata,
        ).validate()
        self.assertEqual(trials.values_for("subject", [0, 2]), ("S1", "S3"))


class SessionConfigAndSplitTests(unittest.TestCase):
    def test_dataset_session_configs_are_present(self):
        self.assertEqual(DATASET_SESSION_CONFIGS["2a"].development_sessions, ("0train",))
        self.assertEqual(DATASET_SESSION_CONFIGS["2a"].test_sessions, ("1test",))
        self.assertEqual(DATASET_SESSION_CONFIGS["2b"].development_sessions, ("0train", "1train", "2train"))
        self.assertEqual(DATASET_SESSION_CONFIGS["2b"].test_sessions, ("3test", "4test"))

    def test_dataset_validation_group_defaults_are_present(self):
        self.assertEqual(DATASET_VALIDATION_GROUP_BY_DEFAULTS["2a"], "run")
        self.assertEqual(DATASET_VALIDATION_GROUP_BY_DEFAULTS["2b"], "session")

    def test_get_development_and_test_indices_uses_configured_sessions(self):
        trials = make_2a_like_trials()
        config = DATASET_SESSION_CONFIGS["2a"]
        development_idx, test_idx = get_development_and_test_indices(trials, subject="S1", session_config=config)
        self.assertEqual(trials.values_for("session", development_idx), ("0train",))
        self.assertEqual(trials.values_for("session", test_idx), ("1test",))

    def test_development_group_split_is_group_disjoint(self):
        trials = make_2a_like_trials()
        development_idx, _ = get_development_and_test_indices(
            trials, subject="S1", session_config=DATASET_SESSION_CONFIGS["2a"]
        )
        train_idx, val_idx, val_groups = split_development_train_validation(
            trials,
            development_idx=development_idx,
            config=ValidationSplitConfig(group_by="run", val_group_count=2, seed=3),
        )
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        self.assertEqual(set(trials.values_for("run", train_idx)) & set(trials.values_for("run", val_idx)), set())
        self.assertEqual(len(val_groups), 2)

    def test_development_group_split_is_seeded_and_order_invariant(self):
        trials = make_2a_like_trials()
        development_idx, _ = get_development_and_test_indices(
            trials, subject="S1", session_config=DATASET_SESSION_CONFIGS["2a"]
        )
        config_a = ValidationSplitConfig(group_by="run", val_group_count=2, seed=7)
        config_b = ValidationSplitConfig(group_by="run", val_group_count=2, seed=7)
        first = split_development_train_validation(trials, development_idx, config_a)[2]
        second = split_development_train_validation(trials, tuple(reversed(development_idx)), config_b)[2]
        self.assertEqual(first, second)

    def test_development_group_split_can_change_with_different_seeds(self):
        trials = make_2a_like_trials()
        development_idx, _ = get_development_and_test_indices(
            trials, subject="S1", session_config=DATASET_SESSION_CONFIGS["2a"]
        )
        first = split_development_train_validation(
            trials, development_idx, ValidationSplitConfig(group_by="run", val_group_count=2, seed=1)
        )[2]
        second = split_development_train_validation(
            trials, development_idx, ValidationSplitConfig(group_by="run", val_group_count=2, seed=2)
        )[2]
        self.assertNotEqual(first, second)

    def test_subject_specific_split_supports_run_group_validation_for_2a(self):
        trials = make_2a_like_trials()
        split = next(get_subject_specific_splits(trials, dataset="2a", validation_config=ValidationSplitConfig("run", 1, 0)))
        self.assertEqual(split.metadata_values(trials, "train", "session"), ("0train",))
        self.assertEqual(split.metadata_values(trials, "val", "session"), ("0train",))
        self.assertEqual(split.metadata_values(trials, "test", "session"), ("1test",))

    def test_subject_specific_split_supports_session_group_validation_for_2b(self):
        trials = make_2b_like_trials()
        split = next(
            get_subject_specific_splits(trials, dataset="2b", validation_config=ValidationSplitConfig("session", 1, 0))
        )
        self.assertEqual(split.metadata_values(trials, "test", "session"), ("3test", "4test"))
        self.assertEqual(len(split.val_groups), 1)
        self.assertIn(split.val_groups[0], {"0train", "1train", "2train"})

    def test_split_indices_are_immutable(self):
        split = SplitIndices(train_idx=(0, 1), val_idx=(2,), test_idx=(3,), protocol="demo")
        with self.assertRaises(AttributeError):
            split.train_idx += (4,)


class LosoTests(unittest.TestCase):
    def test_loso_subject_sets_are_disjoint(self):
        trials = make_2a_like_trials()
        for split in get_loso_subject_splits(trials):
            self.assertEqual(split.metadata_values(trials, "test", "subject"), (split.subject,))
            self.assertEqual(split.metadata_values(trials, "val", "subject"), (split.val_subject,))
            self.assertNotIn(split.subject, split.metadata_values(trials, "train", "subject"))
            self.assertNotIn(split.val_subject, split.metadata_values(trials, "train", "subject"))
            self.assertNotIn(split.subject, split.metadata_values(trials, "val", "subject"))
            self.assertNotIn(split.val_subject, split.metadata_values(trials, "test", "subject"))

    def test_loso_cycles_test_and_validation_subjects(self):
        trials = make_2a_like_trials()
        pairs = [(split.subject, split.val_subject) for split in get_loso_subject_splits(trials)]
        self.assertEqual(len(pairs), 9)
        self.assertEqual({test for test, _ in pairs}, set(trials.subjects))
        self.assertEqual({val for _, val in pairs}, set(trials.subjects))
        self.assertEqual(pairs[0], ("S1", "S2"))
        self.assertEqual(pairs[-1], ("S9", "S1"))


class ScalerTests(unittest.TestCase):
    def test_scaler_channel_statistics_are_pooled_over_trials_and_time(self):
        X = np.array(
            [
                [[1.0, 3.0], [10.0, 14.0]],
                [[5.0, 7.0], [18.0, 22.0]],
            ],
            dtype=np.float32,
        )
        scaler = ZScoreScalerTorch().fit(X)
        np.testing.assert_allclose(scaler.mean_, np.array([4.0, 16.0]))
        np.testing.assert_allclose(scaler.std_, np.array([math.sqrt(5.0), math.sqrt(20.0)]))

    def test_scaler_zero_variance_channel_uses_safe_unit_scale(self):
        X = np.array(
            [
                [[5.0, 5.0], [1.0, 2.0]],
                [[5.0, 5.0], [3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        scaler = ZScoreScalerTorch().fit(X)
        np.testing.assert_allclose(scaler.std_, np.array([1.0, np.std([1.0, 2.0, 3.0, 4.0])]))

    def test_scaler_transform_before_fit_raises_clear_error(self):
        with self.assertRaises(RuntimeError):
            ZScoreScalerTorch().transform(np.zeros((2, 3, 4), dtype=np.float32))

    def test_scaler_transform_does_not_mutate_input(self):
        X = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        X_before = X.copy()
        scaler = ZScoreScalerTorch().fit(X)
        _ = scaler.transform(X)
        np.testing.assert_array_equal(X, X_before)

    def test_scaler_is_not_affected_by_extreme_arrays(self):
        train = np.ones((2, 2, 3), dtype=np.float32)
        extreme = np.full((2, 2, 3), 1e9, dtype=np.float32)
        scaler = ZScoreScalerTorch().fit(train)
        before_mean = scaler.mean_.copy()
        before_std = scaler.std_.copy()
        _ = scaler.transform(extreme)
        np.testing.assert_allclose(scaler.mean_, before_mean)
        np.testing.assert_allclose(scaler.std_, before_std)

    def test_scaler_transform_preserves_shape(self):
        X = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        scaler = ZScoreScalerTorch().fit(X)
        transformed = scaler.transform(X)
        self.assertEqual(transformed.shape, X.shape)


class AggregationAndCliTests(unittest.TestCase):
    def test_classical_model_choices_are_advertised(self):
        parser = create_arg_parser()
        model_action = next(action for action in parser._actions if action.dest == "model")
        for name in CLASSICAL_MODEL_CHOICES:
            self.assertIn(name, model_action.choices)

    def test_accuracy_summary_is_finite_and_correct(self):
        mean_acc = summarize_accuracy([0.25, 0.5, 1.0])
        self.assertTrue(math.isfinite(mean_acc))
        self.assertAlmostEqual(mean_acc, (0.25 + 0.5 + 1.0) / 3.0)

    def test_cli_accepts_dataset_specific_grouping_override(self):
        parser = create_arg_parser()
        args = parser.parse_args(
            ["--dataset", "2a", "--model", "eegnet", "--augment", "none", "--validation-group-by", "run"]
        )
        self.assertEqual(args.mode, "cross_subject")
        self.assertEqual(args.augment, "none")
        self.assertEqual(args.validation_group_by, "run")

    def test_core_modules_import(self):
        import src.eeg_bci.data as data_module
        import src.eeg_bci.train as train_module
        import src.eeg_bci.utils as utils_module

        self.assertIsNotNone(data_module)
        self.assertIsNotNone(train_module)
        self.assertIsNotNone(utils_module)


class ClassicalBaselineTests(unittest.TestCase):
    def test_default_csp_component_count_is_explicitly_four(self):
        estimator = build_classical_estimator("csp_lda", n_channels=8, n_classes=4)
        self.assertEqual(DEFAULT_CSP_COMPONENTS, 4)
        self.assertEqual(estimator.named_steps["csp"].nfilter, 4)

    def test_invalid_csp_component_count_is_rejected(self):
        with self.assertRaises(ValueError):
            build_classical_estimator("csp_lda", n_channels=8, n_classes=4, csp_components=0)

    def test_csp_components_match_channel_count_when_equal(self):
        estimator = build_classical_estimator("csp_lda", n_channels=3, n_classes=4, csp_components=3)
        self.assertEqual(estimator.named_steps["csp"].nfilter, 3)

    def test_csp_components_clip_to_channel_count(self):
        estimator = build_classical_estimator("csp_lda", n_channels=3, n_classes=4, csp_components=6)
        self.assertEqual(estimator.named_steps["csp"].nfilter, 3)

    def test_csp_component_count_requires_positive_channels(self):
        with self.assertRaises(ValueError):
            _effective_csp_components(1, 0)

    def test_mi_loader_uses_explicit_8_to_32_hz_band_without_download(self):
        captured_kwargs = {}

        fake_trials = EEGTrials(
            X=np.zeros((2, 2, 4), dtype=np.float32),
            y=np.array([0, 1], dtype=np.int64),
            metadata={
                "subject": np.asarray(["S1", "S1"]),
                "session": np.asarray(["0train", "1test"]),
                "run": np.asarray(["run_1", "run_2"]),
            },
        ).validate()

        with mock.patch("src.eeg_bci.train.load_mi_data", return_value=fake_trials) as load_mock:
            trials = _load_trials("2b", resample=128)

        captured_kwargs = load_mock.call_args.kwargs
        self.assertEqual(captured_kwargs["fmin"], MOTOR_IMAGERY_FMIN_HZ)
        self.assertEqual(captured_kwargs["fmax"], MOTOR_IMAGERY_FMAX_HZ)
        self.assertEqual(captured_kwargs["resample"], 128)
        self.assertIs(trials, fake_trials)

    def test_csp_lda_supports_multiclass_predictions(self):
        trials = make_classical_trials(n_subjects=3, n_classes=4, trials_per_subject=12, n_channels=8, n_times=32)
        train_idx = tuple(range(0, 24))
        test_idx = tuple(range(24, 36))
        estimator = build_classical_estimator("csp_lda", n_channels=8, n_classes=4, csp_components=4)
        estimator.fit(trials.X[list(train_idx)], trials.y[list(train_idx)])
        preds = estimator.predict(trials.X[list(test_idx)])
        self.assertEqual(preds.shape, (len(test_idx),))
        self.assertTrue(set(np.unique(preds)).issubset(set(trials.y[list(train_idx)])))

    def test_riemann_mdm_supports_binary_predictions(self):
        trials = make_classical_trials(n_subjects=3, n_classes=2, trials_per_subject=10, n_channels=6, n_times=24)
        train_idx = tuple(range(0, 20))
        test_idx = tuple(range(20, 30))
        estimator = build_classical_estimator("riemann_mdm", n_channels=6, n_classes=2)
        estimator.fit(trials.X[list(train_idx)], trials.y[list(train_idx)])
        preds = estimator.predict(trials.X[list(test_idx)])
        self.assertEqual(preds.shape, (len(test_idx),))
        self.assertTrue(set(np.unique(preds)).issubset(set(trials.y[list(train_idx)])))

    def test_fit_predict_wrapper_uses_split_indices_without_leakage(self):
        trials = make_classical_trials(n_subjects=3, n_classes=2, trials_per_subject=6, n_channels=4, n_times=16)
        split = next(get_loso_subject_splits(trials))

        class RecordingEstimator:
            def __init__(self):
                self.fit_X = None
                self.fit_y = None
                self.predict_shapes = []

            def fit(self, X, y):
                self.fit_X = np.array(X, copy=True)
                self.fit_y = np.array(y, copy=True)
                return self

            def predict(self, X):
                self.predict_shapes.append(X.shape)
                return np.zeros(X.shape[0], dtype=np.int64)

        estimator = RecordingEstimator()
        val_preds, test_preds = fit_predict_classical_split(estimator, trials, split)
        np.testing.assert_array_equal(estimator.fit_X, trials.X[list(split.train_idx)])
        np.testing.assert_array_equal(estimator.fit_y, trials.y[list(split.train_idx)])
        self.assertEqual(estimator.predict_shapes, [trials.X[list(split.val_idx)].shape, trials.X[list(split.test_idx)].shape])
        self.assertEqual(val_preds.shape, (len(split.val_idx),))
        self.assertEqual(test_preds.shape, (len(split.test_idx),))


if __name__ == "__main__":
    unittest.main()
