import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REQUIRED_METADATA_FIELDS = ("subject", "session", "run")


def _get_dataset_and_events(name: str):
    from moabb.datasets import BNCI2014_001, BNCI2014_004

    name = name.lower()
    if name in ["2a", "bnci2014_001", "bnci2014-001"]:
        return BNCI2014_001(), ["left_hand", "right_hand", "feet", "tongue"]
    if name in ["2b", "bnci2014_004", "bnci2014-004"]:
        return BNCI2014_004(), ["left_hand", "right_hand"]
    raise ValueError("Dataset must be '2a' or '2b'")


def _metadata_to_dict(metadata_frame) -> Dict[str, np.ndarray]:
    metadata: Dict[str, np.ndarray] = {}
    for column in metadata_frame.columns:
        metadata[column] = np.asarray(metadata_frame.iloc[:, metadata_frame.columns.get_loc(column)].to_numpy())
    return metadata


def _normalize_values(values: Iterable[object]) -> Tuple[str, ...]:
    return tuple(str(value) for value in values)


def _indices_from_mask(mask: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(index) for index in np.flatnonzero(mask.astype(bool)))


def _unique_as_strings(values: Sequence[object]) -> Tuple[str, ...]:
    return tuple(str(value) for value in np.unique(np.asarray(values)))


@dataclass(frozen=True)
class DevelopmentSessionConfig:
    """Normalized development and held-out test session identifiers for a supported MOABB dataset."""

    dataset_name: str
    development_sessions: Tuple[str, ...]
    test_sessions: Tuple[str, ...]

    @classmethod
    def from_dict(cls, dataset_name: str, mapping: Mapping[str, Sequence[object]]) -> "DevelopmentSessionConfig":
        missing = {"development_sessions", "test_sessions"} - set(mapping)
        if missing:
            raise ValueError(f"Session config is missing keys: {sorted(missing)}")
        return cls(
            dataset_name=dataset_name,
            development_sessions=_normalize_values(mapping["development_sessions"]),
            test_sessions=_normalize_values(mapping["test_sessions"]),
        )

    @classmethod
    def from_json(cls, dataset_name: str, raw_json: str) -> "DevelopmentSessionConfig":
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Session config JSON must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Session config JSON must be an object.")
        return cls.from_dict(dataset_name=dataset_name, mapping=payload)

    def validate_disjoint(self) -> None:
        overlap = set(self.development_sessions) & set(self.test_sessions)
        if overlap:
            raise ValueError(f"Development and test sessions must be disjoint, but overlap on {sorted(overlap)}.")


@dataclass(frozen=True)
class ValidationSplitConfig:
    group_by: str
    val_group_count: int = 1
    seed: int = 42

    def validate(self) -> "ValidationSplitConfig":
        if self.val_group_count < 1:
            raise ValueError("val_group_count must be at least 1.")
        if not self.group_by:
            raise ValueError("group_by must be a non-empty metadata field name.")
        return self


DATASET_SESSION_CONFIGS: Dict[str, DevelopmentSessionConfig] = {
    "2a": DevelopmentSessionConfig(
        dataset_name="2a",
        development_sessions=("0train",),
        test_sessions=("1test",),
    ),
    "2b": DevelopmentSessionConfig(
        dataset_name="2b",
        development_sessions=("0train", "1train", "2train"),
        test_sessions=("3test", "4test"),
    ),
}

DATASET_VALIDATION_GROUP_BY_DEFAULTS: Dict[str, str] = {
    "2a": "run",
    "2b": "session",
}


@dataclass(frozen=True)
class EEGTrials:
    X: np.ndarray
    y: np.ndarray
    metadata: Dict[str, np.ndarray]

    def validate(self) -> "EEGTrials":
        if self.X.ndim != 3:
            raise ValueError(f"Expected X shape (N, C, T), got {self.X.shape}.")
        if self.y.ndim != 1:
            raise ValueError(f"Expected y shape (N,), got {self.y.shape}.")
        n_trials = self.X.shape[0]
        if len(self.y) != n_trials:
            raise ValueError(f"X and y length mismatch: {n_trials} vs {len(self.y)}.")
        for field in REQUIRED_METADATA_FIELDS:
            if field not in self.metadata:
                raise ValueError(f"Missing required metadata field: {field}.")
        for field, values in self.metadata.items():
            array_values = np.asarray(values)
            if array_values.shape[0] != n_trials:
                raise ValueError(
                    f"Metadata field '{field}' length mismatch: expected {n_trials}, got {array_values.shape[0]}."
                )
        return self

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    def values_for(self, field: str, indices: Sequence[int]) -> Tuple[str, ...]:
        if field not in self.metadata:
            raise KeyError(f"Unknown metadata field: {field}")
        positional_index = np.asarray(indices, dtype=int)
        return _unique_as_strings(self.metadata[field][positional_index])

    @property
    def subjects(self) -> Tuple[str, ...]:
        return self.values_for("subject", np.arange(self.n_trials))

    @property
    def sessions(self) -> Tuple[str, ...]:
        return self.values_for("session", np.arange(self.n_trials))

    @property
    def runs(self) -> Tuple[str, ...]:
        return self.values_for("run", np.arange(self.n_trials))


@dataclass(frozen=True)
class SplitIndices:
    train_idx: Tuple[int, ...]
    val_idx: Tuple[int, ...]
    test_idx: Tuple[int, ...]
    protocol: str
    subject: Optional[str] = None
    val_subject: Optional[str] = None
    val_groups: Tuple[str, ...] = ()

    def validate(self, trials: EEGTrials) -> "SplitIndices":
        n_trials = trials.n_trials
        for role_name, idx in [("train", self.train_idx), ("val", self.val_idx), ("test", self.test_idx)]:
            if len(idx) == 0:
                raise ValueError(f"{role_name} indices are empty.")
            if len(set(idx)) != len(idx):
                raise ValueError(f"{role_name} indices contain duplicates.")
            if any(index < 0 or index >= n_trials for index in idx):
                raise ValueError(f"{role_name} indices are out of bounds.")
        overlaps = {
            "train/val": set(self.train_idx) & set(self.val_idx),
            "train/test": set(self.train_idx) & set(self.test_idx),
            "val/test": set(self.val_idx) & set(self.test_idx),
        }
        for pair, values in overlaps.items():
            if values:
                raise ValueError(f"Split overlap detected for {pair}: {sorted(values)}")
        return self

    def metadata_values(self, trials: EEGTrials, role: str, field: str) -> Tuple[str, ...]:
        return trials.values_for(field, self._indices_for_role(role))

    def _indices_for_role(self, role: str) -> Tuple[int, ...]:
        if role == "train":
            return self.train_idx
        if role == "val":
            return self.val_idx
        if role == "test":
            return self.test_idx
        raise ValueError(f"Unknown role: {role}")


def load_mi_data(
    dataset: str,
    subjects: Optional[List[int]] = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    fmin: float = 4.0,
    fmax: float = 38.0,
    resample: Optional[int] = 128,
) -> EEGTrials:
    from moabb.paradigms import MotorImagery

    ds, events = _get_dataset_and_events(dataset)
    paradigm = MotorImagery(
        n_classes=len(events),
        events=events,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        resample=resample,
    )
    X, y, meta = paradigm.get_data(dataset=ds, subjects=subjects)
    metadata = _metadata_to_dict(meta)
    X = X.astype(np.float32, copy=False)
    if y.dtype.kind in {"U", "S", "O"}:
        label_map = {event: index for index, event in enumerate(events)}
        try:
            y = np.array([label_map[label] for label in y], dtype=np.int64)
        except KeyError:
            from sklearn.preprocessing import LabelEncoder

            y = LabelEncoder().fit_transform(y).astype(np.int64)
    else:
        y = y.astype(np.int64, copy=False)
    return EEGTrials(X=X, y=y, metadata=metadata).validate()


def get_dataset_session_config(dataset: str) -> DevelopmentSessionConfig:
    key = dataset.lower()
    if key not in DATASET_SESSION_CONFIGS:
        raise ValueError(f"No dataset session configuration found for: {dataset}")
    return DATASET_SESSION_CONFIGS[key]


def get_default_validation_group_by(dataset: str) -> str:
    key = dataset.lower()
    if key not in DATASET_VALIDATION_GROUP_BY_DEFAULTS:
        raise ValueError(f"No default validation grouping found for: {dataset}")
    return DATASET_VALIDATION_GROUP_BY_DEFAULTS[key]


def get_development_and_test_indices(
    trials: EEGTrials,
    subject: str,
    session_config: DevelopmentSessionConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    session_config.validate_disjoint()
    subject_values = np.asarray(trials.metadata["subject"]).astype(str)
    session_values = np.asarray(trials.metadata["session"]).astype(str)
    known_sessions = set(_unique_as_strings(session_values))
    required_sessions = set(session_config.development_sessions + session_config.test_sessions)
    missing_sessions = sorted(required_sessions - known_sessions)
    if missing_sessions:
        raise ValueError(f"Session config references sessions not present in metadata: {missing_sessions}.")
    subject_mask = subject_values == str(subject)
    development_idx = _indices_from_mask(subject_mask & np.isin(session_values, session_config.development_sessions))
    test_idx = _indices_from_mask(subject_mask & np.isin(session_values, session_config.test_sessions))
    if not development_idx:
        raise ValueError(f"No development trials found for subject {subject}.")
    if not test_idx:
        raise ValueError(f"No test trials found for subject {subject}.")
    return development_idx, test_idx


def split_development_train_validation(
    trials: EEGTrials,
    development_idx: Sequence[int],
    config: ValidationSplitConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[str, ...]]:
    config = config.validate()
    if config.group_by not in trials.metadata:
        raise KeyError(f"Unknown metadata field for group split: {config.group_by}")
    dev_idx = np.asarray(development_idx, dtype=int)
    group_values = np.asarray(trials.metadata[config.group_by])[dev_idx].astype(str)
    unique_groups = np.unique(group_values)
    if config.val_group_count >= len(unique_groups):
        raise ValueError(
            f"Validation group count {config.val_group_count} must be smaller than the number of available groups {len(unique_groups)}."
        )
    ordered_groups = np.sort(unique_groups)
    rng = np.random.default_rng(config.seed)
    permuted_groups = ordered_groups[rng.permutation(len(ordered_groups))]
    selected_val_groups = tuple(str(group) for group in permuted_groups[: config.val_group_count])
    val_mask = np.isin(group_values, selected_val_groups)
    train_mask = ~val_mask
    train_idx = tuple(int(index) for index in dev_idx[train_mask])
    val_idx = tuple(int(index) for index in dev_idx[val_mask])
    if not train_idx or not val_idx:
        raise ValueError("Development train/validation split produced an empty role.")
    return train_idx, val_idx, selected_val_groups


def get_subject_specific_splits(
    trials: EEGTrials,
    dataset: str,
    validation_config: ValidationSplitConfig,
    session_config: Optional[DevelopmentSessionConfig] = None,
) -> Iterator[SplitIndices]:
    resolved_config = session_config or get_dataset_session_config(dataset)
    subject_values = np.asarray(trials.metadata["subject"]).astype(str)
    for subject in _unique_as_strings(subject_values):
        development_idx, test_idx = get_development_and_test_indices(
            trials, subject=subject, session_config=resolved_config
        )
        train_idx, val_idx, val_groups = split_development_train_validation(
            trials,
            development_idx=development_idx,
            config=validation_config,
        )
        split = SplitIndices(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            protocol="within_subject_development_vs_final_test",
            subject=subject,
            val_groups=val_groups,
        )
        yield split.validate(trials)


def get_loso_subject_splits(trials: EEGTrials) -> Iterator[SplitIndices]:
    subject_values = np.asarray(trials.metadata["subject"]).astype(str)
    subjects = list(_unique_as_strings(subject_values))
    if len(subjects) < 3:
        raise ValueError("LOSO with grouped validation requires at least three subjects.")
    for index, test_subject in enumerate(subjects):
        val_subject = subjects[(index + 1) % len(subjects)]
        train_subjects = tuple(subject for subject in subjects if subject not in {test_subject, val_subject})
        split = SplitIndices(
            train_idx=_indices_from_mask(np.isin(subject_values, train_subjects)),
            val_idx=_indices_from_mask(subject_values == val_subject),
            test_idx=_indices_from_mask(subject_values == test_subject),
            protocol="cross_subject_loso",
            subject=test_subject,
            val_subject=val_subject,
        )
        yield split.validate(trials)
