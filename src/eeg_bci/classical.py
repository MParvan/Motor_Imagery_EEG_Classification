from typing import Tuple

import numpy as np

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline


CLASSICAL_MODEL_CHOICES: Tuple[str, ...] = ("csp_lda", "riemann_mdm")
DEFAULT_CSP_COMPONENTS = 4


def _effective_csp_components(component_count: int, n_channels: int) -> int:
    if component_count < 1:
        raise ValueError("csp_components must be at least 1.")
    if n_channels < 1:
        raise ValueError("CSP requires at least one channel.")
    return min(int(component_count), n_channels)


def build_classical_estimator(
    name: str,
    n_channels: int,
    n_classes: int,
    csp_components: int = DEFAULT_CSP_COMPONENTS,
) -> Pipeline:
    """Build a leakage-safe classical motor-imagery estimator.

    CSP + LDA uses OAS covariance estimation before the spatial filter. The
    classical baselines operate on the fixed MOABB MotorImagery epochs directly
    and estimate covariance per trial; they do not apply the neural z-score
    scaler.
    """

    name = name.lower()
    if n_classes < 2:
        raise ValueError("Classical baselines require at least two classes.")
    if name == "csp_lda":
        nfilter = _effective_csp_components(csp_components, n_channels)
        return make_pipeline(
            Covariances(estimator="oas"),
            CSP(nfilter=nfilter, metric="euclid", log=True),
            LinearDiscriminantAnalysis(solver="svd"),
        )
    if name == "riemann_mdm":
        return make_pipeline(
            Covariances(estimator="oas"),
            MDM(metric="riemann"),
        )
    raise ValueError(f"Unknown classical model: {name}")


def fit_classical_estimator(estimator: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Fit a classical estimator on training trials only."""

    return estimator.fit(X_train, y_train)


def predict_classical_estimator(estimator: Pipeline, X_eval: np.ndarray) -> np.ndarray:
    """Predict labels for held-out trials using a fitted classical estimator."""

    return estimator.predict(X_eval)


def fit_predict_classical_split(estimator: Pipeline, trials, split):
    """Fit on the training portion of a split and predict validation/test labels."""

    train_X = trials.X[list(split.train_idx)]
    train_y = trials.y[list(split.train_idx)]
    val_X = trials.X[list(split.val_idx)]
    test_X = trials.X[list(split.test_idx)]
    fit_classical_estimator(estimator, train_X, train_y)
    val_preds = predict_classical_estimator(estimator, val_X)
    test_preds = predict_classical_estimator(estimator, test_X)
    return val_preds, test_preds
