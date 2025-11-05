from typing import Dict, List, Tuple, Optional
import numpy as np
from moabb.datasets import BNCI2014_001, BNCI2014_004
from moabb.paradigms import MotorImagery
from sklearn.model_selection import StratifiedKFold
from .utils import ZScoreScalerTorch
from sklearn.preprocessing import LabelEncoder

def _get_dataset_and_events(name: str):
    name = name.lower()
    if name in ["2a", "bnci2014_001", "bnci2014-001"]:
        return BNCI2014_001(), ["left_hand", "right_hand", "feet", "tongue"]
    elif name in ["2b", "bnci2014_004", "bnci2014-004"]:
        return BNCI2014_004(), ["left_hand", "right_hand"]
    else:
        raise ValueError("Dataset must be '2a' or '2b'")

def load_mi_data(dataset: str, subjects: Optional[List[int]] = None,
                 tmin: float = 0.0, tmax: float = 4.0,
                 fmin: float = 4.0, fmax: float = 38.0,
                 resample: Optional[int] = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups) over selected subjects.
    X shape: (n_trials, n_channels, n_times), y: (n_trials,), groups: subject id per trial.
    """
    ds, events = _get_dataset_and_events(dataset)
    # print(events)
    paradigm = MotorImagery(n_classes=len(events), events=events, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample)
    # print(ds)
    # print(ds.subject_list)
    # print(subjects)
    # X, y, meta = paradigm.get_data(dataset=ds, subjects=ds.subject_list)
    X, y, meta = paradigm.get_data(dataset=ds, subjects=subjects)
    # meta is a DataFrame containing 'subject' column
    groups = meta['subject'].values.astype(int)
    # MOABB returns X as (trials, channels, times)
    X = X.astype('float32')
    # print(y)
    # convert string labels (e.g. "right", "left", ...) to integers
    if y.dtype.kind in {"U", "S", "O"}:
        label_map = {ev: i for i, ev in enumerate(events)}
        try:
            y = np.array([label_map[label] for label in y], dtype=int)
        except KeyError:
            y = LabelEncoder().fit_transform(y).astype(int)
    else:
        y = y.astype(int)
    return X, y, groups

def get_within_subject_splits(groups: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """Yield (train_idx, val_idx) splits per subject with stratified KFold."""
    subjects = np.unique(groups)
    for sub in subjects:
        idx = np.where(groups == sub)[0]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_local, val_local in skf.split(idx, y[idx]):
            yield idx[train_local], idx[val_local], int(sub)

def get_loso_splits(groups: np.ndarray):
    """Yield (train_idx, test_idx) Leave-One-Subject-Out splits."""
    subjects = np.unique(groups)
    for sub in subjects:
        test_idx = np.where(groups == sub)[0]
        train_idx = np.where(groups != sub)[0]
        yield train_idx, test_idx, int(sub)

def zscore_fit_transform_train_test(X, train_idx, test_idx):
    scaler = ZScoreScalerTorch()
    scaler.fit(X[train_idx])
    Xtr = scaler.transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])
    return Xtr, Xte, scaler