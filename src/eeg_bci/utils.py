from typing import Optional

import numpy as np


def set_seed(seed: int = 42):
    import os
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ZScoreScalerTorch:
    """Per-channel z-score over the training set using statistics pooled across trials and time."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, C, T), got {X.shape}.")
        X64 = np.asarray(X, dtype=np.float64)
        self.mean_ = X64.mean(axis=(0, 2))
        std = X64.std(axis=(0, 2))
        self.std_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, X: np.ndarray):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fit before transform.")
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, C, T), got {X.shape}.")
        output_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float32
        normalized = (np.asarray(X, dtype=np.float64) - self.mean_[None, :, None]) / self.std_[None, :, None]
        return normalized.astype(output_dtype, copy=False)
