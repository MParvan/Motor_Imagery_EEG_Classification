import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class ZScoreScalerTorch:
    """Channel-wise z-score using sklearn StandardScaler fitted on training set only."""
    def __init__(self):
        self.scaler = None

    def fit(self, X: np.ndarray):
        # X shape: (n_trials, n_channels, n_times)
        n_trials, n_channels, n_times = X.shape
        self.scaler = []
        for ch in range(n_channels):
            sc = StandardScaler()
            sc.fit(X[:, ch, :].reshape(n_trials, -1))
            self.scaler.append(sc)
        return self

    def transform(self, X: np.ndarray):
        n_trials, n_channels, n_times = X.shape
        X_out = np.empty_like(X, dtype=np.float32)
        for ch in range(n_channels):
            sc = self.scaler[ch]
            X_out[:, ch, :] = sc.transform(X[:, ch, :].reshape(n_trials, -1)).reshape(n_trials, n_times)
        return X_out.astype(np.float32)