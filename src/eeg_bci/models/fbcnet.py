# src/eeg_bci/models/fbcnet.py
import torch
import torch.nn as nn

def safe_log(t, eps=1e-6):
    min_t = torch.as_tensor(eps, device=t.device, dtype=t.dtype)
    return torch.log(torch.maximum(t, min_t))

class _FBCBranch(nn.Module):
    """
    One filter-bank branch:
      temporal conv -> spatial depthwise -> BN -> square -> var over time -> log
    Works on (B, 1, C, T); outputs (B, F) feature vector (after pooling).
    """
    def __init__(self, n_channels: int, temporal_filters: int, k_time: int, dropout: float = 0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, k_time), padding=(0, k_time // 2), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(temporal_filters, temporal_filters, kernel_size=(n_channels, 1),
                      groups=temporal_filters, bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,1,C,T)
        x = self.temporal(x)     # (B,F,C,T)
        x = self.spatial(x)      # (B,F,1,T)
        x = x ** 2               # power
        # variance over time dimension
        var = x.var(dim=3, unbiased=False)  # (B, F, 1)
        logvar = safe_log(var).squeeze(2)   # (B, F)
        logvar = self.drop(logvar)
        return logvar

class FBCNet(nn.Module):
    """
    Simplified FBCNet:
      Multiple temporal branches with different kernel sizes emulate filter banks.
      Spatial depthwise + log-variance pooling per branch, then concat + FC.
    """
    def __init__(self, n_channels: int, n_classes: int,
                 branches=(16, 32, 64, 128),  # kernel sizes over time
                 filters_per_branch: int = 16, dropout: float = 0.3):
        super().__init__()
        self.branches = nn.ModuleList([
            _FBCBranch(n_channels, filters_per_branch, k_time=k, dropout=dropout)
            for k in branches
        ])
        feat_dim = filters_per_branch * len(branches)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, None, :, :]  # (B,1,C,T)
        feats = [b(x) for b in self.branches]     # list of (B, F)
        h = torch.cat(feats, dim=1)               # (B, F * n_branches)
        return self.classifier(h)
