# src/eeg_bci/models/mbma_ciac_lite.py
import torch
import torch.nn as nn

def safe_log(t, eps=1e-6):
    min_t = torch.as_tensor(eps, device=t.device, dtype=t.dtype)
    return torch.log(torch.maximum(t, min_t))

class CNNBranch(nn.Module):
    """Shallow-like temporal + spatial depthwise + log-power pooling to an embedding."""
    def __init__(self, n_channels, f_time=40, k_time=25, pool=75, stride=15, p=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, f_time, kernel_size=(1, k_time), padding=(0, k_time // 2), bias=False)
        self.spatial  = nn.Conv2d(f_time, f_time, kernel_size=(n_channels, 1), groups=f_time, bias=False)
        self.bn = nn.BatchNorm2d(f_time)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool), stride=(1, stride))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # x: (B,1,C,T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = safe_log(x)
        # global average to (B, f_time)
        x = x.mean(dim=[2,3])
        x = self.drop(x)
        return x  # (B, f_time)

class TCNBranch(nn.Module):
    """Compact TCN to an embedding vector."""
    def __init__(self, n_channels, hidden=64, k=7, dilations=(1,2,4,8), p=0.2):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=k, padding=k//2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
        )
        blocks = []
        for d in dilations:
            pad = (k - 1) * d // 2
            blocks += [
                nn.Conv1d(hidden, hidden, kernel_size=k, padding=pad, dilation=d, groups=hidden, bias=False),
                nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ELU(),
                nn.Dropout(p),
            ]
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, 0, :, :]
        x = self.inp(x)
        x = self.tcn(x)
        x = self.head(x)   # (B, hidden)
        return x

class SEGate(nn.Module):
    """Squeeze-Excitation style gating over concatenated embedding."""
    def __init__(self, dim, r=4):
        super().__init__()
        mid = max(8, dim // r)
        self.net = nn.Sequential(
            nn.Linear(dim, mid),
            nn.ELU(),
            nn.Linear(mid, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.net(x)
        return x * w

class MBMA_CIAC_Lite(nn.Module):
    """
    Two-branch fusion: Shallow-like CNN branch + TCN branch + SE attention, then FC.
    Stand-in for MBMANet/CIACNet flavor (multi-branch + attention/TCN).
    """
    def __init__(self, n_channels: int, n_classes: int,
                 f_time: int = 40, hidden_tcn: int = 64, p: float = 0.3):
        super().__init__()
        self.cnn = CNNBranch(n_channels, f_time=f_time, p=p)
        self.tcn = TCNBranch(n_channels, hidden=hidden_tcn, p=p)
        self.se = SEGate(f_time + hidden_tcn, r=4)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(f_time + hidden_tcn),
            nn.Dropout(p),
            nn.Linear(f_time + hidden_tcn, n_classes)
        )

    def forward(self, x):
        if x.ndim == 3:
            x2d = x[:, None, :, :]
            x1d = x
        else:
            x2d = x
            x1d = x[:, 0, :, :]
        h_cnn = self.cnn(x2d)     # (B, f_time)
        h_tcn = self.tcn(x1d)     # (B, hidden_tcn)
        h = torch.cat([h_cnn, h_tcn], dim=1)
        h = self.se(h)
        return self.fc(h)
