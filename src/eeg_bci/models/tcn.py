import torch
import torch.nn as nn

class _ResBlock(nn.Module):
    def __init__(self, ch=64, k=3, dilation=1, p=0.1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)

class TCN(nn.Module):
    """
    Simple TCN for EEG. Mix channels first, then temporal dilations.
    Input: (B, C, T)
    """
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 64, p: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        dilations = [1, 2, 4, 8]
        self.blocks = nn.Sequential(*[_ResBlock(hidden, k=3, dilation=d, p=p) for d in dilations])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        if x.ndim == 4:
            # (B,1,C,T) -> (B,C,T)
            x = x[:, 0, :, :]
        x = self.inp(x)
        x = self.blocks(x)
        return self.head(x)
