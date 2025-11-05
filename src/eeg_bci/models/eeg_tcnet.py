# src/eeg_bci/models/eeg_tcnet.py
import torch
import torch.nn as nn

class SepConv1d(nn.Module):
    """Depthwise-separable Conv1d: depthwise (groups=in_ch) + pointwise."""
    def __init__(self, in_ch, out_ch, k=15, dilation=1, bias=False):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad,
                            dilation=dilation, groups=in_ch, bias=bias)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))

class TCBlock(nn.Module):
    """Residual temporal block using depthwise-separable Conv1d."""
    def __init__(self, ch, k=15, dilation=1, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            SepConv1d(ch, ch, k=k, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
            nn.ELU(),
            nn.Dropout(p),
            SepConv1d(ch, ch, k=k, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.net(x) + x)

class EEGTCNet(nn.Module):
    """
    EEG-TCNet style:
      - Mix channels -> hidden with 1x1 temporal conv
      - Stack dilated temporal residual blocks (depthwise-separable)
      - Global avg pool -> FC
    Input: (B,C,T)
    """
    def __init__(self, n_channels: int, n_classes: int,
                 hidden: int = 64, k: int = 15,
                 dilations=(1, 2, 4, 8), p: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
        )
        blocks = []
        for d in dilations:
            blocks.append(TCBlock(hidden, k=k, dilation=d, p=p))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # x: (B,C,T) or (B,1,C,T)
        if x.ndim == 4:
            x = x[:, 0, :, :]
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
