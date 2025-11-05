import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """EEGNet-v4 style (simplified, with adaptive pooling for robustness).
    Input shape: (B, 1, C, T)
    """
    def __init__(self, n_channels: int, n_classes: int, 
                 F1: int = 8, D: int = 2, kernel_length: int = 64, dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise temporal
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(F2, n_classes)
        )

    def forward(self, x):
        # x: (B, C, T) -> (B,1,C,T)
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x