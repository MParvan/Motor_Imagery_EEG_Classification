import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    """
    Simplified DeepConvNet for MI-BCI.
    Input: (B, C, T) or (B, 1, C, T)
    """
    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        def conv_block(cin, cout, k_t=10, pool=3, p=0.5):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=(1, k_t), padding=(0, k_t//2), bias=False),
                nn.BatchNorm2d(cout),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, pool), stride=(1, pool)),
                nn.Dropout(p),
            )

        # Block 1: temporal then depthwise spatial
        self.temporal = nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.spatial  = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), groups=25, bias=False)
        self.bn1      = nn.BatchNorm2d(25)
        self.act1     = nn.ELU()
        self.pool1    = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1    = nn.Dropout(dropout)

        # Deeper temporal blocks
        self.block2 = conv_block(25, 50, k_t=10, pool=3, p=dropout)
        self.block3 = conv_block(50, 100, k_t=10, pool=3, p=dropout)
        self.block4 = conv_block(100, 200, k_t=10, pool=3, p=dropout)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(200, n_classes)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, None, :, :]  # (B,1,C,T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)
