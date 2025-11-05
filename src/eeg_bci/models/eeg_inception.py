# src/eeg_bci/models/eeg_inception.py
import torch
import torch.nn as nn

class InceptionBlock2D(nn.Module):
    """
    Inception over time (width=kernel along T), with optional pool branch.
    Operates on (B, F_in, C, T).
    """
    def __init__(self, in_ch: int, bottleneck: int = 32,
                 kernel_sizes=(9, 19, 39), pool_kernel=3, activation="ELU", dropout=0.0):
        super().__init__()
        act = nn.ELU() if activation.upper() == "ELU" else nn.ReLU(inplace=True)

        self.bottleneck = nn.Conv2d(in_ch, bottleneck, kernel_size=(1, 1), bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(bottleneck, bottleneck, kernel_size=(1, k), padding=(0, k // 2), bias=False)
            for k in kernel_sizes
        ])
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, 1), padding=(0, pool_kernel // 2)),
            nn.Conv2d(in_ch, bottleneck, kernel_size=(1, 1), bias=False),
        )
        self.bn = nn.BatchNorm2d(bottleneck * (len(kernel_sizes) + 1))
        self.act = act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, F_in, C, T)
        z = []
        b = self.bottleneck(x)
        for conv in self.branches:
            z.append(conv(b))
        z.append(self.pool(x))
        x = torch.cat(z, dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class EEGInception(nn.Module):
    """
    EEG-Inception (compact):
      - 2–3 temporal Inception blocks
      - Spatial depthwise conv across channels (kernel=(C,1))
      - Global pooling + FC
    Input:  (B, C, T) or (B, 1, C, T)
    Output: (B, n_classes)
    """
    def __init__(self, n_channels: int, n_classes: int,
                 stem_filters: int = 32, blocks: int = 3,
                 bottleneck: int = 32, kernel_sizes=(9, 19, 39),
                 dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_filters, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(stem_filters),
            nn.ELU(),
        )
        feats = stem_filters
        mods = []
        for _ in range(blocks):
            mods.append(InceptionBlock2D(feats, bottleneck=bottleneck,
                                         kernel_sizes=kernel_sizes, dropout=dropout))
            feats = bottleneck * (len(kernel_sizes) + 1)
        self.incepts = nn.Sequential(*mods)

        # Spatial depthwise conv to mix across channels (height = C)
        self.spatial = nn.Sequential(
            nn.Conv2d(feats, feats, kernel_size=(n_channels, 1), groups=feats, bias=False),
            nn.BatchNorm2d(feats),
            nn.ELU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feats, n_classes)
        )

    def forward(self, x):
        # Accept (B, C, T)
        if x.ndim == 3:
            x = x[:, None, :, :]  # (B,1,C,T)
        x = self.stem(x)
        x = self.incepts(x)
        x = self.spatial(x)
        x = self.head(x)
        return x
