import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    """Simplified ShallowConvNet (Schirrmeister et al., 2017).
    Input shape: (B, 1, C, T)
    """
    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 40, kernel_size=(1, 25), padding=(0, 12), bias=False)
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=40, bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(40, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        x = x.mean(dim=[2,3])  # global average over C and T dims
        x = self.fc(x)
        return x