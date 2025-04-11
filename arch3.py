import torch
import torch.nn as nn
import torch.nn.functional as F
from const import batch_size

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Channel and Spatial Attention Module (CBAM Lite)
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAMBlock(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return Mish()(out + residual)

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, batch_size, 3, padding=1),
            nn.BatchNorm2d(batch_size),
            Mish(),
        )
        self.resblock1 = ResidualBlock(batch_size, 128)
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x48 → 24x24

        self.resblock2 = ResidualBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)  # 24x24 → 12x12

        self.resblock3 = ResidualBlock(256, 512)
        self.pool3 = nn.MaxPool2d(2, 2)  # 12x12 → 6x6

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 6x6 → 1x1
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(self.resblock1(x))
        x = self.pool2(self.resblock2(x))
        x = self.pool3(self.resblock3(x))
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, 512, 1, 1] -> [B, 512]
        x = self.dropout(x)
        x = self.fc(x)
        return x
