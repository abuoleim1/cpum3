import torch
import torch.nn as nn
import torch.nn.functional as F
from const import batch_size

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, batch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(batch_size, batch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_size),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24

            # Block 2
            nn.Conv2d(batch_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 12x12 -> 6x6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x