import torch
import torch.nn as nn
import torch.nn.functional as F
from const import batch_size

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input is grayscale (1 channel)
        self.conv2 = nn.Conv2d(32, batch_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # After two poolings: 48x48 -> 24x24 -> 12x12
        self.fc1 = nn.Linear(batch_size * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 32, 48, 48]
        x = self.pool(x)           # [B, 32, 24, 24]
        x = F.relu(self.conv2(x))  # [B, batch_size, 24, 24]
        x = self.pool(x)           # [B, batch_size, 12, 12]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x