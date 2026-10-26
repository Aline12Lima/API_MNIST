from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        return self.fc1(x)


def carregar_modelo():
    caminho = Path(__file__).resolve().parent / "MulticlassCNN.pth"
    model = CNN(in_channels=1, num_classes=10)
    state_dict = torch.load(caminho, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
