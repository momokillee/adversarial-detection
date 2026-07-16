"""Frequency-domain detector for 224x224 images."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class FrequencyDetector(nn.Module):
    """Binary classifier for clean vs adversarial images."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128 * 28 * 28, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fft2(x, dim=(-2, -3)).abs()
        x = self.features(x)
        return self.classifier(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
