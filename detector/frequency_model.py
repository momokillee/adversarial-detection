"""Frequency-domain CNN detector for clean vs adversarial images."""

import torch
import torch.nn as nn


class FrequencyDetector(nn.Module):
    """Binary classifier: clean (0) vs adversarial (1) using frequency-domain features."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_result = torch.fft.fft2(x, dim=(-2, -1))
        magnitude = torch.abs(fft_result)
        log_magnitude = torch.log1p(magnitude)

        features = self.features(log_magnitude)
        features = features.view(features.size(0), -1)
        return self.classifier(features).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
