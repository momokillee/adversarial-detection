"""Dual-domain CNN detector combining pixel and frequency streams."""

import torch
import torch.nn as nn


class DualDomainDetector(nn.Module):
    """Binary classifier using both pixel-domain and frequency-domain features."""

    def __init__(self):
        super().__init__()
        self.pixel_stream = nn.Sequential(
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
        self.freq_stream = nn.Sequential(
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
        self.fusion = nn.Sequential(
            nn.Linear(128 * 8 * 8 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_features = self.pixel_stream(x)
        pixel_features = pixel_features.view(pixel_features.size(0), -1)

        fft_result = torch.fft.fft2(x, dim=(-2, -1))
        magnitude = torch.abs(fft_result)
        log_magnitude = torch.log1p(magnitude)

        freq_features = self.freq_stream(log_magnitude)
        freq_features = freq_features.view(freq_features.size(0), -1)

        combined_features = torch.cat([pixel_features, freq_features], dim=1)
        return self.fusion(combined_features).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
