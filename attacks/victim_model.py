"""Lightweight victim classifier for generating attacks (torch only)."""

from pathlib import Path

import torch
import torch.nn as nn


class VictimCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_victim(device: torch.device, image_size: int = 64) -> VictimCNN:
    model = VictimCNN(num_classes=10).to(device)
    weights_path = Path(__file__).resolve().parent.parent / "models" / "victim.pt"

    if not weights_path.exists():
        raise RuntimeError("Victim model not trained yet - run scripts/train_victim.py first")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model