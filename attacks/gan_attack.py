"""Simple GAN-style adversarial generator for Run 3 experiments."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator that learns a bounded perturbation for clean images."""

    def __init__(self, epsilon: float = 0.15):
        super().__init__()
        self.epsilon = epsilon
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        perturbation = x * self.epsilon
        return torch.clamp(residual + perturbation, 0.0, 1.0)
