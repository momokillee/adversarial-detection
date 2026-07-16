"""Simple CNN-based adversarial sample detector."""

import torch
import torch.nn as nn
from torchvision import models


class AdversarialDetector(nn.Module):
    """Binary classifier: clean (0) vs adversarial (1)."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in backbone.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.classifier = nn.Linear(backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
