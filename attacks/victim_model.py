"""Pretrained ResNet18 victim classifier for generating attacks."""

import torch
from torchvision.models import ResNet18_Weights, resnet18


def load_victim(device: torch.device, image_size: int = 64) -> torch.nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    return model
