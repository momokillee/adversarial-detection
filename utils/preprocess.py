"""PIL-based image loading (no torchvision / numpy required)."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    w, h = img.size
    buf = img.tobytes()
    t = torch.tensor(list(buf), dtype=torch.float32, device=device)
    t = t.reshape(h, w, 3).permute(2, 0, 1) / 255.0
    return t.unsqueeze(0)


def load_image_tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    return pil_to_tensor(Image.open(path), size, device)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    t = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
