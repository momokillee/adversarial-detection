"""PIL-based image loading (no torchvision / numpy required)."""

from pathlib import Path

import torch
from PIL import Image


def pil_to_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    w, h = img.size
    buf = img.tobytes()
    t = torch.tensor(list(buf), dtype=torch.float32, device=device)
    t = t.reshape(h, w, 3).permute(2, 0, 1) / 255.0
    return t.unsqueeze(0)


def load_image_tensor(
    path: Path,
    size: int,
    device: torch.device,
    normalize: bool = False,
) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = pil_to_tensor(image, size, device)

    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    t = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    _, h, w = t.shape
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (
                int(t[0, y, x].item() * 255),
                int(t[1, y, x].item() * 255),
                int(t[2, y, x].item() * 255),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
