"""Fast Gradient Sign Method (FGSM) attack."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from attacks.victim_model import load_victim
from utils.preprocess import load_image_tensor, save_tensor_image

IMAGE_SIZE = 64


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    sign_grad = data_grad.sign()
    perturbed = image + epsilon * sign_grad
    return torch.clamp(perturbed, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Generate FGSM adversarial samples")
    parser.add_argument("--input-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/adversarial"))
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to use. If None, auto-selects GPU if available.")
    args = parser.parse_args()

    # Auto-select device or use user choice
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    model = load_victim(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(args.input_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = load_image_tensor(img_path, IMAGE_SIZE, device)
        image.requires_grad = True

        output = model(image)
        pred = output.argmax(dim=1)
        loss = F.cross_entropy(output, pred)
        model.zero_grad()
        loss.backward()

        adv = fgsm_attack(image, args.epsilon, image.grad.data)
        out_path = args.output_dir / f"fgsm_{img_path.name}"
        save_tensor_image(adv, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
