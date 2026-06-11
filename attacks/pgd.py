"""Projected Gradient Descent (PGD) attack."""

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


def pgd_attack(
    image: torch.Tensor,
    model: torch.nn.Module,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    perturbed = image.clone().detach()
    orig = image.clone().detach()

    for _ in range(steps):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = F.cross_entropy(output, output.argmax(dim=1))
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            perturbed = perturbed + alpha * perturbed.grad.sign()
            perturbed = torch.max(torch.min(perturbed, orig + epsilon), orig - epsilon)
            perturbed = torch.clamp(perturbed, 0, 1).detach()

    return perturbed


def main():
    parser = argparse.ArgumentParser(description="Generate PGD adversarial samples")
    parser.add_argument("--input-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/adversarial"))
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
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
        adv = pgd_attack(image, model, args.epsilon, args.alpha, args.steps)
        out_path = args.output_dir / f"pgd_{img_path.name}"
        save_tensor_image(adv, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
