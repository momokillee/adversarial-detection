"""BIM adversarial attack script."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.victim_model import load_victim
from utils.preprocess import load_image_tensor, save_tensor_image


IMAGE_SIZE = 64


def bim_attack(
    image: torch.Tensor,
    model: torch.nn.Module,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """Generate a BIM adversarial example."""
    perturbed = image.clone().detach()
    original = image.clone().detach()

    for _ in range(steps):
        perturbed.requires_grad_(True)

        outputs = model(perturbed)
        pred = outputs.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(outputs, pred)
        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            gradient = perturbed.grad.detach()
            perturbed = perturbed + alpha * gradient.sign()
            perturbed = torch.max(torch.min(perturbed, original + epsilon), original - epsilon)
            perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()


def main():
    """Generate BIM adversarial images from clean images."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate BIM adversarial samples")
    parser.add_argument("--input-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bim_adversarial"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model = load_victim(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(args.input_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = load_image_tensor(img_path, IMAGE_SIZE, device)
        adv = bim_attack(image, model, args.epsilon, args.alpha, args.steps)

        out_path = args.output_dir / f"bim_{img_path.name}"
        save_tensor_image(adv, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
