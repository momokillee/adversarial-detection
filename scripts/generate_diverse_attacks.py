"""Generate FGSM and PGD adversarial examples for diverse clean images."""

import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.victim_model import load_victim
from utils.preprocess import load_image_tensor, save_tensor_image

IMAGE_SIZE = 224
BATCH_SIZE = 32
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}
INPUT_DIR = Path("data/diverse_clean_224")
OUTPUT_DIR = Path("data/diverse_adversarial_224")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_batch(paths: List[Path], image_size: int, device: torch.device) -> torch.Tensor:
    tensors = []
    for path in paths:
        tensor = load_image_tensor(path, image_size, device)
        tensors.append(tensor)
    if not tensors:
        return torch.empty(0, 3, image_size, image_size, device=device)
    return torch.cat(tensors, dim=0)


def save_batch_images(tensor_batch: torch.Tensor, paths: List[Path], prefix: str, output_dir: Path) -> int:
    saved = 0
    for tensor, path in zip(tensor_batch, paths):
        out_path = output_dir / f"{prefix}_{path.name}"
        try:
            save_tensor_image(tensor.unsqueeze(0), out_path)
            saved += 1
        except Exception as exc:
            print(f"Failed saving {out_path}: {exc}")
    return saved


def run_attacks(
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    epsilon: float,
    alpha: float,
    steps: int,
) -> None:
    model = load_victim(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXT)
    total_images = len(image_paths)
    if total_images == 0:
        print("No images found in", input_dir)
        return

    fgsm_count = 0
    pgd_count = 0
    for start in range(0, total_images, BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        try:
            batch = load_batch(batch_paths, IMAGE_SIZE, device)
            batch.requires_grad = True
        except Exception as exc:
            print(f"Failed loading batch starting at {start + 1}: {exc}")
            continue

        output = model(batch)
        preds = output.argmax(dim=1)
        loss = F.cross_entropy(output, preds)
        model.zero_grad()
        loss.backward()

        adv_fgsm = fgsm_attack(batch, epsilon, batch.grad.data)
        adv_pgd = pgd_attack(batch, model, epsilon, alpha, steps)

        fgsm_count += save_batch_images(adv_fgsm, batch_paths, "fgsm", output_dir)
        pgd_count += save_batch_images(adv_pgd, batch_paths, "pgd", output_dir)

        processed = min(start + BATCH_SIZE, total_images)
        if processed % 500 == 0 or processed == total_images:
            print(f"Processed {processed}/{total_images} images (FGSM {fgsm_count}, PGD {pgd_count})")

    print("\nAttack generation complete")
    print(f"  fgsm: {fgsm_count}")
    print(f"  pgd: {pgd_count}")
    print(f"  total images processed: {total_images}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate FGSM and PGD attacks for diverse clean images")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to use. If None, auto-selects GPU if available.")
    args = parser.parse_args()

    if args.device is None:
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    run_attacks(args.input_dir, args.output_dir, device, args.epsilon, args.alpha, args.steps)


if __name__ == "__main__":
    main()
