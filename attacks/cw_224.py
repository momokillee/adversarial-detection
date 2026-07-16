"""Generate Carlini-Wagner adversarial examples on 224x224 images."""

import os
import sys
import time
from pathlib import Path
import argparse

import torch
import torchattacks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.victim_model import load_victim
from utils.preprocess import load_image_tensor, save_tensor_image

IMAGE_SIZE = 224


def main():
    parser = argparse.ArgumentParser(description="Generate CW adversarial samples at 224x224")
    parser.add_argument("--input-dir", type=Path, default=Path("data/diverse_clean_224"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cw_adversarial_224"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = load_victim(device)
    model.eval()

    attacker = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.input_dir.glob("*"))
    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ][: args.limit]

    start_time = time.time()

    for idx, img_path in enumerate(image_paths, 1):
        image = load_image_tensor(img_path, IMAGE_SIZE, device)

        with torch.no_grad():
            logits = model(image)
            labels = logits.argmax(dim=1)

        adv = attacker(image, labels)

        out_path = args.output_dir / f"cw_{img_path.name}"
        save_tensor_image(adv, out_path)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(image_paths)} images")

    elapsed = time.time() - start_time
    print(f"Generated {len(image_paths)} CW adversarial images")
    print(f"Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
