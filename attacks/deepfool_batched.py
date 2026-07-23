"""Generate DeepFool adversarial examples, BATCHED for real GPU utilization.

The single-image version left the GPU idle (0% util) because DeepFool's
Jacobian construction is vectorized across the batch dimension internally --
feeding it one image at a time means each backward pass only covers 1 sample,
wasting almost all of a GPU's parallel capacity. Batching fixes this: the
same number of backward passes (per class, per iteration) now cover many
images simultaneously.

Includes resume support (skips images with existing output files).
"""

import sys
import time
from pathlib import Path

import torch
import torchattacks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.victim_model import load_victim
from utils.preprocess import denormalize_tensor, load_image_tensor, save_tensor_image

IMAGE_SIZE = 64
INPUT_DIR = Path("data/clean_labeled/attack_source")
OUTPUT_DIR = Path("data/deepfool_adversarial_v2")
DEFAULT_BATCH_SIZE = 64


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate DeepFool adversarial samples (batched)")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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

    image_paths = [
        path for path in sorted(args.input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {args.input_dir}")

    print(f"Found {len(image_paths)} clean source images in {args.input_dir}")

    remaining = []
    skipped_existing = 0
    for img_path in image_paths:
        rel_path = img_path.relative_to(args.input_dir)
        out_path = args.output_dir / rel_path
        if out_path.exists():
            skipped_existing += 1
        else:
            remaining.append(img_path)

    print(f"Already done: {skipped_existing}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing left to do.")
        return

    atk = torchattacks.DeepFool(model, steps=args.steps)

    saved_count = 0
    start_time = time.time()
    batch_size = args.batch_size

    for batch_start in range(0, len(remaining), batch_size):
        batch_paths = remaining[batch_start:batch_start + batch_size]

        images = torch.cat(
            [load_image_tensor(p, IMAGE_SIZE, device, normalize=True) for p in batch_paths],
            dim=0,
        ).to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)

        adv_images = atk(images, preds)

        for i, img_path in enumerate(batch_paths):
            adv_to_save = denormalize_tensor(adv_images[i:i+1])
            rel_path = img_path.relative_to(args.input_dir)
            out_path = args.output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_tensor_image(adv_to_save, out_path)
            saved_count += 1

        elapsed = time.time() - start_time
        done = batch_start + len(batch_paths)
        rate = done / elapsed if elapsed > 0 else 0
        eta_sec = (len(remaining) - done) / rate if rate > 0 else 0
        print(f"[deepfool] {done}/{len(remaining)} done "
              f"({rate:.1f} img/s, ETA {eta_sec/60:.1f} min)")

    print(f"\n[deepfool] completed. Saved {saved_count} new images.")
    print(f"[deepfool] total on disk now: {skipped_existing + saved_count}/{len(image_paths)}")


if __name__ == "__main__":
    main()
