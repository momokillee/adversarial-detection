"""Generate DeepFool adversarial examples only (isolated from CW).

Includes:
- Resume support: skips images that already have a saved output file
- Per-image timeout: if a single image takes too long, it's skipped
  and logged, rather than blocking the whole run indefinitely
"""

import signal
import sys
from pathlib import Path

import torch
import torchattacks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.victim_model import load_victim
from utils.preprocess import denormalize_tensor, load_image_tensor, save_tensor_image

IMAGE_SIZE = 64
INPUT_DIR = Path("data/clean")
OUTPUT_DIR = Path("data/deepfool_adversarial")
PER_IMAGE_TIMEOUT_SECONDS = 30


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate DeepFool adversarial samples")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=PER_IMAGE_TIMEOUT_SECONDS)
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

    # Resume support: figure out which images already have output
    remaining = []
    skipped_existing = 0
    for img_path in image_paths:
        rel_path = img_path.relative_to(args.input_dir)
        out_path = args.output_dir / ("deepfool_" + rel_path.name)
        if out_path.exists():
            skipped_existing += 1
        else:
            remaining.append(img_path)

    print(f"Already done: {skipped_existing}, remaining: {len(remaining)}")

    atk = torchattacks.DeepFool(model, steps=args.steps)

    # Signal-based timeout only works on Unix (macOS/Linux), not Windows.
    use_timeout = hasattr(signal, "SIGALRM")
    if use_timeout:
        signal.signal(signal.SIGALRM, _timeout_handler)

    saved_count = 0
    timed_out = []

    for index, img_path in enumerate(remaining, start=1):
        image = load_image_tensor(img_path, IMAGE_SIZE, device, normalize=True)
        image = image.to(device).float()

        with torch.no_grad():
            outputs = model(image)
            pred = outputs.argmax(dim=1).item()
        label = torch.tensor([pred], dtype=torch.long, device=device)

        try:
            if use_timeout:
                signal.alarm(args.timeout)
            adv_image = atk(image, label)
            if use_timeout:
                signal.alarm(0)
        except TimeoutError:
            print(f"  TIMEOUT after {args.timeout}s on {img_path} -- skipping")
            timed_out.append(str(img_path))
            continue
        finally:
            if use_timeout:
                signal.alarm(0)

        adv_to_save = denormalize_tensor(adv_image)
        rel_path = img_path.relative_to(args.input_dir)
        out_path = args.output_dir / ("deepfool_" + rel_path.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_tensor_image(adv_to_save, out_path)
        saved_count += 1

        if index % 25 == 0:
            print(f"[deepfool] processed {index}/{len(remaining)} remaining "
                  f"({saved_count} saved, {len(timed_out)} timed out)...")

    print(f"\n[deepfool] completed this run. Saved {saved_count} new images.")
    print(f"[deepfool] total on disk now: {skipped_existing + saved_count}/{len(image_paths)}")
    if timed_out:
        print(f"[deepfool] {len(timed_out)} images timed out and were skipped:")
        for p in timed_out:
            print(f"    {p}")


if __name__ == "__main__":
    main()