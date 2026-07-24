"""Re-save ALL clean images (attack_source) through the identical
load -> tensor -> denormalize -> save pipeline used for adversarial
images, with ZERO perturbation applied.

This removes the recompression-artifact shortcut discovered by the
control test: previously, adversarial images had all gone through a
resave step while clean images were untouched originals, letting
detectors trivially learn "was this resaved" instead of "was this
perturbed". After this fix, BOTH classes share the same compression
baseline, forcing genuine perturbation detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.preprocess import denormalize_tensor, load_image_tensor, save_tensor_image
import torch

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
SOURCE_DIR = Path("data/clean_labeled/attack_source")
OUTPUT_DIR = Path("data/clean_labeled/attack_source_resaved")


def main():
    image_paths = [
        p for p in SOURCE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    print(f"Found {len(image_paths)} clean images to re-save")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_path in image_paths:
        rel_path = src_path.relative_to(SOURCE_DIR)
        out_path = OUTPUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = load_image_tensor(src_path, IMAGE_SIZE, DEVICE, normalize=True)
        # No perturbation -- straight round-trip, identical to how
        # adversarial images are produced minus the actual attack step.
        to_save = denormalize_tensor(image)
        save_tensor_image(to_save, out_path)
        count += 1

        if count % 250 == 0:
            print(f"  {count}/{len(image_paths)} done...")

    print(f"\nDone. {count} clean images re-saved to {OUTPUT_DIR}")
    print("Use this directory as --clean-dir for all detector retraining "
          "from now on, instead of the original attack_source.")


if __name__ == "__main__":
    main()