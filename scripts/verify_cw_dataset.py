"""
CW Dataset Sanity Check
------------------------
Verifies that your CW adversarial images:
  1. Actually differ from their clean counterparts at the pixel level
     (L2 / L-inf norm of the perturbation).
  2. Actually flip the model's predicted class (attack success).

Visual indistinguishability is EXPECTED for CW -- it's a minimum-norm
attack by design. This script checks the two things that actually
matter instead of eyeballing images.

Assumes:
  - Clean images live in data/clean/<name>.png
  - CW images live in data/cw_adversarial/<name>.png (same filenames,
    or adjust the pairing logic in `pair_files()` below)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Tuple
import torch
import torchvision.models as models
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")


def pair_files(clean_dir: Path, cw_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Pair clean and CW files by matching filename. CW files are named
    'cw_<clean_filename>', e.g. clean 'img_103.png' <-> CW 'cw_img_103.png'.
    """
    clean_files = {p.name: p for p in clean_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    cw_files_raw = {p.name: p for p in cw_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    # Strip the "cw_" prefix so CW filenames match their clean counterparts
    cw_files = {}
    for name, path in cw_files_raw.items():
        stripped = name[3:] if name.startswith("cw_") else name
        cw_files[stripped] = path

    common = sorted(set(clean_files) & set(cw_files))
    if not common:
        print("WARNING: no filenames matched even after stripping 'cw_' prefix.")
        print("Clean sample names:", list(clean_files)[:5])
        print("CW sample names:   ", list(cw_files_raw)[:5])
        print("You'll need to adjust pair_files() further to match your naming scheme.")
    return [(clean_files[name], cw_files[name]) for name in common]


def main():
    clean_dir = Path("data/clean")
    cw_dir = Path("data/cw_adversarial_v2")

    pairs = pair_files(clean_dir, cw_dir)
    print(f"Found {len(pairs)} matched clean/CW pairs\n")
    if not pairs:
        return

    print("Loading ResNet18 for prediction check...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    model.eval()

    l2_norms = []
    linf_norms = []
    zero_diff_count = 0
    flipped_count = 0
    total = 0

    with torch.no_grad():
        for clean_path, cw_path in pairs:
            clean_img = load_image_tensor(clean_path, IMAGE_SIZE, DEVICE)
            cw_img = load_image_tensor(cw_path, IMAGE_SIZE, DEVICE)

            diff = (cw_img - clean_img).flatten()
            l2 = diff.norm(p=2).item()
            linf = diff.abs().max().item()
            l2_norms.append(l2)
            linf_norms.append(linf)

            if l2 < 1e-6:
                zero_diff_count += 1

            clean_pred = model(clean_img).argmax(dim=1).item()
            cw_pred = model(cw_img).argmax(dim=1).item()
            if clean_pred != cw_pred:
                flipped_count += 1
            total += 1

    print("=" * 60)
    print("PERTURBATION MAGNITUDE")
    print("=" * 60)
    print(f"  Mean L2 norm:   {sum(l2_norms)/len(l2_norms):.6f}")
    print(f"  Min L2 norm:    {min(l2_norms):.6f}")
    print(f"  Max L2 norm:    {max(l2_norms):.6f}")
    print(f"  Mean Linf norm: {sum(linf_norms)/len(linf_norms):.6f}")
    print(f"  Images with essentially ZERO perturbation (L2 < 1e-6): {zero_diff_count}/{total}")

    print("\n" + "=" * 60)
    print("ATTACK SUCCESS RATE (does CW actually flip the prediction?)")
    print("=" * 60)
    print(f"  Flipped predictions: {flipped_count}/{total} ({100*flipped_count/total:.2f}%)")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if zero_diff_count == total:
        print("  -> CW images are byte-identical to clean images. The attack")
        print("     generation script did NOT apply any perturbation at all.")
        print("     This IS a real bug -- check your CW generation script,")
        print("     likely the perturbed tensor isn't being saved correctly")
        print("     (e.g. saving the wrong variable, or clean image overwriting it).")
    elif flipped_count / total < 0.5:
        print("  -> Perturbation exists but rarely flips the prediction.")
        print("     Either your CW attack parameters (confidence, c, steps)")
        print("     are too weak, or something is off in how the attack")
        print("     is being run against this model specifically.")
    else:
        print("  -> Perturbation is small (as expected for CW) AND it")
        print("     successfully flips predictions most of the time.")
        print("     This is CW working correctly. Visual similarity to the")
        print("     human eye is expected, not a sign of a bug -- it")
        print("     actually reinforces your core finding: CW produces a")
        print("     perturbation small enough to evade both human perception")
        print("     and your detectors, while still being an effective attack.")


if __name__ == "__main__":
    main()