"""Full sanity check across all 5 attack types before detector retraining.

Checks:
1. File counts per attack folder (and per class)
2. Images are valid, correct size/mode
3. Predictions actually flip vs the trained victim (not just "files exist")
"""

import sys
from pathlib import Path
import random

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attacks.victim_model import load_victim
from utils.preprocess import load_image_tensor

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

ATTACK_DIRS = {
    "fgsm": ("data/adversarial_v2", "fgsm_"),
    "pgd": ("data/adversarial_v2", "pgd_"),
    "bim": ("data/bim_adversarial_v2", ""),
    "cw": ("data/cw_adversarial_v2", ""),
    "deepfool": ("data/deepfool_adversarial_v2", ""),
}

CLEAN_DIR = Path("data/clean_labeled/attack_source")
DEVICE = torch.device("cpu")


def main():
    print("=" * 70)
    print("STEP 1: File counts per attack type")
    print("=" * 70)

    for attack, (out_dir, prefix) in ATTACK_DIRS.items():
        out_path = Path(out_dir)
        if prefix:
            files = list(out_path.rglob(f"{prefix}*"))
        else:
            files = [p for p in out_path.rglob("*") if p.is_file()
                     and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"{attack:10s}: {len(files):5d} files in {out_dir}")

        # per-class breakdown
        per_class = {}
        for f in files:
            cls = f.parent.name
            per_class[cls] = per_class.get(cls, 0) + 1
        missing = [c for c in CLASSES if per_class.get(c, 0) == 0]
        if missing:
            print(f"           WARNING: missing/empty classes: {missing}")
        else:
            counts = sorted(per_class.values())
            print(f"           per-class range: {counts[0]}-{counts[-1]}")

    print()
    print("=" * 70)
    print("STEP 2: Image validity spot-check (5 random per attack)")
    print("=" * 70)

    for attack, (out_dir, prefix) in ATTACK_DIRS.items():
        out_path = Path(out_dir)
        if prefix:
            files = list(out_path.rglob(f"{prefix}*"))
        else:
            files = [p for p in out_path.rglob("*") if p.is_file()
                     and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        sample = random.sample(files, min(5, len(files)))
        bad = 0
        for f in sample:
            try:
                img = Image.open(f)
                img.verify()
            except Exception as e:
                print(f"  CORRUPT: {f} -- {e}")
                bad += 1
        status = "OK" if bad == 0 else f"{bad} CORRUPT"
        print(f"{attack:10s}: {len(sample)} sampled -- {status}")

    print()
    print("=" * 70)
    print("STEP 3: Prediction flip rate vs trained victim (20 samples/attack)")
    print("=" * 70)

    model = load_victim(DEVICE)
    model.eval()

    for attack, (out_dir, prefix) in ATTACK_DIRS.items():
        out_path = Path(out_dir)
        if prefix:
            adv_files = list(out_path.rglob(f"{prefix}*"))
        else:
            adv_files = [p for p in out_path.rglob("*") if p.is_file()
                         and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        sample = random.sample(adv_files, min(20, len(adv_files)))
        changed = 0
        checked = 0

        for adv_path in sample:
            cls = adv_path.parent.name
            clean_name = adv_path.name
            if prefix:
                clean_name = clean_name[len(prefix):]
            clean_path = CLEAN_DIR / cls / clean_name

            if not clean_path.exists():
                continue

            clean_t = load_image_tensor(clean_path, 64, DEVICE)
            adv_t = load_image_tensor(adv_path, 64, DEVICE)

            with torch.no_grad():
                c_pred = model(clean_t).argmax(dim=1).item()
                a_pred = model(adv_t).argmax(dim=1).item()

            checked += 1
            changed += int(c_pred != a_pred)

        pct = (changed / checked * 100) if checked else 0
        print(f"{attack:10s}: {changed}/{checked} predictions changed ({pct:.0f}%)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()