"""Build data/clean_labeled/{victim_train,attack_source} from a locally
cloned copy of https://github.com/YoongiKim/CIFAR-10-images.

Usage:
    1. First clone the mirror once (fast, single git operation):
         git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images.git /tmp/cifar_mirror

    2. Then run this script from the project root:
         python build_labeled_split.py --per-class 500 --source /tmp/cifar_mirror/train

No network requests happen in this script itself -- it only copies files
that are already on disk, so it's fast and can't hit rate limits or
connection resets.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
SEED = 42
TRAIN_RATIO = 0.75


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=500,
                         help="Max images to use per class")
    parser.add_argument("--source", type=Path, required=True,
                         help="Path to the cloned mirror's train/ folder, "
                              "e.g. /tmp/cifar_mirror/train")
    parser.add_argument("--output", type=Path,
                         default=Path("data/clean_labeled"),
                         help="Output directory")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"Source folder not found: {args.source}\n"
            f"Did you run: git clone --depth 1 "
            f"https://github.com/YoongiKim/CIFAR-10-images.git /tmp/cifar_mirror"
        )

    rng = random.Random(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    victim_train_dir = args.output / "victim_train"
    attack_source_dir = args.output / "attack_source"

    labels = {name: idx for idx, name in enumerate(CLASSES)}
    summary = {}

    for class_name in CLASSES:
        class_source = args.source / class_name
        if not class_source.exists():
            print(f"WARNING: {class_source} not found, skipping {class_name}")
            summary[class_name] = {"victim_train": 0, "attack_source": 0}
            continue

        image_files = sorted(
            p for p in class_source.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        rng.shuffle(image_files)
        image_files = image_files[: args.per_class]

        split_idx = int(len(image_files) * TRAIN_RATIO)
        train_files = image_files[:split_idx]
        attack_files = image_files[split_idx:]

        vt_dir = victim_train_dir / class_name
        as_dir = attack_source_dir / class_name
        vt_dir.mkdir(parents=True, exist_ok=True)
        as_dir.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy2(f, vt_dir / f.name)
        for f in attack_files:
            shutil.copy2(f, as_dir / f.name)

        summary[class_name] = {
            "victim_train": len(train_files),
            "attack_source": len(attack_files),
        }
        print(f"{class_name}: victim_train={len(train_files)} "
              f"attack_source={len(attack_files)}")

    with open(args.output / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    total = sum(v["victim_train"] + v["attack_source"] for v in summary.values())
    print(f"\nTotal images: {total}")
    print(f"Labels written to: {args.output / 'labels.json'}")


if __name__ == "__main__":
    main()