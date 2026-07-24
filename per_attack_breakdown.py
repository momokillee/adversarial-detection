"""Strict per-attack breakdown: only evaluates images that were in the
TRUE held-out validation set during training (per model), using the
manifests saved by train_dual.py / train_frequency.py.

For the pixel detector (no val split saved), falls back to full-set
evaluation with a clear warning, since that script doesn't split.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.model import AdversarialDetector
from detector.dual_model import DualDomainDetector
from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
ATTACKS = ["fgsm", "pgd", "bim", "cw", "deepfool"]

DETECTORS = {
    "pixel": ("models/detector.pt", AdversarialDetector, None),
    "dual": ("models/detector_dual.pt", DualDomainDetector, "models/detector_dual_val_split.json"),
    "frequency": ("models/detector_frequency.pt", FrequencyDetector, "models/detector_frequency_val_split.json"),
}


def load_detector(path, model_class):
    model = model_class().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def categorize(path_str: str) -> str:
    """Figure out which attack (or 'clean') a path belongs to."""
    p = Path(path_str)
    parts = p.parts
    for attack in ATTACKS:
        if attack in parts:
            return attack
    if "attack_source" in parts:
        return "clean"
    return "unknown"


def evaluate(model, paths_labels):
    correct = 0
    total = 0
    for path_str, label in paths_labels:
        x = load_image_tensor(Path(path_str), IMAGE_SIZE, DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        pred = 1 if prob > 0.5 else 0
        correct += int(pred == label)
        total += 1
    return correct, total


def main():
    models = {}
    manifests = {}

    for name, (path, cls, manifest_path) in DETECTORS.items():
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        models[name] = load_detector(path, cls)

        if manifest_path and Path(manifest_path).exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifests[name] = manifest
            print(f"{name}: using strict held-out validation set ({len(manifest)} files)")
        else:
            manifests[name] = None
            print(f"{name}: NO val split manifest found -- will evaluate on FULL dataset (results may be memorization-inflated)")

    print()
    print(f"{'Category':<10} | " + " | ".join(f"{n:>18}" for n in models.keys()))
    print("-" * (13 + 21 * len(models)))

    categories = ["clean"] + ATTACKS
    for category in categories:
        row = [category]
        for name, model in models.items():
            manifest = manifests[name]
            if manifest is not None:
                subset = [(m["path"], m["label"]) for m in manifest
                          if categorize(m["path"]) == category]
            else:
                # Fallback: full dataset (no strict split available)
                if category == "clean":
                    paths = [p for p in Path("data/clean_labeled/attack_source").rglob("*")
                              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                    subset = [(str(p), 0) for p in paths]
                else:
                    paths = [p for p in (Path("data/adversarial_all_v2") / category).rglob("*")
                              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                    subset = [(str(p), 1) for p in paths]

            if not subset:
                row.append("n/a")
                continue

            correct, total = evaluate(model, subset)
            pct = correct / total * 100 if total else 0
            row.append(f"{correct}/{total} ({pct:.1f}%)")

        print(f"{row[0]:<10} | " + " | ".join(f"{r:>18}" for r in row[1:]))


if __name__ == "__main__":
    main()