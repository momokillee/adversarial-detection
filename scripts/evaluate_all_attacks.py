"""Evaluate frequency-domain detector on clean and adversarial image sets.

Loads `models/detector_frequency.pt` and runs the `FrequencyDetector` on
all image categories, printing a results table and sorted summary.

Device: CPU (per your request)
IMAGE_SIZE=64
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch

# ensure project imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/detector_frequency.pt")


def gather_image_paths() -> List[Tuple[str, List[Path], int]]:
    """
    Return list of (label_name, [paths], expected_label) for each category.
    expected_label: 0 for Clean, 1 for Adversarial
    """
    base = Path(".")
    data_clean = base / "data" / "clean"
    data_adv = base / "data" / "adversarial"
    bim_dir = base / "data" / "bim_adversarial"
    cw_dir = base / "data" / "cw_adversarial"
    deepfool_dir = base / "data" / "deepfool_adversarial"

    categories = []

    # Clean
    clean_paths = sorted([p for p in data_clean.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    categories.append(("Clean", clean_paths, 0))

    # FGSM (prefix fgsm_)
    fgsm_paths = sorted([p for p in data_adv.glob("fgsm_*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    categories.append(("FGSM", fgsm_paths, 1))

    # PGD (prefix pgd_)
    pgd_paths = sorted([p for p in data_adv.glob("pgd_*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    categories.append(("PGD", pgd_paths, 1))

    # BIM
    bim_paths = sorted([p for p in bim_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if bim_dir.exists() else []
    categories.append(("BIM", bim_paths, 1))

    # CW
    cw_paths = sorted([p for p in cw_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if cw_dir.exists() else []
    categories.append(("CW", cw_paths, 1))

    # DeepFool
    df_paths = sorted([p for p in deepfool_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if deepfool_dir.exists() else []
    categories.append(("DeepFool", df_paths, 1))

    return categories


def load_detector(path: Path, device: torch.device) -> FrequencyDetector:
    model = FrequencyDetector().to(device)
    if path.exists():
        try:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
    else:
        raise FileNotFoundError(f"Model weights not found at {path}")
    return model


def evaluate_category(model: FrequencyDetector, paths: List[Path], expected_label: int) -> Tuple[int, int, float]:
    """
    Run detector on list of image paths.

    Returns:
        total, correct, avg_confidence (avg adversarial probability)
    """
    total = 0
    correct = 0
    conf_sum = 0.0

    for p in paths:
        try:
            tensor = load_image_tensor(p, IMAGE_SIZE, DEVICE)  # shape (1,3,H,W)
            with torch.no_grad():
                prob = model.predict_proba(tensor.to(DEVICE)).item()  # adversarial probability
            pred_label = 1 if prob > 0.5 else 0
            if pred_label == expected_label:
                correct += 1
            conf_sum += prob
            total += 1
        except Exception as e:
            # skip problematic images but report
            print(f"Warning: failed to process {p}: {e}")

    avg_conf = (conf_sum / total) if total > 0 else 0.0
    return total, correct, avg_conf


def print_results_table(results: List[Tuple[str, int, int, float]]):
    """
    Print table:
    Attack Type | Images Tested | Correct | Detection Rate | Avg Confidence
    """
    header = f"{'Attack Type':<12} | {'Images':>6} | {'Correct':>7} | {'Detection Rate':>14} | {'Avg Conf':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, total, correct, avg_conf in results:
        rate = (correct / total * 100) if total > 0 else 0.0
        print(f"{name:<12} | {total:6d} | {correct:7d} | {rate:13.2f}% | {avg_conf:9.3f}")
    print(sep)


def main():
    print("Evaluate frequency-domain detector on all attack types (device=cpu)")
    categories = gather_image_paths()
    model = load_detector(MODEL_PATH, DEVICE)

    results = []
    for name, paths, expected in categories:
        total, correct, avg_conf = evaluate_category(model, paths, expected)
        results.append((name, total, correct, avg_conf))

    # Print table
    print_results_table(results)

    # Summary sorted by detection rate (highest first)
    sortable = []
    for name, total, correct, avg_conf in results:
        rate = (correct / total * 100) if total > 0 else 0.0
        sortable.append((rate, name, total, correct, avg_conf))
    sortable.sort(reverse=True)

    print("\nOverall summary (sorted by detection rate):")
    print("Rank | Detection Rate | Attack Type | Images | Correct | Avg Conf")
    print("-----|----------------|-------------|--------|---------|---------")
    for rank, (rate, name, total, correct, avg_conf) in enumerate(sortable, start=1):
        print(f"{rank:4d} | {rate:13.2f}% | {name:11s} | {total:6d} | {correct:7d} | {avg_conf:8.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
