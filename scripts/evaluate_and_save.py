"""Evaluate detector, save JSON summary and confusion matrix image.

Produces:
- experiments/eval_results.json
- experiments/confusion_matrix.png

Device: CPU
"""

import sys
from pathlib import Path
import json
from typing import List, Tuple, Dict

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/detector_frequency.pt")
OUT_DIR = Path("experiments")
OUT_JSON = OUT_DIR / "eval_results.json"
OUT_CM = OUT_DIR / "confusion_matrix.png"


def gather_image_paths() -> List[Tuple[str, List[Path], int]]:
    base = Path(".")
    data_clean = base / "data" / "clean"
    data_adv = base / "data" / "adversarial"
    bim_dir = base / "data" / "bim_adversarial"
    cw_dir = base / "data" / "cw_adversarial"
    deepfool_dir = base / "data" / "deepfool_adversarial"

    categories = []
    clean_paths = sorted([p for p in data_clean.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    categories.append(("Clean", clean_paths, 0))
    fgsm_paths = sorted([p for p in data_adv.glob("fgsm_*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    categories.append(("FGSM", fgsm_paths, 1))
    pgd_paths = sorted([p for p in data_adv.glob("pgd_*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    categories.append(("PGD", pgd_paths, 1))
    bim_paths = sorted([p for p in bim_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if bim_dir.exists() else []
    categories.append(("BIM", bim_paths, 1))
    cw_paths = sorted([p for p in cw_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if cw_dir.exists() else []
    categories.append(("CW", cw_paths, 1))
    df_paths = sorted([p for p in deepfool_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if deepfool_dir.exists() else []
    categories.append(("DeepFool", df_paths, 1))
    return categories


def load_detector(path: Path, device: torch.device) -> FrequencyDetector:
    model = FrequencyDetector().to(device)
    if path.exists():
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        raise FileNotFoundError(f"Model weights not found at {path}")
    return model


def evaluate_and_collect(model: FrequencyDetector, categories: List[Tuple[str, List[Path], int]]):
    all_records = []
    per_category = {}
    # confusion matrix counters: [[TN, FP],[FN, TP]] where rows=true label 0/1
    cm = [[0, 0], [0, 0]]

    for name, paths, expected in categories:
        cat_records = []
        total = 0
        correct = 0
        conf_sum = 0.0
        for p in paths:
            try:
                tensor = load_image_tensor(p, IMAGE_SIZE, DEVICE)
                with torch.no_grad():
                    prob = model.predict_proba(tensor.to(DEVICE)).item()
                pred = 1 if prob > 0.5 else 0
                is_correct = (pred == expected)
                total += 1
                if is_correct:
                    correct += 1
                conf_sum += prob
                # update confusion matrix
                cm[expected][pred] += 1
                rec = {
                    "path": str(p),
                    "attack_type": name,
                    "expected": int(expected),
                    "pred": int(pred),
                    "prob": float(prob),
                    "correct": bool(is_correct),
                }
                cat_records.append(rec)
                all_records.append(rec)
            except Exception as e:
                print(f"Warning: failed to process {p}: {e}")
        avg_conf = (conf_sum / total) if total > 0 else 0.0
        per_category[name] = {
            "images_tested": total,
            "correct": correct,
            "detection_rate": (correct / total * 100) if total > 0 else 0.0,
            "avg_confidence": avg_conf,
        }
    return all_records, per_category, cm


def save_json(out_path: Path, per_category: Dict, cm: List[List[int]], records: List[Dict]):
    OUT_DIR = out_path.parent
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_category": per_category,
        "confusion_matrix": cm,
        "records_count": len(records),
        "records": records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results to {out_path}")


def plot_and_save_cm(cm: List[List[int]], out_path: Path):
    OUT_DIR = out_path.parent
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cm_arr = torch.tensor(cm, dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_arr.numpy(), cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Clean (0)", "Adv (1)"])
    ax.set_yticklabels(["Clean (0)", "Adv (1)"])
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, int(cm[i][j]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix image to {out_path}")


def main():
    print("Running evaluation and saving JSON + confusion matrix (device=cpu)")
    categories = gather_image_paths()
    model = load_detector(MODEL_PATH, DEVICE)
    records, per_category, cm = evaluate_and_collect(model, categories)
    save_json(OUT_JSON, per_category, cm, records)
    plot_and_save_cm(cm, OUT_CM)
    print("Done.")


if __name__ == "__main__":
    main()
