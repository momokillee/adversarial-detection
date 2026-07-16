"""Evaluate the 224x224 frequency detector on clean, CW, FGSM, and PGD samples."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 224


def collect_files(root: Path, prefix: str = None, limit: int = None):
    files = sorted(root.glob("*"))
    files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if prefix is not None:
        files = [p for p in files if p.name.startswith(prefix)]

    if limit is not None:
        files = files[:limit]

    return files


def evaluate_dataset(model, files, label):
    correct = 0
    confidences = []

    for path in files:
        image = load_image_tensor(path, IMAGE_SIZE, torch.device("cuda"))
        with torch.no_grad():
            prob = torch.sigmoid(model(image)).item()

        pred = int(prob >= 0.5)
        if pred == label:
            correct += 1

        if label == 1:
            confidences.append(prob)
        else:
            confidences.append(1.0 - prob)

    return correct, len(files), sum(confidences) / len(confidences) if confidences else 0.0


def main():
    model = FrequencyDetector().to("cuda")
    model.load_state_dict(torch.load("models/detector_frequency_224.pt", map_location="cuda"))
    model.eval()

    clean_files = collect_files(Path("data/diverse_clean_224"), limit=500)
    cw_files = collect_files(Path("data/cw_adversarial_224"))
    fgsm_files = collect_files(Path("data/diverse_adversarial_224"), prefix="fgsm_", limit=500)
    pgd_files = collect_files(Path("data/diverse_adversarial_224"), prefix="pgd_", limit=500)

    datasets = [
        ("Clean", clean_files, 0),
        ("CW", cw_files, 1),
        ("FGSM", fgsm_files, 1),
        ("PGD", pgd_files, 1),
    ]

    print("Attack Type | Images | Correct | Detection Rate | Avg Confidence")
    print("---|---:|---:|---:|---:")

    for name, files, label in datasets:
        correct, total, avg_conf = evaluate_dataset(model, files, label)
        detection_rate = correct / total if total > 0 else 0.0
        print(f"{name} | {total} | {correct} | {detection_rate:.2%} | {avg_conf:.4f}")


if __name__ == "__main__":
    main()
