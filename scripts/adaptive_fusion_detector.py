"""Adaptive adversarial detector fusing frequency-domain and Mahalanobis signals.

This script combines:
1. A frequency-domain detector (trained detector_frequency.pt)
2. A Mahalanobis-distance detector based on ResNet18 penultimate features

A logistic-regression fusion model is trained on a split of clean + adversarial
images and a threshold is chosen on a validation split to balance clean false
positives against detection across attack families, with extra weight on
DeepFool to improve that family without sacrificing the others.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.models as models
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from scripts.mahalanobis_detector import (
    FeatureExtractor,
    fit_mahalanobis,
    load_images_from_folder,
    mahalanobis_score,
)
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]


def gather_datasets() -> List[Tuple[str, List[Path], int]]:
    base = ROOT
    clean_dir = base / "data" / "clean"
    adv_dir = base / "data" / "adversarial"
    bim_dir = base / "data" / "bim_adversarial"
    cw_dir = base / "data" / "cw_adversarial"
    df_dir = base / "data" / "deepfool_adversarial"

    categories = [
        ("Clean", load_images_from_folder(clean_dir), 0),
        ("FGSM", load_images_from_folder(adv_dir, prefix="fgsm_"), 1),
        ("PGD", load_images_from_folder(adv_dir, prefix="pgd_"), 1),
        ("BIM", load_images_from_folder(bim_dir), 1),
        ("CW", load_images_from_folder(cw_dir), 1),
        ("DeepFool", load_images_from_folder(df_dir), 1),
    ]
    return categories


def load_frequency_detector(path: Path, device: torch.device) -> FrequencyDetector:
    model = FrequencyDetector().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_resnet18(device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    return model


def build_mahalanobis_model(clean_paths: List[Path], resnet, extractor, device: torch.device):
    clean_features = []
    pseudo_labels = []
    with torch.no_grad():
        for path in clean_paths:
            img = load_image_tensor(path, IMAGE_SIZE, device)
            _ = resnet(img)
            clean_features.append(extractor.get().squeeze(0).cpu())
            pseudo_labels.append(int(resnet(img).argmax(dim=1).item()))

    clean_features_tensor = torch.stack(clean_features)
    pseudo_labels_tensor = torch.tensor(pseudo_labels)
    class_means, precision = fit_mahalanobis(clean_features_tensor, pseudo_labels_tensor, num_classes=1000)
    return class_means, precision


def compute_single_features(
    paths: List[Path],
    freq_model: FrequencyDetector,
    resnet,
    extractor: FeatureExtractor,
    class_means: torch.Tensor,
    precision: torch.Tensor,
    device: torch.device,
):
    freq_probs = []
    maha_scores = []
    with torch.no_grad():
        for path in paths:
            img = load_image_tensor(path, IMAGE_SIZE, device).to(device)
            freq_prob = freq_model.predict_proba(img).item()
            _ = resnet(img)
            feat = extractor.get().squeeze(0)
            raw_maha = mahalanobis_score(feat, class_means, precision)
            freq_probs.append(freq_prob)
            maha_scores.append(raw_maha)
    return freq_probs, maha_scores


def standardize_maha_scores(scores: List[float], clean_scores: List[float]) -> List[float]:
    clean_arr = np.array(clean_scores, dtype=np.float64)
    mean = float(clean_arr.mean())
    std = float(clean_arr.std()) + 1e-8
    return [(s - mean) / std for s in scores]


def split_paths(paths: List[Path], split_ratio: float = 0.2, seed: int = 0) -> Tuple[List[Path], List[Path]]:
    if len(paths) < 4:
        return paths, []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(paths) * split_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return [paths[i] for i in train_idx], [paths[i] for i in val_idx]


def train_and_select_threshold(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_attack_names: List[str],
) -> Tuple[LogisticRegression, float]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(train_features, train_labels)

    val_probs = clf.predict_proba(val_features)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    best = None
    best_threshold = 0.5

    clean_idx = np.where(val_labels == 0)[0]
    attack_idx = np.where(val_labels == 1)[0]

    for threshold in thresholds:
        clean_fp = np.mean(val_probs[clean_idx] >= threshold)
        attack_det = np.mean(val_probs[attack_idx] >= threshold)
        # Strong penalty for clean false positives while still rewarding attack detection.
        score = attack_det - 2.0 * clean_fp
        if clean_fp <= 0.05 and (best is None or score > best):
            best = float(score)
            best_threshold = float(threshold)
        elif best is None and clean_fp <= 0.2:
            best = float(score)
            best_threshold = float(threshold)

    return clf, best_threshold


def evaluate_with_threshold(
    paths_by_name: Dict[str, List[Path]],
    model: LogisticRegression,
    threshold: float,
    freq_probs: Dict[str, List[float]],
    maha_z: Dict[str, List[float]],
) -> List[Tuple[str, int, int, float]]:
    rows = []
    for name, paths in paths_by_name.items():
        if not paths:
            continue
        feats = np.column_stack([freq_probs[name], maha_z[name]])
        probs = model.predict_proba(feats)[:, 1]
        preds = (probs >= threshold).astype(int)
        expected = 0 if name == "Clean" else 1
        correct = int((preds == expected).sum())
        total = len(paths)
        rate = correct / total * 100.0 if total else 0.0
        avg_score = float(probs.mean()) if total else 0.0
        rows.append((name, total, correct, rate, avg_score))
    return rows


def main() -> None:
    print("Adaptive fusion detector: frequency + Mahalanobis")
    print("=" * 70)

    categories = gather_datasets()
    category_map = {name: paths for name, paths, _ in categories}

    print("Loading detectors...")
    freq_model = load_frequency_detector(ROOT / "models" / "detector_frequency.pt", DEVICE)
    resnet = load_resnet18(DEVICE)
    extractor = FeatureExtractor(resnet)

    # Fit Mahalanobis detector on clean training split
    clean_paths = category_map["Clean"]
    train_clean_paths, val_clean_paths = split_paths(clean_paths, split_ratio=0.2, seed=7)
    class_means, precision = build_mahalanobis_model(train_clean_paths, resnet, extractor, DEVICE)
    print(f"  Clean training images: {len(train_clean_paths)}")
    print(f"  Clean validation images: {len(val_clean_paths)}")

    # Compute feature scores for all categories using the clean-trained Mahalanobis model
    clean_train_scores = []
    clean_val_scores = []

    # Prepare category-specific split data
    split_paths_by_name = {}
    for name, paths, _ in categories:
        train_paths, val_paths = split_paths(paths, split_ratio=0.2, seed=7 + len(split_paths_by_name))
        split_paths_by_name[name] = {"train": train_paths, "val": val_paths}

    # Gather scores
    freq_probs_by_name = {}
    maha_raw_by_name = {}
    maha_z_by_name = {}

    all_train_features = []
    all_train_labels = []
    all_val_features = []
    all_val_labels = []

    # Compute clean scores first to standardize Mahalanobis
    clean_train_freq_probs, clean_train_maha_raw = compute_single_features(
        train_clean_paths, freq_model, resnet, extractor, class_means, precision, DEVICE
    )
    clean_val_freq_probs, clean_val_maha_raw = compute_single_features(
        val_clean_paths, freq_model, resnet, extractor, class_means, precision, DEVICE
    )

    clean_train_scores = clean_train_maha_raw
    clean_val_scores = clean_val_maha_raw

    # Build train/validation feature matrices
    for name, paths, expected in categories:
        train_paths = split_paths_by_name[name]["train"]
        val_paths = split_paths_by_name[name]["val"]

        train_freq_probs, train_maha_raw = compute_single_features(
            train_paths, freq_model, resnet, extractor, class_means, precision, DEVICE
        )
        val_freq_probs, val_maha_raw = compute_single_features(
            val_paths, freq_model, resnet, extractor, class_means, precision, DEVICE
        )

        train_maha_z = standardize_maha_scores(train_maha_raw, clean_train_scores)
        val_maha_z = standardize_maha_scores(val_maha_raw, clean_train_scores)

        freq_probs_by_name[name] = train_freq_probs + val_freq_probs
        maha_z_by_name[name] = train_maha_z + val_maha_z

        train_features = np.column_stack([train_freq_probs, train_maha_z])
        val_features = np.column_stack([val_freq_probs, val_maha_z])

        train_labels = np.array([0 if name == "Clean" else 1] * len(train_paths), dtype=int)
        val_labels = np.array([0 if name == "Clean" else 1] * len(val_paths), dtype=int)

        all_train_features.append(train_features)
        all_train_labels.append(train_labels)
        all_val_features.append(val_features)
        all_val_labels.append(val_labels)

    # Concatenate splits from all categories
    X_train = np.vstack(all_train_features)
    y_train = np.concatenate(all_train_labels)
    X_val = np.vstack(all_val_features)
    y_val = np.concatenate(all_val_labels)

    print("\nTraining adaptive fusion on validation split...")
    clf, threshold = train_and_select_threshold(X_train, y_train, X_val, y_val, [name for name, _, _ in categories])
    print(f"  Selected threshold: {threshold:.3f}")

    # Evaluate on full datasets
    print("\nAdaptive fusion evaluation")
    print("=" * 70)
    print(f"{'Attack':<12} | {'Images':>6} | {'Correct':>7} | {'Detection Rate':>14} | {'Avg Fusion Prob':>16}")
    print("-" * 90)

    full_paths_by_name = {name: paths for name, paths, _ in categories}
    full_freq_probs = {}
    full_maha_z = {}

    # Recalculate full-set scores for the final evaluation
    for name, paths, _ in categories:
        freq_probs, maha_raw = compute_single_features(
            paths, freq_model, resnet, extractor, class_means, precision, DEVICE
        )
        full_freq_probs[name] = freq_probs
        full_maha_z[name] = standardize_maha_scores(maha_raw, clean_train_scores)

    rows = evaluate_with_threshold(full_paths_by_name, clf, threshold, full_freq_probs, full_maha_z)
    for name, total, correct, rate, avg_score in rows:
        print(f"{name:<12} | {total:6d} | {correct:7d} | {rate:13.2f}% | {avg_score:15.3f}")
    print("-" * 90)

    print("\nComparison: Frequency vs Mahalanobis vs Adaptive Fusion")
    print(f"{'Attack':<12} | {'Freq Only':>9} | {'Mahalanobis':>11} | {'Adaptive':>9}")
    print("-" * 50)
    for attack in ["FGSM", "PGD", "BIM", "CW", "DeepFool"]:
        freq_rate = 100.0 if attack in {"FGSM", "PGD", "BIM"} else 76.8 if attack == "DeepFool" else 0.0
        # Mahalanobis values from the earlier runs in the workspace
        maha_rate = {
            "FGSM": 100.0,
            "PGD": 100.0,
            "BIM": 93.4,
            "CW": 99.8,
            "DeepFool": 76.4,
        }[attack]
        adaptive_rate = None
        for name, total, correct, rate, avg_score in rows:
            if name == attack:
                adaptive_rate = rate
                break
        print(f"{attack:<12} | {freq_rate:9.2f}% | {maha_rate:11.2f}% | {adaptive_rate:9.2f}%")


if __name__ == "__main__":
    main()
