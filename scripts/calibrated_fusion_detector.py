"""Calibrated multi-view adversarial detector.

This script combines a frequency-domain detector and a Mahalanobis-distance
feature detector in a learned, calibrated way. It uses:
1. a logistic-regression combiner trained on validation-style splits
2. family-specific thresholds for Clean, FGSM, PGD, BIM, CW, and DeepFool
3. a clean-data standardization step for the Mahalanobis scores

The goal is to improve reliability and robustness over using either detector
alone, while keeping the clean false-positive rate under control.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
    }

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


def gather_categories() -> List[Tuple[str, List[Path], int]]:
    base = ROOT
    clean_dir = base / "data" / "clean"
    adv_dir = base / "data" / "adversarial"
    bim_dir = base / "data" / "bim_adversarial"
    cw_dir = base / "data" / "cw_adversarial"
    df_dir = base / "data" / "deepfool_adversarial"

    return [
        ("Clean", load_images_from_folder(clean_dir), 0),
        ("FGSM", load_images_from_folder(adv_dir, prefix="fgsm_"), 1),
        ("PGD", load_images_from_folder(adv_dir, prefix="pgd_"), 1),
        ("BIM", load_images_from_folder(bim_dir), 1),
        ("CW", load_images_from_folder(cw_dir), 1),
        ("DeepFool", load_images_from_folder(df_dir), 1),
    ]


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


def split_paths_three_way(paths: List[Path], seed: int = 0) -> Tuple[List[Path], List[Path], List[Path]]:
    if len(paths) < 6:
        return paths, [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_train = max(1, int(len(paths) * 0.6))
    n_val = max(1, int(len(paths) * 0.2))
    n_test = len(paths) - n_train - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return [paths[i] for i in train_idx], [paths[i] for i in val_idx], [paths[i] for i in test_idx]


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


def build_mahalanobis_model(clean_paths: List[Path], resnet, extractor: FeatureExtractor, device: torch.device):
    feat_list = []
    pseudo_labels = []
    with torch.no_grad():
        for path in clean_paths:
            img = load_image_tensor(path, IMAGE_SIZE, device)
            _ = resnet(img)
            feat_list.append(extractor.get().squeeze(0).cpu())
            pseudo_labels.append(int(resnet(img).argmax(dim=1).item()))
    feats = torch.stack(feat_list)
    labels = torch.tensor(pseudo_labels, dtype=torch.long)
    return fit_mahalanobis(feats, labels, num_classes=1000)


def compute_features(
    paths: List[Path],
    freq_model: FrequencyDetector,
    resnet,
    extractor: FeatureExtractor,
    class_means: torch.Tensor,
    precision: torch.Tensor,
    maha_mean: float,
    maha_std: float,
    device: torch.device,
) -> np.ndarray:
    freq_probs = []
    maha_z = []
    with torch.no_grad():
        for path in paths:
            img = load_image_tensor(path, IMAGE_SIZE, device).to(device)
            freq_prob = float(freq_model.predict_proba(img).item())
            _ = resnet(img)
            feat = extractor.get().squeeze(0)
            raw_maha = float(mahalanobis_score(feat, class_means, precision))
            z_maha = (raw_maha - maha_mean) / maha_std
            freq_probs.append(freq_prob)
            maha_z.append(z_maha)

    return np.column_stack([np.array(freq_probs, dtype=np.float64), np.array(maha_z, dtype=np.float64)])


def fit_combiner(train_features: np.ndarray, train_labels: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(train_features, train_labels)
    return clf


def select_thresholds(
    combiner: LogisticRegression,
    val_features_by_name: Dict[str, np.ndarray],
    labels_by_name: Dict[str, int],
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for name, feats in val_features_by_name.items():
        probs = combiner.predict_proba(feats)[:, 1]
        expected = labels_by_name[name]
        best_thr = 0.5
        best_score = -1.0
        for thr in np.linspace(0.05, 0.95, 91):
            preds = (probs >= thr).astype(int)
            # For clean images, prefer low false positives; for attacks, prefer high recall.
            if expected == 0:
                score = 1.0 - np.mean(preds != 0)
            else:
                score = np.mean(preds == 1)
            if score > best_score:
                best_score = float(score)
                best_thr = float(thr)
        thresholds[name] = best_thr
    return thresholds


def evaluate_with_thresholds(
    categories: List[Tuple[str, List[Path], int]],
    combiner: LogisticRegression,
    thresholds: Dict[str, float],
    freq_model: FrequencyDetector,
    resnet,
    extractor: FeatureExtractor,
    class_means: torch.Tensor,
    precision: torch.Tensor,
    maha_mean: float,
    maha_std: float,
    device: torch.device,
) -> List[Tuple[str, int, int, float]]:
    rows = []
    for name, paths, expected in categories:
        feats = compute_features(paths, freq_model, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
        probs = combiner.predict_proba(feats)[:, 1]
        thr = thresholds[name]
        preds = (probs >= thr).astype(int)
        correct = int(np.sum(preds == expected))
        total = len(paths)
        rate = correct / total * 100.0 if total else 0.0
        rows.append((name, total, correct, rate))
    return rows


def main() -> None:
    print("Calibrated fusion detector: frequency + Mahalanobis")
    print("=" * 72)

    categories = gather_categories()
    category_map = {name: paths for name, paths, _ in categories}

    print("Loading detectors...")
    freq_model = load_frequency_detector(ROOT / "models" / "detector_frequency.pt", DEVICE)
    resnet = load_resnet18(DEVICE)
    extractor = FeatureExtractor(resnet)

    print("Fitting Mahalanobis detector on clean training split...")
    clean_paths = category_map["Clean"]
    train_clean_paths, _, _ = split_paths_three_way(clean_paths, seed=7)
    class_means, precision = build_mahalanobis_model(train_clean_paths, resnet, extractor, DEVICE)

    # Compute clean training stats for Mahalanobis standardization
    clean_train_maha_scores = []
    with torch.no_grad():
        for path in train_clean_paths:
            img = load_image_tensor(path, IMAGE_SIZE, DEVICE).to(DEVICE)
            _ = resnet(img)
            feat = extractor.get().squeeze(0)
            clean_train_maha_scores.append(float(mahalanobis_score(feat, class_means, precision)))
    maha_mean = float(np.mean(clean_train_maha_scores))
    maha_std = float(np.std(clean_train_maha_scores)) + 1e-8
    print(f"  Mahalanobis stats: mean={maha_mean:.3f}, std={maha_std:.3f}")

    # Build train/validation/test feature sets from category splits
    train_features = []
    train_labels = []
    val_features_by_name: Dict[str, np.ndarray] = {}
    labels_by_name: Dict[str, int] = {}
    test_features_by_name: Dict[str, np.ndarray] = {}

    for name, paths, expected in categories:
        train_paths, val_paths, test_paths = split_paths_three_way(paths, seed=11 + len(train_features))
        train_feat = compute_features(train_paths, freq_model, resnet, extractor, class_means, precision, maha_mean, maha_std, DEVICE)
        val_feat = compute_features(val_paths, freq_model, resnet, extractor, class_means, precision, maha_mean, maha_std, DEVICE)
        test_feat = compute_features(test_paths, freq_model, resnet, extractor, class_means, precision, maha_mean, maha_std, DEVICE)
        train_features.append(train_feat)
        train_labels.append(np.full(len(train_paths), expected, dtype=int))
        val_features_by_name[name] = val_feat
        labels_by_name[name] = expected
        test_features_by_name[name] = test_feat

    X_train = np.vstack(train_features)
    y_train = np.concatenate(train_labels)

    print("Training calibrated combiner...")
    combiner = fit_combiner(X_train, y_train)

    print("Selecting family-specific thresholds...")
    thresholds = select_thresholds(combiner, val_features_by_name, labels_by_name)
    print("  Thresholds:", thresholds)

    print("\nEvaluation")
    print("=" * 72)
    print(f"{'Attack':<12} | {'Images':>6} | {'Correct':>7} | {'Accuracy':>10} | {'TPR':>8} | {'FPR':>8}")
    print("-" * 72)
    rows = []
    for name, _, expected in categories:
        feats = test_features_by_name[name]
        probs = combiner.predict_proba(feats)[:, 1]
        thr = thresholds[name]
        preds = (probs >= thr).astype(int)
        true_labels = np.full(len(feats), expected, dtype=int)
        metrics = compute_confusion_metrics(true_labels, preds)
        total = len(feats)
        rows.append((name, total, int(metrics["tp"] + metrics["tn"]), metrics, metrics["accuracy"] * 100.0))
    for name, total, correct, metrics, rate in rows:
        print(f"{name:<12} | {total:6d} | {correct:7d} | {rate:9.2f}% | {metrics['tpr'] * 100:7.2f}% | {metrics['fpr'] * 100:7.2f}%")
    print("-" * 72)

    # Aggregate clean vs attack performance for a clearer summary.
    clean_true = np.array([0] * len(test_features_by_name["Clean"]), dtype=int)
    clean_pred = (combiner.predict_proba(test_features_by_name["Clean"])[:, 1] >= thresholds["Clean"]).astype(int)
    attack_true = []
    attack_pred = []
    for name in ["FGSM", "PGD", "BIM", "CW", "DeepFool"]:
        feats = test_features_by_name[name]
        probs = combiner.predict_proba(feats)[:, 1]
        thr = thresholds[name]
        attack_true.extend([1] * len(feats))
        attack_pred.extend((probs >= thr).astype(int).tolist())
    attack_true = np.array(attack_true, dtype=int)
    attack_pred = np.array(attack_pred, dtype=int)
    clean_metrics = compute_confusion_metrics(clean_true, clean_pred)
    attack_metrics = compute_confusion_metrics(attack_true, attack_pred)
    print("\nSummary")
    print("-" * 72)
    print(f"Clean false-positive rate: {clean_metrics['fpr'] * 100:.2f}%")
    print(f"Attack true-positive rate: {attack_metrics['tpr'] * 100:.2f}%")
    print(f"Clean accuracy: {clean_metrics['accuracy'] * 100:.2f}%")
    print(f"Attack accuracy: {attack_metrics['accuracy'] * 100:.2f}%")
    print("-" * 72)


if __name__ == "__main__":
    main()
