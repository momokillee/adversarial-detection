"""Paper-ready evaluation for the adversarial detectors.

This script adds:
1. repeated split evaluation across multiple seeds,
2. ablation studies (frequency-only, Mahalanobis-only, dual-model, fusion),
3. evaluation on unseen attack data if present,
4. ROC/PR curve generation and bootstrap confidence intervals.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.dual_model import DualDomainDetector
from detector.frequency_model import FrequencyDetector
from scripts.calibrated_fusion_detector import compute_confusion_metrics, load_frequency_detector, load_resnet18, split_paths_three_way
from scripts.mahalanobis_detector import FeatureExtractor, fit_mahalanobis, mahalanobis_score
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "paper_ready"


def load_image_paths(folder: Path, prefix: str | None = None) -> List[Path]:
    paths = sorted(
        p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if prefix:
        paths = [p for p in paths if p.name.startswith(prefix)]
    return paths


def create_category_lists() -> List[Tuple[str, List[Path], int]]:
    base = ROOT
    clean_dir = base / "data" / "clean"
    adv_dir = base / "data" / "adversarial"
    bim_dir = base / "data" / "bim_adversarial"
    cw_dir = base / "data" / "cw_adversarial"
    df_dir = base / "data" / "deepfool_adversarial"
    unseen_clean_dir = base / "data" / "unseen_test"
    unseen_adv_dir = base / "data" / "unseen_test_adversarial"

    return [
        ("Clean", load_image_paths(clean_dir), 0),
        ("FGSM", load_image_paths(adv_dir, prefix="fgsm_"), 1),
        ("PGD", load_image_paths(adv_dir, prefix="pgd_"), 1),
        ("BIM", load_image_paths(bim_dir), 1),
        ("CW", load_image_paths(cw_dir), 1),
        ("DeepFool", load_image_paths(df_dir), 1),
        ("UnseenClean", load_image_paths(unseen_clean_dir), 0),
        ("UnseenAttack", load_image_paths(unseen_adv_dir), 1),
    ]


def load_dual_detector(path: Path, device: torch.device) -> DualDomainDetector:
    model = DualDomainDetector().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_mahalanobis_model(clean_paths: List[Path], resnet, extractor: FeatureExtractor, device: torch.device):
    feat_list = []
    labels = []
    with torch.no_grad():
        for path in clean_paths:
            img = load_image_tensor(path, IMAGE_SIZE, device)
            _ = resnet(img)
            feat_list.append(extractor.get().squeeze(0).cpu())
            labels.append(int(resnet(img).argmax(dim=1).item()))
    feats = torch.stack(feat_list)
    labels = torch.tensor(labels, dtype=torch.long)
    return fit_mahalanobis(feats, labels, num_classes=1000)


def compute_frequency_scores(paths: List[Path], frequency_model: FrequencyDetector, device: torch.device) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for path in paths:
            img = load_image_tensor(path, IMAGE_SIZE, device).to(device)
            score = float(frequency_model.predict_proba(img).item())
            scores.append(score)
    return np.array(scores, dtype=np.float64)


def compute_mahalanobis_scores(paths: List[Path], resnet, extractor: FeatureExtractor, class_means: torch.Tensor, precision: torch.Tensor, maha_mean: float, maha_std: float, device: torch.device) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for path in paths:
            img = load_image_tensor(path, IMAGE_SIZE, device).to(device)
            _ = resnet(img)
            feat = extractor.get().squeeze(0)
            raw = float(mahalanobis_score(feat, class_means, precision))
            z = (raw - maha_mean) / maha_std
            scores.append(z)
    return np.array(scores, dtype=np.float64)


def compute_dual_scores(paths: List[Path], dual_model: DualDomainDetector, device: torch.device) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for path in paths:
            img = load_image_tensor(path, IMAGE_SIZE, device).to(device)
            score = float(dual_model.predict_proba(img).item())
            scores.append(score)
    return np.array(scores, dtype=np.float64)


def predict_with_threshold(scores: np.ndarray, threshold: float, higher_is_positive: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if higher_is_positive:
        return (scores >= threshold).astype(int)
    return (scores < threshold).astype(int)


def select_threshold(scores: np.ndarray, labels: np.ndarray, higher_is_positive: bool = True) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float(np.median(scores))

    adjusted_scores = scores if higher_is_positive else -scores
    fpr, tpr, thresholds = roc_curve(labels, adjusted_scores)
    if len(thresholds) == 0:
        return float(np.median(scores))

    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    return float(-best_threshold) if not higher_is_positive else best_threshold


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float, higher_is_positive: bool = True) -> Dict[str, float]:
    preds = predict_with_threshold(scores, threshold, higher_is_positive=higher_is_positive)
    metrics = compute_confusion_metrics(y_true, preds)
    metric_scores = scores if higher_is_positive else -scores
    if len(np.unique(y_true)) < 2:
        roc_auc = 0.5
        average_precision = 0.5
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, metric_scores))
        except ValueError:
            roc_auc = 0.5
        try:
            average_precision = float(average_precision_score(y_true, metric_scores))
        except ValueError:
            average_precision = 0.5
    return {
        "accuracy": metrics["accuracy"],
        "tpr": metrics["tpr"],
        "fpr": metrics["fpr"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def bootstrap_confidence_intervals(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 200, seed: int = 0, higher_is_positive: bool = True) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        sample_scores = scores[idx]
        thr = select_threshold(sample_scores, sample_true, higher_is_positive=higher_is_positive)
        metrics = evaluate_scores(sample_true, sample_scores, thr, higher_is_positive=higher_is_positive)
        estimates.append((metrics["accuracy"], metrics["tpr"], metrics["fpr"], metrics["roc_auc"], metrics["average_precision"]))
    acc_vals = np.array([e[0] for e in estimates])
    tpr_vals = np.array([e[1] for e in estimates])
    fpr_vals = np.array([e[2] for e in estimates])
    roc_vals = np.array([e[3] for e in estimates])
    pr_vals = np.array([e[4] for e in estimates])
    return {
        "accuracy": {"estimate": float(np.mean(acc_vals)), "low": float(np.percentile(acc_vals, 2.5)), "high": float(np.percentile(acc_vals, 97.5))},
        "tpr": {"estimate": float(np.mean(tpr_vals)), "low": float(np.percentile(tpr_vals, 2.5)), "high": float(np.percentile(tpr_vals, 97.5))},
        "fpr": {"estimate": float(np.mean(fpr_vals)), "low": float(np.percentile(fpr_vals, 2.5)), "high": float(np.percentile(fpr_vals, 97.5))},
        "roc_auc": {"estimate": float(np.mean(roc_vals)), "low": float(np.percentile(roc_vals, 2.5)), "high": float(np.percentile(roc_vals, 97.5))},
        "average_precision": {"estimate": float(np.mean(pr_vals)), "low": float(np.percentile(pr_vals, 2.5)), "high": float(np.percentile(pr_vals, 97.5))},
    }


def save_curves(out_dir: Path, method_name: str, y_true: np.ndarray, scores: np.ndarray, higher_is_positive: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_scores = scores if higher_is_positive else -scores
    fpr, tpr, _ = roc_curve(y_true, plot_scores)
    precision, recall, _ = precision_recall_curve(y_true, plot_scores)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, lw=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title(f"ROC - {method_name}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(recall, precision, lw=2)
    axes[1].set_title(f"PR - {method_name}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    fig.tight_layout()
    fig.savefig(out_dir / f"{method_name.lower().replace(' ', '_')}_curves.png", dpi=200)
    plt.close(fig)


def summarize_results(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = ["accuracy", "tpr", "fpr", "precision", "f1", "roc_auc", "average_precision"]
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.array([r[key] for r in results], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def summarize_breakdown(breakdown: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not breakdown:
        return summary

    attack_names = sorted({row["attack"] for row in breakdown})
    metric_names = ["accuracy", "tpr", "fpr", "precision", "f1", "roc_auc", "average_precision"]
    for attack_name in attack_names:
        rows = [row for row in breakdown if row["attack"] == attack_name]
        summary[attack_name] = {}
        for metric_name in metric_names:
            values = np.array([row[metric_name] for row in rows], dtype=np.float64)
            summary[attack_name][metric_name] = float(np.mean(values))
    return summary


def evaluate_method_over_seeds(method_name: str, categories: List[Tuple[str, List[Path], int]], seeds: List[int], frequency_model: FrequencyDetector, dual_model: DualDomainDetector, resnet, extractor: FeatureExtractor, device: torch.device) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, List[float]], List[Dict[str, float]]]:
    test_results: List[Dict[str, float]] = []
    unseen_results: List[Dict[str, float]] = []
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    breakdown_rows: List[Dict[str, float]] = []

    for seed in seeds:
        split_categories = []
        for idx, (name, paths, expected) in enumerate(categories):
            if name in {"UnseenClean", "UnseenAttack"}:
                continue
            train_paths, val_paths, test_paths = split_paths_three_way(paths, seed=seed + idx)
            split_categories.append((name, train_paths, val_paths, test_paths, expected))

        clean_train_paths = next(paths for name, train_paths, _, _, _ in split_categories if name == "Clean")
        class_means, precision = build_mahalanobis_model(clean_train_paths, resnet, extractor, device)
        clean_train_scores = compute_mahalanobis_scores(clean_train_paths, resnet, extractor, class_means, precision, 0.0, 1.0, device)
        maha_mean = float(np.mean(clean_train_scores))
        maha_std = float(np.std(clean_train_scores)) + 1e-8

        train_scores = []
        val_scores = []
        test_scores = []
        train_labels = []
        val_labels = []
        test_labels = []

        for name, train_paths, val_paths, test_paths, expected in split_categories:
            if method_name == "frequency":
                train_score = compute_frequency_scores(train_paths, frequency_model, device)
                val_score = compute_frequency_scores(val_paths, frequency_model, device)
                test_score = compute_frequency_scores(test_paths, frequency_model, device)
            elif method_name == "mahalanobis":
                train_score = compute_mahalanobis_scores(train_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
                val_score = compute_mahalanobis_scores(val_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
                test_score = compute_mahalanobis_scores(test_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
            elif method_name == "dual":
                train_score = compute_dual_scores(train_paths, dual_model, device)
                val_score = compute_dual_scores(val_paths, dual_model, device)
                test_score = compute_dual_scores(test_paths, dual_model, device)
            else:
                train_freq = compute_frequency_scores(train_paths, frequency_model, device)
                val_freq = compute_frequency_scores(val_paths, frequency_model, device)
                test_freq = compute_frequency_scores(test_paths, frequency_model, device)
                train_maha = compute_mahalanobis_scores(train_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
                val_maha = compute_mahalanobis_scores(val_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
                test_maha = compute_mahalanobis_scores(test_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
                train_score = np.column_stack([train_freq, train_maha])
                val_score = np.column_stack([val_freq, val_maha])
                test_score = np.column_stack([test_freq, test_maha])

            train_scores.append(train_score)
            val_scores.append(val_score)
            test_scores.append(test_score)
            train_labels.append(np.full(len(train_paths), expected, dtype=int))
            val_labels.append(np.full(len(val_paths), expected, dtype=int))
            test_labels.append(np.full(len(test_paths), expected, dtype=int))

        higher_is_positive = method_name != "mahalanobis"
        if method_name == "fusion":
            X_train = np.vstack([arr for arr in train_scores])
            X_val = np.vstack([arr for arr in val_scores])
            X_test = np.vstack([arr for arr in test_scores])
            y_train = np.concatenate(train_labels)
            y_val = np.concatenate(val_labels)
            y_test = np.concatenate(test_labels)
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
            clf.fit(X_train, y_train)
            val_scores_for_threshold = clf.predict_proba(X_val)[:, 1]
            test_scores_for_eval = clf.predict_proba(X_test)[:, 1]
            threshold = select_threshold(val_scores_for_threshold, y_val, higher_is_positive=higher_is_positive)
            metrics = evaluate_scores(y_test, test_scores_for_eval, threshold, higher_is_positive=higher_is_positive)
        else:
            y_val = np.concatenate(val_labels)
            y_test = np.concatenate(test_labels)
            val_scores_for_threshold = np.concatenate([arr for arr in val_scores])
            test_scores_for_eval = np.concatenate([arr for arr in test_scores])
            threshold = select_threshold(val_scores_for_threshold, y_val, higher_is_positive=higher_is_positive)
            metrics = evaluate_scores(y_test, test_scores_for_eval, threshold, higher_is_positive=higher_is_positive)

        test_results.append(metrics)
        pooled_scores.extend(test_scores_for_eval.tolist())
        pooled_labels.extend(y_test.tolist())

        for name, _, _, test_paths, expected in split_categories:
            if name in {"UnseenClean", "UnseenAttack"}:
                continue
            if method_name == "frequency":
                category_scores = compute_frequency_scores(test_paths, frequency_model, device)
            elif method_name == "mahalanobis":
                category_scores = compute_mahalanobis_scores(test_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
            elif method_name == "dual":
                category_scores = compute_dual_scores(test_paths, dual_model, device)
            else:
                freq_scores = compute_frequency_scores(test_paths, frequency_model, device)
                maha_scores = compute_mahalanobis_scores(test_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
                category_scores = np.column_stack([freq_scores, maha_scores])
                category_scores = clf.predict_proba(category_scores)[:, 1]
            category_labels = np.full(len(test_paths), expected, dtype=int)
            category_metrics = evaluate_scores(category_labels, category_scores, threshold, higher_is_positive=higher_is_positive)
            breakdown_rows.append({"attack": name, **category_metrics})

        unseen_clean_paths = load_image_paths(ROOT / "data" / "unseen_test")
        unseen_attack_paths = load_image_paths(ROOT / "data" / "unseen_test_adversarial")
        if method_name == "frequency":
            unseen_clean_scores = compute_frequency_scores(unseen_clean_paths, frequency_model, device)
            unseen_attack_scores = compute_frequency_scores(unseen_attack_paths, frequency_model, device)
            unseen_scores = np.concatenate([unseen_clean_scores, unseen_attack_scores])
        elif method_name == "mahalanobis":
            unseen_clean_scores = compute_mahalanobis_scores(unseen_clean_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
            unseen_attack_scores = compute_mahalanobis_scores(unseen_attack_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
            unseen_scores = np.concatenate([unseen_clean_scores, unseen_attack_scores])
        elif method_name == "dual":
            unseen_clean_scores = compute_dual_scores(unseen_clean_paths, dual_model, device)
            unseen_attack_scores = compute_dual_scores(unseen_attack_paths, dual_model, device)
            unseen_scores = np.concatenate([unseen_clean_scores, unseen_attack_scores])
        else:
            unseen_clean_freq = compute_frequency_scores(unseen_clean_paths, frequency_model, device)
            unseen_attack_freq = compute_frequency_scores(unseen_attack_paths, frequency_model, device)
            unseen_clean_maha = compute_mahalanobis_scores(unseen_clean_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
            unseen_attack_maha = compute_mahalanobis_scores(unseen_attack_paths, resnet, extractor, class_means, precision, maha_mean, maha_std, device)
            unseen_features = np.vstack([np.column_stack([unseen_clean_freq, unseen_clean_maha]), np.column_stack([unseen_attack_freq, unseen_attack_maha])])
            unseen_y = np.array([0] * len(unseen_clean_paths) + [1] * len(unseen_attack_paths), dtype=int)
            unseen_scores = clf.predict_proba(unseen_features)[:, 1]
            unseen_results.append(evaluate_scores(unseen_y, unseen_scores, threshold, higher_is_positive=higher_is_positive))
            continue

        unseen_y = np.array([0] * len(unseen_clean_paths) + [1] * len(unseen_attack_paths), dtype=int)
        unseen_results.append(evaluate_scores(unseen_y, unseen_scores, threshold, higher_is_positive=higher_is_positive))

    return test_results, unseen_results, {"scores": pooled_scores, "labels": pooled_labels}, breakdown_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    categories = create_category_lists()
    seeds = [0, 1, 2, 3, 4]

    frequency_model = load_frequency_detector(ROOT / "models" / "detector_frequency.pt", DEVICE)
    dual_model = load_dual_detector(ROOT / "models" / "detector_dual.pt", DEVICE)
    resnet = load_resnet18(DEVICE)
    extractor = FeatureExtractor(resnet)

    methods = ["frequency", "mahalanobis", "dual", "fusion"]
    all_results = {}
    all_unseen_results = {}
    all_breakdowns = {}
    for method in methods:
        test_results, unseen_results, pooled, breakdown_rows = evaluate_method_over_seeds(method, categories, seeds, frequency_model, dual_model, resnet, extractor, DEVICE)
        all_results[method] = test_results
        all_unseen_results[method] = unseen_results
        all_breakdowns[method] = breakdown_rows
        summary = summarize_results(test_results)
        unseen_summary = summarize_results(unseen_results)
        print(f"\nMethod: {method}")
        print("Test set summary (mean ± std across seeds)")
        for key in ["accuracy", "tpr", "fpr", "roc_auc", "average_precision"]:
            print(f"  {key}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")
        print("Unseen set summary")
        for key in ["accuracy", "tpr", "fpr", "roc_auc", "average_precision"]:
            print(f"  {key}: {unseen_summary[key]['mean']:.3f} ± {unseen_summary[key]['std']:.3f}")

        ci = bootstrap_confidence_intervals(np.array(pooled["labels"], dtype=int), np.array(pooled["scores"], dtype=float), higher_is_positive=method != "mahalanobis")
        print("Bootstrap 95% CI (test set)")
        for key, values in ci.items():
            print(f"  {key}: {values['estimate']:.3f} [{values['low']:.3f}, {values['high']:.3f}]")

        breakdown_summary = summarize_breakdown(breakdown_rows)
        print("Per-attack breakdown")
        for attack_name, metrics in breakdown_summary.items():
            print(f"  {attack_name}: accuracy={metrics['accuracy']:.3f}, roc_auc={metrics['roc_auc']:.3f}, fpr={metrics['fpr']:.3f}")

        save_curves(OUTPUT_DIR, method.capitalize(), np.array(pooled["labels"], dtype=int), np.array(pooled["scores"], dtype=float), higher_is_positive=method != "mahalanobis")

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"test": all_results, "unseen": all_unseen_results, "breakdown": all_breakdowns}, handle, indent=2)

    with (OUTPUT_DIR / "per_attack_breakdown.json").open("w", encoding="utf-8") as handle:
        json.dump(all_breakdowns, handle, indent=2)

    with (OUTPUT_DIR / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "accuracy", "tpr", "fpr", "roc_auc", "average_precision", "unseen_accuracy", "unseen_roc_auc"])
        for method in methods:
            summary = summarize_results(all_results[method])
            unseen_summary = summarize_results(all_unseen_results[method])
            writer.writerow([
                method,
                summary["accuracy"]["mean"],
                summary["tpr"]["mean"],
                summary["fpr"]["mean"],
                summary["roc_auc"]["mean"],
                summary["average_precision"]["mean"],
                unseen_summary["accuracy"]["mean"],
                unseen_summary["roc_auc"]["mean"],
            ])

    print(f"\nCurves, summary, breakdown, and metrics written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
