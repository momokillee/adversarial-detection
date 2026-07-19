"""
Mahalanobis Distance Adversarial Detector
Following Lee et al. NeurIPS 2018 - A Simple Unified Framework for
Detecting Out-of-Distribution Samples and Adversarial Attacks

Uses pretrained ResNet18 (512-dim features) as feature extractor.
Tests against CW and DeepFool which frequency domain misses completely.

CHANGES FROM ORIGINAL:
  1. Scores are dumped to scores_dump.json after Step 3 so they can be
     re-analyzed without re-running feature extraction.
  2. Step 4 now ALSO runs a fine-grained threshold search restricted to
     just the clean+CW score range, plus a threshold-free ROC-AUC for
     clean-vs-CW separability. The original search_threshold() used one
     grid spanning clean+CW+DeepFool scores together -- since DeepFool
     scores are ~10^6 and clean/CW scores are ~10^2-10^3, the step size
     was far too coarse to ever land a candidate threshold inside the
     narrow clean/CW range, so CW showed 0% regardless of whether real
     separation existed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import statistics
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torchvision.models as models
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────
# Feature extraction hook
# ──────────────────────────────────────────────

class FeatureExtractor:
    """
    Hooks into ResNet18 avgpool layer to extract
    512-dimensional feature vectors before final classifier.
    """
    def __init__(self, model: nn.Module):
        self.features = None
        model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output shape: (batch, 512, 1, 1) -> flatten to (batch, 512)
        self.features = output.flatten(1)

    def get(self) -> torch.Tensor:
        return self.features


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_images_from_folder(
    folder: Path,
    limit: int = None,
    prefix: str = None
) -> List[Path]:
    paths = sorted(
        p for p in folder.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if prefix:
        paths = [p for p in paths if p.name.startswith(prefix)]
    if limit:
        paths = paths[:limit]
    return paths


def extract_features(
    model: nn.Module,
    extractor: FeatureExtractor,
    paths: List[Path],
    device: torch.device
) -> torch.Tensor:
    """Extract 512-dim features from ResNet18 avgpool for all images."""
    all_features = []
    model.eval()
    with torch.no_grad():
        for i, path in enumerate(paths):
            img = load_image_tensor(path, IMAGE_SIZE, device)
            _ = model(img)
            all_features.append(extractor.get().squeeze(0))
            if (i + 1) % 100 == 0:
                print(f"  Extracted features: {i+1}/{len(paths)}")
    return torch.stack(all_features)  # (N, 512)


# ──────────────────────────────────────────────
# Mahalanobis fitting
# ──────────────────────────────────────────────

def fit_mahalanobis(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute class-conditional means and shared precision matrix
    from clean training features.

    Returns:
        class_means: (num_classes, 512)
        precision:   (512, 512)
    """
    feat_dim = features.shape[1]
    class_means = []
    pooled_cov = torch.zeros(feat_dim, feat_dim)
    classes_found = 0

    unique_classes = labels.unique()
    print(f"  Found {len(unique_classes)} unique predicted classes")

    for c in unique_classes.tolist():
        mask = labels == c
        class_feats = features[mask]
        mu_c = class_feats.mean(dim=0)
        diff = class_feats - mu_c.unsqueeze(0)
        pooled_cov += diff.T @ diff
        classes_found += 1

    class_means_list = []
    for c in unique_classes.tolist():
        mask = labels == c
        class_means_list.append(features[mask].mean(dim=0))

    pooled_cov /= features.shape[0]
    pooled_cov += torch.eye(feat_dim) * 1e-5
    precision = torch.linalg.inv(pooled_cov)
    class_means_tensor = torch.stack(class_means_list)

    return class_means_tensor, precision


def mahalanobis_score(
    feature: torch.Tensor,
    class_means: torch.Tensor,
    precision: torch.Tensor
) -> float:
    """
    Compute Mahalanobis distance score for one feature vector.
    Higher score = more likely clean.
    Lower score = more likely adversarial.

    Score = max_c [ -(f - mu_c)^T * precision * (f - mu_c) ]
    """
    scores = []
    for mu_c in class_means:
        diff = feature - mu_c
        score = -diff @ precision @ diff
        scores.append(score.item())
    return max(scores)


# ──────────────────────────────────────────────
# Original threshold search (kept for FGSM/PGD/BIM/DeepFool, unchanged)
# ──────────────────────────────────────────────

def search_threshold(
    clean_scores: List[float],
    cw_scores: List[float],
    deepfool_scores: List[float],
    n_steps: int = 50
) -> Tuple[float, List[Dict]]:
    """
    Search for optimal threshold maximising:
    score = CW_detection + DeepFool_detection - 2 * clean_FP

    NOTE: kept as-is for backward compatibility / comparison, but see
    fine_threshold_search() below for the CW-specific fix.
    """
    all_scores = clean_scores + cw_scores + deepfool_scores
    if not all_scores:
        return 0.0, []
    min_s = min(all_scores)
    max_s = max(all_scores)
    step = (max_s - min_s) / n_steps if n_steps else 1.0

    results = []
    for i in range(n_steps + 1):
        threshold = min_s + i * step
        clean_fp = sum(
            1 for s in clean_scores if s < threshold
        ) / len(clean_scores) * 100
        cw_det = (
            sum(1 for s in cw_scores if s < threshold) / len(cw_scores) * 100
            if cw_scores else 0.0
        )
        df_det = (
            sum(1 for s in deepfool_scores if s < threshold) / len(deepfool_scores) * 100
            if deepfool_scores else 0.0
        )
        score = cw_det + df_det - 2 * clean_fp
        results.append({
            "threshold": threshold,
            "clean_fp": clean_fp,
            "cw_rate": cw_det,
            "deepfool_rate": df_det,
            "score": score
        })

    best = max(results, key=lambda x: x["score"])
    return best["threshold"], results


# ──────────────────────────────────────────────
# NEW: fine-grained CW-vs-clean diagnostic
# ──────────────────────────────────────────────

def describe(name: str, scores: List[float]) -> None:
    s = sorted(scores)
    n = len(s)
    print(f"\n{name} (n={n})")
    print(f"  mean:   {statistics.mean(s):.2f}")
    print(f"  stdev:  {statistics.stdev(s):.2f}")
    print(f"  min:    {s[0]:.2f}")
    print(f"  p25:    {s[int(0.25 * n)]:.2f}")
    print(f"  median: {s[int(0.50 * n)]:.2f}")
    print(f"  p75:    {s[int(0.75 * n)]:.2f}")
    print(f"  max:    {s[-1]:.2f}")


def fine_threshold_search(
    clean_scores: List[float],
    cw_scores: List[float],
    n_steps: int = 2000,
) -> Tuple[float, List[Dict]]:
    """
    Threshold search restricted ONLY to the clean+CW score range,
    so the grid resolution is fine enough to actually find separation
    if it exists, instead of being swamped by DeepFool/FGSM/PGD scale.
    """
    combined = clean_scores + cw_scores
    min_s, max_s = min(combined), max(combined)
    step = (max_s - min_s) / n_steps
    min_s -= step * 2
    max_s += step * 2

    results = []
    for i in range(n_steps + 1):
        threshold = min_s + i * step
        clean_fp = sum(1 for s in clean_scores if s < threshold) / len(clean_scores) * 100
        cw_det = sum(1 for s in cw_scores if s < threshold) / len(cw_scores) * 100
        score = cw_det - 2 * clean_fp
        results.append({
            "threshold": threshold,
            "clean_fp": clean_fp,
            "cw_rate": cw_det,
            "score": score,
        })

    best = max(results, key=lambda x: x["score"])
    return best["threshold"], results


def roc_auc(clean_scores: List[float], cw_scores: List[float]) -> float:
    """
    Mann-Whitney U based AUC: probability that a random CW score is
    lower (more adversarial-looking) than a random clean score.
    Threshold-independent -- gives the ceiling on separability
    regardless of grid resolution. 0.5 = no separation, 1.0 = perfect.
    """
    combined = [(s, 0) for s in clean_scores] + [(s, 1) for s in cw_scores]
    combined.sort(key=lambda x: x[0])

    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n_cw = len(cw_scores)
    n_clean = len(clean_scores)
    rank_sum_cw = sum(ranks[idx] for idx, (val, label) in enumerate(combined) if label == 1)
    u_cw_lower = n_cw * n_clean + (n_cw * (n_cw + 1)) / 2 - rank_sum_cw
    return u_cw_lower / (n_cw * n_clean)


def run_cw_diagnostic(clean_scores: List[float], cw_scores: List[float]) -> None:
    print("\n" + "=" * 60)
    print("CW vs CLEAN FINE-GRAINED SEPARABILITY DIAGNOSTIC")
    print("=" * 60)

    describe("Clean", clean_scores)
    describe("CW", cw_scores)

    auc = roc_auc(clean_scores, cw_scores)
    print(f"\nROC-AUC (clean vs CW, threshold-free): {auc:.4f}")
    if auc < 0.55:
        print("  -> essentially no separation exists in this feature space.")
        print("     A finer threshold search will not help much further.")
    elif auc < 0.7:
        print("  -> weak but real separation. A well-tuned threshold may")
        print("     recover some CW detections, with a real FP tradeoff.")
    else:
        print("  -> meaningful separation exists. The original threshold")
        print("     grid was almost certainly too coarse to find it.")

    best_threshold, results = fine_threshold_search(clean_scores, cw_scores)
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\n{'Threshold':>14} | {'Clean FP':>8} | {'CW Rate':>8} | {'Score':>7}")
    print("-" * 50)
    for r in sorted_results[:10]:
        print(f"{r['threshold']:14.2f} | {r['clean_fp']:8.2f}% | {r['cw_rate']:8.2f}% | {r['score']:7.2f}")

    print(f"\nBest fine-grained threshold: {best_threshold:.2f}")
    print("(compare to the original optimal_threshold printed above --")
    print(" if this number sits inside the clean/CW range while the original")
    print(" was off in DeepFool-scale territory, that confirms the original")
    print(" grid was simply too coarse to test this region at all)")


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_category(
    scores: List[float],
    expected_label: int,
    threshold: float
) -> Tuple[int, int]:
    correct = 0
    for s in scores:
        predicted = 0 if s >= threshold else 1
        if predicted == expected_label:
            correct += 1
    return correct, len(scores)


def print_results_table(rows: List[Tuple]):
    header = (
        f"{'Attack':<12} | {'Images':>6} | "
        f"{'Correct':>7} | {'Detection Rate':>14} | {'Avg Score':>12}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, total, correct, avg_score in rows:
        rate = correct / total * 100 if total > 0 else 0
        print(
            f"{name:<12} | {total:6d} | "
            f"{correct:7d} | {rate:13.2f}% | {avg_score:12.2f}"
        )
    print(sep)


def print_comparison_table(
    freq_rates: Dict,
    maha_rates: Dict
):
    header = (
        f"{'Attack':<10} | {'Freq Only':>9} | "
        f"{'Mahalanobis':>11} | {'Improvement':>12}"
    )
    sep = "-" * len(header)
    print("\nComparison: Frequency Detector vs Mahalanobis")
    print(sep)
    print(header)
    print(sep)
    for attack in ["FGSM", "PGD", "BIM", "DeepFool", "CW"]:
        freq = freq_rates.get(attack, 0.0)
        maha = maha_rates.get(attack, 0.0)
        diff = maha - freq
        sign = "+" if diff >= 0 else ""
        print(
            f"{attack:<10} | {freq:9.2f}% | "
            f"{maha:11.2f}% | {sign}{diff:11.2f}%"
        )
    print(sep)

    print("\nCombined cascade (Freq Stage1 + Mahalanobis Stage2):")
    print(sep)
    print(header)
    print(sep)
    for attack in ["FGSM", "PGD", "BIM", "DeepFool", "CW"]:
        freq = freq_rates.get(attack, 0.0)
        maha = maha_rates.get(attack, 0.0)
        # Cascade: if freq catches it use freq, else use mahalanobis
        combined = max(freq, maha)
        print(
            f"{attack:<10} | {freq:9.2f}% | "
            f"{maha:11.2f}% | {combined:11.2f}% (cascade)"
        )
    print(sep)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── Load pretrained ResNet18 ──
    print("Loading pretrained ResNet18 for feature extraction...")
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    ).to(DEVICE)
    model.eval()
    extractor = FeatureExtractor(model)
    print("  ResNet18 loaded — extracting 512-dim penultimate features")

    # ── Step 1: Extract features from clean images ──
    print("\nStep 1: Extracting features from clean images...")
    clean_dir = Path("data/clean")
    clean_paths = load_images_from_folder(clean_dir)
    print(f"  Found {len(clean_paths)} clean images")
    clean_features = extract_features(
        model, extractor, clean_paths, DEVICE
    )

    # Get pseudo-labels using ResNet18 predictions
    print("  Getting pseudo-labels from ResNet18 predictions...")
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for path in clean_paths:
            img = load_image_tensor(path, IMAGE_SIZE, DEVICE)
            out = model(img)
            pseudo_labels.append(out.argmax(dim=1).item())
    pseudo_labels = torch.tensor(pseudo_labels)
    unique, counts = pseudo_labels.unique(return_counts=True)
    print(f"  Labels span {len(unique)} ImageNet classes")

    # ── Step 2: Fit Mahalanobis ──
    print("\nStep 2: Fitting Mahalanobis detector...")
    class_means, precision = fit_mahalanobis(
        clean_features, pseudo_labels, num_classes=1000
    )
    print(f"  Class means: {class_means.shape}")
    print(f"  Precision matrix: {precision.shape}")

    # ── Step 3: Score all images ──
    print("\nStep 3: Computing Mahalanobis scores...")
    print("  Scoring clean images...")
    clean_scores = [
        mahalanobis_score(f, class_means, precision)
        for f in clean_features
    ]
    print(f"  Clean avg score: {sum(clean_scores)/len(clean_scores):.2f}")

    attack_folders = {
        "FGSM": ("data/adversarial", "fgsm_"),
        "PGD":  ("data/adversarial", "pgd_"),
        "BIM":  ("data/bim_adversarial", None),
        "CW":   ("data/cw_adversarial_v2", None),
        "DeepFool": ("data/deepfool_adversarial", None),
    }

    attack_scores = {}
    for name, (folder, prefix) in attack_folders.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"  Skipping {name} — folder not found")
            continue
        print(f"  Scoring {name}...")
        paths = load_images_from_folder(folder_path, prefix=prefix)
        feats = extract_features(model, extractor, paths, DEVICE)
        scores = [
            mahalanobis_score(f, class_means, precision)
            for f in feats
        ]
        attack_scores[name] = scores
        avg = sum(scores) / len(scores)
        print(f"  {name} avg score: {avg:.2f}")

    # ── NEW: dump scores so they can be reanalyzed without re-running ──
    with open("scores_dump.json", "w") as f:
        json.dump({
            "clean": clean_scores,
            **{name: scores for name, scores in attack_scores.items()},
        }, f)
    print("\nSaved scores_dump.json (clean + all attack scores)")

    # ── Step 4: Threshold search ──
    print("\nStep 4: Searching for optimal threshold...")
    cw_scores = attack_scores.get("CW", [])
    df_scores = attack_scores.get("DeepFool", [])

    optimal_threshold, search_results = search_threshold(
        clean_scores, cw_scores, df_scores
    )

    print(
        f"\n{'Threshold':>12} | {'Clean FP':>8} | "
        f"{'CW Rate':>8} | {'DeepFool':>9} | {'Score':>7}"
    )
    print("-" * 60)
    sorted_results = sorted(
        search_results,
        key=lambda x: x["score"],
        reverse=True
    )
    for r in sorted_results[:10]:
        print(
            f"{r['threshold']:12.2f} | {r['clean_fp']:8.2f}% | "
            f"{r['cw_rate']:8.2f}% | "
            f"{r['deepfool_rate']:9.2f}% | {r['score']:7.2f}"
        )
    print(f"\nOptimal threshold: {optimal_threshold:.2f}")

    # ── NEW: fine-grained CW-vs-clean diagnostic (the actual fix) ──
    if cw_scores:
        run_cw_diagnostic(clean_scores, cw_scores)

    # ── Step 5: Final evaluation ──
    print("\nStep 5: Final evaluation")
    print("=" * 60)
    print("MAHALANOBIS DETECTOR RESULTS")
    print("=" * 60)

    rows = []
    maha_rates = {}

    correct, total = evaluate_category(
        clean_scores, 0, optimal_threshold
    )
    avg_s = sum(clean_scores) / len(clean_scores)
    rows.append(("Clean", total, correct, avg_s))
    maha_rates["Clean"] = correct / total * 100

    for name in ["FGSM", "PGD", "BIM", "CW", "DeepFool"]:
        if name not in attack_scores:
            continue
        scores = attack_scores[name]
        correct, total = evaluate_category(scores, 1, optimal_threshold)
        avg_s = sum(scores) / len(scores)
        rows.append((name, total, correct, avg_s))
        maha_rates[name] = correct / total * 100

    print_results_table(rows)

    freq_rates = {
        "FGSM": 100.0,
        "PGD":  100.0,
        "BIM":  100.0,
        "DeepFool": 76.8,
        "CW":   0.0,
    }
    print_comparison_table(freq_rates, maha_rates)


if __name__ == "__main__":
    main()