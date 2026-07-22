"""Noise consistency detector for CW and DeepFool attacks.

Detects adversarial examples by measuring prediction variance under
small random Gaussian perturbations. Clean images should remain stable,
while CW and DeepFool examples often flip predictions near the boundary.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/detector_frequency.pt")
THRESHOLD_STEPS = [round(0.001 + i * (0.1 - 0.001) / 19, 6) for i in range(20)]
RANDOM_SEED = 42


def load_detector(path: Path, device: torch.device) -> FrequencyDetector:
    model = FrequencyDetector().to(device)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def gather_paths() -> List[Tuple[str, List[Path], int]]:
    base = Path(".")
    clean_dir = base / "data" / "clean"
    cw_dir = base / "data" / "cw_adversarial"
    deepfool_dir = base / "data" / "deepfool_adversarial"

    def find_images(folder: Path) -> List[Path]:
        return sorted(
            [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

    return [
        ("Clean", find_images(clean_dir), 0),
        ("CW", find_images(cw_dir), 1),
        ("DeepFool", find_images(deepfool_dir), 1),
    ]


def split_paths(paths: List[Path], train_ratio: float = 0.7) -> Tuple[List[Path], List[Path]]:
    if len(paths) < 2:
        return paths, []

    shuffled = list(paths)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def noise_variance(model: FrequencyDetector, image: torch.Tensor) -> float:
    predictions: List[float] = []

    for _ in range(20):
        noise = torch.randn_like(image) * 0.01
        noisy = torch.clamp(image + noise, 0.0, 1.0)
        predictions.append(model.predict_proba(noisy).item())

    return torch.tensor(predictions, dtype=torch.float32).std(unbiased=False).item()


def compute_variances(model: FrequencyDetector, paths: List[Path]) -> Dict[Path, float]:
    return {
        path: noise_variance(model, load_image_tensor(path, IMAGE_SIZE, DEVICE))
        for path in paths
    }


def evaluate_category(
    paths: List[Path],
    expected: int,
    threshold: float,
    variance_cache: Dict[Path, float],
    model: FrequencyDetector,
) -> Tuple[int, int, int, float]:
    total = 0
    freq_correct = 0
    noise_correct = 0
    variance_sum = 0.0

    for path in paths:
        total += 1
        image = load_image_tensor(path, IMAGE_SIZE, DEVICE)
        variance = variance_cache[path]
        variance_sum += variance

        original_prob = predict_prob(model, image)
        freq_pred = 1 if original_prob > 0.5 else 0
        freq_correct += int(freq_pred == expected)

        flagged = variance > threshold
        noise_pred = 1 if flagged else 0
        noise_correct += int(noise_pred == expected)

    avg_variance = (variance_sum / total) if total > 0 else 0.0
    return total, freq_correct, noise_correct, avg_variance


def evaluate_cascade(
    paths: List[Path],
    expected: int,
    freq_threshold_low: float,
    freq_threshold_high: float,
    variance_threshold: float,
    variance_cache: Dict[Path, float],
    model: FrequencyDetector,
) -> Tuple[int, int, int, float]:
    total = 0
    correct = 0
    uncertain_count = 0

    for path in paths:
        total += 1
        image = load_image_tensor(path, IMAGE_SIZE, DEVICE)
        variance = variance_cache[path]

        freq_prob = predict_prob(model, image)
        if freq_prob < freq_threshold_low or freq_prob > freq_threshold_high:
            pred = 1 if freq_prob > 0.5 else 0
        else:
            uncertain_count += 1
            pred = 1 if variance > variance_threshold else 0

        correct += int(pred == expected)

    uncertain_ratio = (uncertain_count / total * 100.0) if total > 0 else 0.0
    return total, correct, uncertain_count, uncertain_ratio


def search_optimal_threshold(
    model: FrequencyDetector,
    categories: List[Tuple[str, List[Path], int]],
    variance_caches: Dict[str, Dict[Path, float]],
) -> Tuple[float, List[Dict[str, float]]]:
    search_results: List[Dict[str, float]] = []

    for threshold in THRESHOLD_STEPS:
        result = {
            "threshold": threshold,
            "clean_fp": 0.0,
            "cw_rate": 0.0,
            "deepfool_rate": 0.0,
            "score": 0.0,
        }

        for name, paths, expected in categories:
            train_paths, _ = split_paths(paths)
            variance_cache = variance_caches[name]
            total, _, noise_correct, _ = evaluate_category(
                train_paths,
                expected,
                threshold,
                variance_cache,
                model,
            )
            rate = (noise_correct / total * 100) if total > 0 else 0.0

            if name == "Clean":
                result["clean_fp"] = 100.0 - rate
            elif name == "CW":
                result["cw_rate"] = rate
            elif name == "DeepFool":
                result["deepfool_rate"] = rate

        result["score"] = (
            result["cw_rate"] + result["deepfool_rate"] - 2.0 * result["clean_fp"]
        )
        search_results.append(result)

    best = max(search_results, key=lambda x: x["score"])
    return best["threshold"], search_results


def print_threshold_search(search_results: List[Dict[str, float]]) -> None:
    header = (
        f"{'Threshold':>9} | {'Clean FP':>8} | {'CW Rate':>8} | "
        f"{'DeepFool Rate':>13} | {'Score':>7}"
    )
    sep = "-" * len(header)
    print("\nThreshold search results")
    print(sep)
    print(header)
    print(sep)

    for result in search_results:
        print(
            f"{result['threshold']:9.6f} | "
            f"{result['clean_fp']:8.2f}% | "
            f"{result['cw_rate']:8.2f}% | "
            f"{result['deepfool_rate']:13.2f}% | "
            f"{result['score']:7.2f}"
        )

    print(sep)


def print_category_results(results: List[Tuple[str, int, int, float]]) -> None:
    header = (
        f"{'Category':<10} | {'Images':>6} | {'Correct':>7} | "
        f"{'Detection Rate':>14} | {'Avg Variance':>13}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for name, images, correct, avg_variance in results:
        rate = (correct / images * 100) if images > 0 else 0.0
        print(
            f"{name:<10} | {images:6d} | {correct:7d} | "
            f"{rate:13.2f}% | {avg_variance:13.6f}"
        )

    print(sep)


def print_comparison(
    freq_rates: Dict[str, float],
    noise_rates: Dict[str, float],
    clean_fp: Dict[str, float],
) -> None:
    header = f"{'Attack':<9} | {'Freq Only':>9} | {'Noise Consistency':>17} | {'Improvement':>11}"
    sep = "-" * len(header)
    print("\nComparison with frequency detector alone")
    print(sep)
    print(header)
    print(sep)

    for attack in ["CW", "DeepFool"]:
        freq = freq_rates.get(attack, 0.0)
        noise = noise_rates.get(attack, 0.0)
        improvement = noise - freq
        sign = "+" if improvement >= 0 else ""
        print(
            f"{attack:<9} | {freq:9.2f}% | {noise:17.2f}% | {sign}{improvement:10.2f}%"
        )

    fp_freq = clean_fp.get("freq", 0.0)
    fp_noise = clean_fp.get("noise", 0.0)
    improvement = fp_noise - fp_freq
    sign = "+" if improvement >= 0 else ""
    print(sep)
    print(
        f"{'Clean FP':<9} | {fp_freq:9.2f}% | {fp_noise:17.2f}% | {sign}{improvement:10.2f}%"
    )
    print(sep)


def print_cascade_results(
    cascade_rates: Dict[str, float],
    uncertain_rates: Dict[str, float],
    freq_rates: Dict[str, float],
    noise_rates: Dict[str, float],
) -> None:
    header = (
        f"{'Category':<10} | {'Cascade Rate':>12} | {'Noise-check %':>13} | "
        f"{'Freq Only':>10} | {'Noise Only':>11}"
    )
    sep = "-" * len(header)
    print("\nCascade evaluation results")
    print(sep)
    print(header)
    print(sep)

    for name in ["Clean", "CW", "DeepFool"]:
        print(
            f"{name:<10} | {cascade_rates.get(name, 0.0):12.2f}% | "
            f"{uncertain_rates.get(name, 0.0):13.2f}% | "
            f"{freq_rates.get(name, 0.0):10.2f}% | "
            f"{noise_rates.get(name, 0.0):11.2f}%"
        )
    print(sep)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect CW and DeepFool attacks by prediction variance under noise"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Variance threshold for adversarial detection; if omitted the script chooses the best threshold automatically",
    )
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    model = load_detector(MODEL_PATH, DEVICE)
    categories = gather_paths()

    variance_caches: Dict[str, Dict[Path, float]] = {}
    for name, paths, _ in categories:
        train_paths, _ = split_paths(paths)
        variance_caches[name] = compute_variances(model, train_paths)

    optimal_threshold, search_results = search_optimal_threshold(
        model,
        categories,
        variance_caches,
    )
    print_threshold_search(search_results)
    print(f"\nOptimal threshold selected: {optimal_threshold:.6f}\n")

    threshold = args.threshold if args.threshold is not None else optimal_threshold

    category_results: List[Tuple[str, int, int, float]] = []
    freq_rates: Dict[str, float] = {}
    noise_rates: Dict[str, float] = {}
    cascade_rates: Dict[str, float] = {}
    uncertain_rates: Dict[str, float] = {}
    clean_fp = {"freq": 0.0, "noise": 0.0}

    for name, paths, expected in categories:
        _, test_paths = split_paths(paths)
        variance_cache = compute_variances(model, test_paths)
        total, freq_correct, noise_correct, avg_variance = evaluate_category(
            test_paths,
            expected,
            threshold,
            variance_cache,
            model,
        )

        category_results.append((name, total, noise_correct, avg_variance))
        freq_rates[name] = (freq_correct / total * 100) if total > 0 else 0.0
        noise_rates[name] = (noise_correct / total * 100) if total > 0 else 0.0

        cascade_total, cascade_correct, uncertain_count, uncertain_ratio = evaluate_cascade(
            test_paths,
            expected,
            0.3,
            0.7,
            threshold,
            variance_cache,
            model,
        )
        cascade_rates[name] = (cascade_correct / cascade_total * 100.0) if cascade_total > 0 else 0.0
        uncertain_rates[name] = uncertain_ratio

        if name == "Clean":
            clean_fp["freq"] = 100.0 - freq_rates[name]
            clean_fp["noise"] = 100.0 - noise_rates[name]

    print("\nNoise consistency detection results")
    print_category_results(category_results)
    print_comparison(freq_rates, noise_rates, clean_fp)
    print_cascade_results(cascade_rates, uncertain_rates, freq_rates, noise_rates)


def predict_prob(model: FrequencyDetector, tensor: torch.Tensor) -> float:
    with torch.no_grad():
        return model.predict_proba(tensor.to(DEVICE)).item()


if __name__ == "__main__":
    main()
