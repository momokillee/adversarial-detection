"""Noise consistency detector for CW and DeepFool attacks.

Detects adversarial examples by measuring prediction variance under
small random Gaussian perturbations. Clean images should remain stable,
while CW and DeepFool examples often flip predictions near the boundary.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/detector_frequency.pt")
THRESHOLD_STEPS = [round(0.001 + i * (0.1 - 0.001) / 19, 6) for i in range(20)]


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


def noise_variance(model: FrequencyDetector, image: torch.Tensor) -> float:
    predictions: List[float] = []

    for _ in range(20):
        noise = torch.randn_like(image) * 0.01
        noisy = torch.clamp(image + noise, 0.0, 1.0)
        predictions.append(model.predict_proba(noisy).item())

    return torch.tensor(predictions, dtype=torch.float32).std(unbiased=False).item()


def evaluate_image(
    model: FrequencyDetector,
    path: Path,
    threshold: float,
) -> Tuple[float, bool]:
    image = load_image_tensor(path, IMAGE_SIZE, DEVICE)
    variance = noise_variance(model, image)
    flagged = variance > threshold
    return variance, flagged


def evaluate_category(
    model: FrequencyDetector,
    paths: List[Path],
    expected: int,
    threshold: float,
) -> Tuple[int, int, float]:
    total = 0
    correct = 0
    variance_sum = 0.0

    for path in paths:
        total += 1
        variance, flagged = evaluate_image(model, path, threshold)
        predicted = 1 if flagged else 0
        correct += int(predicted == expected)
        variance_sum += variance

    avg_variance = (variance_sum / total) if total > 0 else 0.0
    return total, correct, avg_variance


def search_optimal_threshold(
    model: FrequencyDetector,
    categories: List[Tuple[str, List[Path], int]],
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
            total, correct, _ = evaluate_category(model, paths, expected, threshold)
            rate = (correct / total * 100) if total > 0 else 0.0

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

    model = load_detector(MODEL_PATH, DEVICE)
    categories = gather_paths()

    optimal_threshold, search_results = search_optimal_threshold(model, categories)
    print_threshold_search(search_results)
    print(f"\nOptimal threshold selected: {optimal_threshold:.6f}\n")

    threshold = args.threshold if args.threshold is not None else optimal_threshold

    category_results: List[Tuple[str, int, int, float]] = []
    freq_rates: Dict[str, float] = {}
    noise_rates: Dict[str, float] = {}
    clean_fp = {"freq": 0.0, "noise": 0.0}

    for name, paths, expected in categories:
        total = 0
        freq_correct = 0
        noise_correct = 0
        variance_sum = 0.0

        for path in paths:
            total += 1
            image = load_image_tensor(path, IMAGE_SIZE, DEVICE)

            original_prob = predict_prob(model, image)
            freq_pred = 1 if original_prob > 0.5 else 0
            freq_correct += int(freq_pred == expected)

            variance, flagged = evaluate_image(model, path, threshold)
            noise_pred = 1 if flagged else 0
            noise_correct += int(noise_pred == expected)

            variance_sum += variance

        avg_variance = (variance_sum / total) if total > 0 else 0.0
        category_results.append((name, total, noise_correct, avg_variance))
        freq_rates[name] = (freq_correct / total * 100) if total > 0 else 0.0
        noise_rates[name] = (noise_correct / total * 100) if total > 0 else 0.0

        if name == "Clean":
            clean_fp["freq"] = 100.0 - freq_rates[name]
            clean_fp["noise"] = 100.0 - noise_rates[name]

    print("\nNoise consistency detection results")
    print_category_results(category_results)
    print_comparison(freq_rates, noise_rates, clean_fp)


def predict_prob(model: FrequencyDetector, tensor: torch.Tensor) -> float:
    with torch.no_grad():
        return model.predict_proba(tensor.to(DEVICE)).item()


if __name__ == "__main__":
    main()
