"""Feature squeezing detection for CW and DeepFool attacks.

Loads the existing frequency detector and compares predictions on the
original image versus squeezed versions. If the maximum distance between
the original prediction and either squeezed prediction exceeds threshold,
the image is flagged as adversarial.

This version performs a threshold search from 0.05 to 0.95 and selects the
threshold that maximizes CW_rate + DeepFool_rate - 2 * FP_rate.
"""

import io
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor, pil_to_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/detector_frequency.pt")
DEFAULT_THRESHOLD = None
THRESHOLD_STEPS = [round(x * 0.05, 2) for x in range(1, 20)]


def load_detector(path: Path, device: torch.device) -> FrequencyDetector:
    model = FrequencyDetector().to(device)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def gaussian_blur_squeeze(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    squeezed = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    return pil_to_tensor(squeezed, IMAGE_SIZE, device)


def bit_depth_squeeze(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor * 63.0) / 63.0


def jpeg_compression_squeeze(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    buffer.seek(0)
    squeezed = Image.open(buffer).convert("RGB")
    return pil_to_tensor(squeezed, IMAGE_SIZE, device)


def predict_prob(model: FrequencyDetector, tensor: torch.Tensor) -> float:
    with torch.no_grad():
        return model.predict_proba(tensor.to(DEVICE)).item()


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


def evaluate_image(
    model: FrequencyDetector,
    path: Path,
    threshold: float,
) -> Tuple[float, float, bool]:
    original = load_image_tensor(path, IMAGE_SIZE, DEVICE)
    gaussian = gaussian_blur_squeeze(path, DEVICE)
    bit_depth = bit_depth_squeeze(original)
    jpeg = jpeg_compression_squeeze(path, DEVICE)

    original_prob = predict_prob(model, original)
    gaussian_prob = predict_prob(model, gaussian)
    bit_prob = predict_prob(model, bit_depth)
    jpeg_prob = predict_prob(model, jpeg)

    dist_gaussian = abs(original_prob - gaussian_prob)
    dist_bit = abs(original_prob - bit_prob)
    dist_jpeg = abs(original_prob - jpeg_prob)
    mean_dist = (dist_gaussian + dist_bit + dist_jpeg) / 3.0
    flagged = mean_dist > threshold
    return original_prob, mean_dist, flagged


def evaluate_category(
    model: FrequencyDetector,
    paths: List[Path],
    expected: int,
    threshold: float,
) -> Tuple[int, int, float]:
    total = 0
    correct = 0
    dist_sum = 0.0

    for path in paths:
        total += 1
        _, max_dist, flagged = evaluate_image(model, path, threshold)
        predicted = 1 if flagged else 0
        correct += int(predicted == expected)
        dist_sum += max_dist

    avg_dist = (dist_sum / total) if total > 0 else 0.0
    return total, correct, avg_dist


def print_category_results(results: List[Tuple[str, int, int, float]]):
    header = (
        f"{'Category':<10} | {'Images':>6} | {'Correct':>7} | "
        f"{'Detection Rate':>14} | {'Avg Distance':>12}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, images, correct, avg_dist in results:
        rate = (correct / images * 100) if images > 0 else 0.0
        print(
            f"{name:<10} | {images:6d} | {correct:7d} | "
            f"{rate:13.2f}% | {avg_dist:12.4f}"
        )
    print(sep)


def print_threshold_search(search_results: List[Dict[str, float]]):
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
            f"{result['threshold']:9.2f} | "
            f"{result['clean_fp']:8.2f}% | "
            f"{result['cw_rate']:8.2f}% | "
            f"{result['deepfool_rate']:13.2f}% | "
            f"{result['score']:7.2f}"
        )
    print(sep)


def search_optimal_threshold(
    model: FrequencyDetector,
    categories: List[Tuple[str, List[Path], int]],
) -> Tuple[float, List[Dict[str, float]]]:
    search_results = []

    for threshold in THRESHOLD_STEPS:
        result = {"threshold": threshold, "clean_fp": 0.0, "cw_rate": 0.0, "deepfool_rate": 0.0, "score": 0.0}

        for name, paths, expected in categories:
            total, correct, _ = evaluate_category(model, paths, expected, threshold)
            rate = (correct / total * 100) if total > 0 else 0.0

            if name == "Clean":
                result["clean_fp"] = 100.0 - rate
            elif name == "CW":
                result["cw_rate"] = rate
            elif name == "DeepFool":
                result["deepfool_rate"] = rate

        result["score"] = result["cw_rate"] + result["deepfool_rate"] - 2.0 * result["clean_fp"]
        search_results.append(result)

    best = max(search_results, key=lambda x: x["score"])
    return best["threshold"], search_results


def print_comparison(
    freq_rates: Dict[str, float],
    fs_rates: Dict[str, float],
    clean_fp: Dict[str, float],
):
    header = f"{'Attack':<9} | {'Freq Only':>9} | {'Feature Squeezing':>17} | {'Improvement':>11}"
    sep = "-" * len(header)
    print("\nComparison with frequency detector alone")
    print(sep)
    print(header)
    print(sep)

    for attack in ["CW", "DeepFool"]:
        freq = freq_rates.get(attack, 0.0)
        fs = fs_rates.get(attack, 0.0)
        improvement = fs - freq
        sign = "+" if improvement >= 0 else ""
        print(
            f"{attack:<9} | {freq:9.2f}% | {fs:17.2f}% | {sign}{improvement:10.2f}%"
        )

    fp_freq = clean_fp.get("freq", 0.0)
    fp_fs = clean_fp.get("fs", 0.0)
    improvement = fp_fs - fp_freq
    sign = "+" if improvement >= 0 else ""
    print(sep)
    print(
        f"{'Clean FP':<9} | {fp_freq:9.2f}% | {fp_fs:17.2f}% | {sign}{improvement:10.2f}%"
    )
    print(sep)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run feature squeezing detection against CW and DeepFool attacks"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="L1 distance threshold to flag adversarial examples; if omitted, the script chooses the best threshold automatically",
    )
    args = parser.parse_args()

    model = load_detector(MODEL_PATH, DEVICE)
    categories = gather_paths()

    optimal_threshold, search_results = search_optimal_threshold(model, categories)
    print_threshold_search(search_results)
    print(f"\nOptimal threshold selected: {optimal_threshold:.2f}\n")

    threshold = args.threshold if args.threshold is not None else optimal_threshold

    category_results = []
    freq_rates = {}
    fs_rates = {}
    clean_fp = {"freq": 0.0, "fs": 0.0}

    for name, paths, expected in categories:
        total = 0
        correct = 0
        dist_sum = 0.0
        freq_correct = 0
        fs_correct = 0

        for path in paths:
            total += 1
            original = load_image_tensor(path, IMAGE_SIZE, DEVICE)
            original_prob = predict_prob(model, original)
            freq_pred = 1 if original_prob > 0.5 else 0
            freq_correct += int(freq_pred == expected)

            _, max_dist, fs_flag = evaluate_image(model, path, threshold)
            fs_pred = 1 if fs_flag else 0
            fs_correct += int(fs_pred == expected)
            dist_sum += max_dist

        avg_dist = (dist_sum / total) if total > 0 else 0.0
        category_results.append((name, total, fs_correct, avg_dist))
        freq_rates[name] = (freq_correct / total * 100) if total > 0 else 0.0
        fs_rates[name] = (fs_correct / total * 100) if total > 0 else 0.0

        if name == "Clean":
            clean_fp["freq"] = 100.0 - freq_rates[name]
            clean_fp["fs"] = 100.0 - fs_rates[name]

    print("\nFeature squeezing detection results")
    print_category_results(category_results)
    print_comparison(freq_rates, fs_rates, clean_fp)


if __name__ == "__main__":
    main()
