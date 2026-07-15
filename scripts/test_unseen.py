#!/usr/bin/env python3
"""Evaluate the trained detector on completely unseen images.

This script loads images from data/unseen_test/, preprocesses them using the
same PIL-based transform as the rest of the project, runs them through the
saved detector model, generates FGSM adversarial variants, and prints results.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from attacks.fgsm import fgsm_attack
from attacks.victim_model import load_victim
from detector.model import AdversarialDetector
from utils.preprocess import load_image_tensor, save_tensor_image

MODEL_PATH = Path("models/detector.pt")
IMAGE_SIZE = 64
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_detector(device: torch.device) -> AdversarialDetector:
    model = AdversarialDetector().to(device)
    if MODEL_PATH.exists():
        try:
            state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    else:
        print(f"Warning: model file not found at {MODEL_PATH}. Using uninitialized detector.")
    model.eval()
    return model


def classify_detector(model: AdversarialDetector, image_tensor: torch.Tensor) -> tuple[str, float, float]:
    with torch.no_grad():
        prob = torch.sigmoid(model(image_tensor)).item()
    label = "Adversarial" if prob > 0.5 else "Clean"
    confidence = prob if label == "Adversarial" else 1.0 - prob
    return label, confidence, prob


def create_fgsm_adversarial(
    image_tensor: torch.Tensor,
    victim_model: torch.nn.Module,
    epsilon: float,
) -> torch.Tensor:
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    output = victim_model(image_tensor)
    pred = output.argmax(dim=1)
    loss = F.cross_entropy(output, pred)
    victim_model.zero_grad()
    loss.backward()
    return fgsm_attack(image_tensor, epsilon, image_tensor.grad).detach()


def gather_image_paths(source_dir: Path) -> list[Path]:
    return sorted(
        [path for path in source_dir.glob("*") if path.suffix.lower() in VALID_EXTENSIONS]
    )


def print_results_table(rows: list[tuple[str, str, str, str]]) -> None:
    print(f"{'Image Name':<36} {'Type':<14} {'Predicted Label':<14} {'Confidence':>10}")
    print("-" * 76)
    for image_name, image_type, predicted_label, confidence in rows:
        print(f"{image_name:<36} {image_type:<14} {predicted_label:<14} {confidence:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the adversarial detector on unseen images.")
    parser.add_argument(
        "--unseen-dir",
        type=Path,
        default=Path("data/unseen_test"),
        help="Directory containing unseen images to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/unseen_test_adversarial"),
        help="Directory to save generated adversarial images.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.08,
        help="FGSM epsilon for adversarial example generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use. If None, auto-selects cuda when available.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    unseen_dir = args.unseen_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = gather_image_paths(unseen_dir)
    if not image_paths:
        print(f"No images found in {unseen_dir}. Supported extensions: {', '.join(sorted(VALID_EXTENSIONS))}")
        raise SystemExit(1)

    detector = load_detector(device)
    victim_model = load_victim(device)
    victim_model.eval()

    rows: list[tuple[str, str, str, str]] = []
    clean_correct = 0
    adv_correct = 0
    total_correct = 0

    for image_path in image_paths:
        clean_tensor = load_image_tensor(image_path, IMAGE_SIZE, device)
        clean_label, clean_confidence, _ = classify_detector(detector, clean_tensor)
        clean_predicted = clean_label == "Clean"
        clean_correct += int(clean_predicted)
        total_correct += int(clean_predicted)
        rows.append((image_path.name, "Clean", clean_label, f"{clean_confidence:.2%}"))

        adversarial_tensor = create_fgsm_adversarial(clean_tensor, victim_model, args.epsilon)
        adversarial_name = f"adv_{image_path.name}"
        adversarial_path = output_dir / adversarial_name
        save_tensor_image(adversarial_tensor, adversarial_path)

        adv_label, adv_confidence, _ = classify_detector(detector, adversarial_tensor)
        adv_predicted = adv_label == "Adversarial"
        adv_correct += int(adv_predicted)
        total_correct += int(adv_predicted)
        rows.append((adversarial_name, "Adversarial", adv_label, f"{adv_confidence:.2%}"))

    print()
    print_results_table(rows)
    print()

    num_clean = len(image_paths)
    num_adv = len(image_paths)
    overall = num_clean + num_adv
    clean_accuracy = clean_correct / num_clean if num_clean else 0.0
    adv_accuracy = adv_correct / num_adv if num_adv else 0.0
    overall_accuracy = total_correct / overall if overall else 0.0

    print("Summary")
    print("-------")
    print(f"Clean unseen accuracy:        {clean_accuracy:.2%} ({clean_correct}/{num_clean})")
    print(f"Adversarial unseen accuracy:  {adv_accuracy:.2%} ({adv_correct}/{num_adv})")
    print(f"Overall accuracy:             {overall_accuracy:.2%} ({total_correct}/{overall})")
    print(f"Generated adversarial images saved to: {output_dir}")


if __name__ == "__main__":
    main()
