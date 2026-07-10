"""Generate CW, DeepFool, and AutoAttack adversarial examples with torchattacks."""

import sys
from pathlib import Path

import torch
import torchattacks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.victim_model import load_victim
from utils.preprocess import load_image_tensor, save_tensor_image


IMAGE_SIZE = 64
INPUT_DIR = Path("data/clean")


def run_attack_suite(
    model: torch.nn.Module,
    device: torch.device,
    input_dir: Path,
) -> None:
    """Run CW, DeepFool, and AutoAttack over all clean images."""
    output_dirs = {
        "cw": Path("data/cw_adversarial"),
        "deepfool": Path("data/deepfool_adversarial"),
    }

    image_paths = [
        path for path in sorted(input_dir.glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    attacks = {
        "cw": torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01),
        "deepfool": torchattacks.DeepFool(model, steps=50),
    }

    summary = {}

    for attack_name, atk in attacks.items():
        output_dir = output_dirs[attack_name]
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for index, img_path in enumerate(image_paths, start=1):
            image = load_image_tensor(img_path, IMAGE_SIZE, device)
            image = image.to(device).float()

            with torch.no_grad():
                outputs = model(image)
                pred = outputs.argmax(dim=1).item()

            label = torch.tensor([pred], dtype=torch.long, device=device)

            adv_image = atk(image, label)

            out_path = output_dir / f"{attack_name}_{img_path.name}"
            save_tensor_image(adv_image, out_path)

            saved_count += 1

            if index % 50 == 0:
                print(f"[{attack_name}] processed {index} images...")

        summary[attack_name] = saved_count
        print(f"[{attack_name}] completed. Saved {saved_count} images to {output_dir}")

    print("\nFinal summary:")
    for attack_name, count in summary.items():
        print(f"  {attack_name}: {count} images saved")


def main():
    """Load model and run all requested attacks."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate adversarial examples using torchattacks")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model = load_victim(device)
    model.eval()

    run_attack_suite(model, device, args.input_dir)


if __name__ == "__main__":
    main()
