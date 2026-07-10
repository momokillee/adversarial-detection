"""Visualize FFT magnitude spectra for clean and adversarial images."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.preprocess import load_image_tensor


IMAGE_SIZE = 64
OUTPUT_PATH = Path("experiments/fft_comparison.png")


def load_image_tensor_from_path(path: Path, device: torch.device) -> torch.Tensor:
    """Load an image as a tensor with shape (1, 3, 64, 64)."""
    return load_image_tensor(path, IMAGE_SIZE, device)


def compute_fft_spectrum(tensor: torch.Tensor) -> torch.Tensor:
    """Compute log-scaled, shifted FFT magnitude spectrum for display."""
    tensor = tensor.squeeze(0)  # (3, 64, 64)

    fft = torch.fft.fft2(tensor, dim=(-2, -1))
    magnitude = torch.abs(fft)
    log_magnitude = torch.log1p(magnitude)
    shifted = torch.fft.fftshift(log_magnitude)

    averaged = shifted.mean(dim=0)
    return averaged.cpu().detach()


def find_first_matching_image(directory: Path, prefix: str) -> Path:
    """Find the first image in a directory matching a prefix."""
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.name.startswith(prefix):
            return path
    raise FileNotFoundError(f"No image found in {directory} with prefix {prefix}")


def main():
    """Create a comparison figure of clean and adversarial FFT spectra."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare FFT spectra of clean and adversarial images")
    parser.add_argument("--input-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--adversarial-dir", type=Path, default=Path("data/adversarial"))
    parser.add_argument("--bim-dir", type=Path, default=Path("data/bim_adversarial"))
    parser.add_argument("--cw-dir", type=Path, default=Path("data/cw_adversarial"))
    parser.add_argument("--deepfool-dir", type=Path, default=Path("data/deepfool_adversarial"))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    clean_dir = args.input_dir
    adv_dir = args.adversarial_dir
    bim_dir = args.bim_dir
    cw_dir = args.cw_dir
    deepfool_dir = args.deepfool_dir

    clean_image_path = None
    for path in sorted(clean_dir.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            clean_image_path = path
            break

    if clean_image_path is None:
        raise FileNotFoundError(f"No clean image found in {clean_dir}")

    clean_tensor = load_image_tensor_from_path(clean_image_path, device)

    image_paths = [("Clean", clean_tensor)]

    fgsm_path = find_first_matching_image(adv_dir, "fgsm_")
    image_paths.append(("FGSM", load_image_tensor_from_path(fgsm_path, device)))

    pgd_path = find_first_matching_image(adv_dir, "pgd_")
    image_paths.append(("PGD", load_image_tensor_from_path(pgd_path, device)))

    bim_path = find_first_matching_image(bim_dir, "bim_")
    image_paths.append(("BIM", load_image_tensor_from_path(bim_path, device)))

    cw_image = None
    if cw_dir.exists():
        try:
            cw_path = find_first_matching_image(cw_dir, "cw_")
            cw_image = load_image_tensor_from_path(cw_path, device)
        except FileNotFoundError:
            cw_image = None

    if cw_image is not None:
        image_paths.append(("CW", cw_image))

    deepfool_image = None
    if deepfool_dir.exists():
        try:
            deepfool_path = find_first_matching_image(deepfool_dir, "deepfool_")
            deepfool_image = load_image_tensor_from_path(deepfool_path, device)
        except FileNotFoundError:
            deepfool_image = None

    if deepfool_image is not None:
        image_paths.append(("DeepFool", deepfool_image))

    images = []
    spectra = []

    for _, tensor in image_paths:
        images.append(tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        spectra.append(compute_fft_spectrum(tensor))

    fig, axes = plt.subplots(2, len(image_paths), figsize=(3 * len(image_paths), 6))

    for idx, (name, _) in enumerate(image_paths):
        ax_img = axes[0, idx]
        ax_img.imshow(images[idx])
        ax_img.set_title(name)
        ax_img.axis("off")

        ax_spec = axes[1, idx]
        spec = spectra[idx]
        ax_spec.imshow(spec.numpy(), cmap="viridis")
        ax_spec.set_title(f"{name} FFT")
        ax_spec.axis("off")

    fig.suptitle("FFT Magnitude Comparison Across Attack Types", fontsize=14)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved FFT comparison figure to {args.output}")


if __name__ == "__main__":
    main()
