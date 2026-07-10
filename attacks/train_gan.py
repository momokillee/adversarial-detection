"""Train a simple GAN-style generator to fool the frequency-domain detector."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from attacks.gan_attack import Generator
from detector.model import AdversarialDetector
from utils.preprocess import load_image_tensor, save_tensor_image

IMAGE_SIZE = 64
EPSILON = 0.15


class CleanImageDataset(Dataset):
    """Dataset of clean images for generator training."""

    def __init__(self, clean_dir: Path, device: torch.device = torch.device("cpu")):
        self.samples = []
        self.device = device
        for p in clean_dir.glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        x = load_image_tensor(path, IMAGE_SIZE, self.device).squeeze(0)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gan_adversarial"))
    parser.add_argument("--model-output", type=Path, default=Path("models/gan_generator.pt"))
    parser.add_argument("--detector-path", type=Path, default=Path("models/detector.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to train on. If None, auto-selects GPU if available.")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    dataset = CleanImageDataset(args.clean_dir, device=device)
    if len(dataset) == 0:
        raise SystemExit("No images found. Add clean images to data/clean/.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator(epsilon=EPSILON).to(device)
    detector = AdversarialDetector().to(device)
    detector.load_state_dict(torch.load(args.detector_path, map_location=device))

    optimizer = torch.optim.Adam(generator.parameters(), lr=5e-3)

    for epoch in range(args.epochs):
        detector_losses = []
        diversity_losses = []
        for images in loader:
            optimizer.zero_grad()

            adv = generator(images)
            detector_logits = detector(adv)
            detector_loss = torch.mean(torch.clamp(detector_logits + 3.0, min=0.0))
            diversity = -torch.mean(torch.abs(adv - images))
            total_loss = detector_loss + 0.5 * diversity
            total_loss.backward()
            optimizer.step()

            detector_losses.append(detector_loss.item())
            diversity_losses.append(diversity.item())

        avg_detector_loss = sum(detector_losses) / len(detector_losses)
        avg_diversity = sum(diversity_losses) / len(diversity_losses)
        print(f"Epoch {epoch + 1}/{args.epochs} detector_loss={avg_detector_loss:.4f} diversity={avg_diversity:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), args.model_output)
    print(f"Saved generator weights to {args.model_output}")

    detector.eval()
    generator.eval()
    with torch.no_grad():
        for idx, images in enumerate(loader):
            adv = generator(images)
            for j, img in enumerate(adv):
                out_path = args.output_dir / f"gan_adv_{idx * args.batch_size + j + 1}.png"
                save_tensor_image(img.unsqueeze(0), out_path)

    print(f"Saved generated adversarial images to {args.output_dir}")


if __name__ == "__main__":
    main()
