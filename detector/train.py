"""Train the adversarial detector on clean vs adversarial images."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from detector.model import AdversarialDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64


class CleanAdvDataset(Dataset):
    """Binary dataset: clean (label=0) vs adversarial (label=1) images."""
    
    def __init__(self, clean_dir: Path, adv_dir: Path, device: torch.device = torch.device("cpu")):
        """
        Initialize dataset.
        
        Args:
            clean_dir: Path to clean images
            adv_dir: Path to adversarial images
            device: Device to load tensors on (cpu or cuda)
        """
        self.samples = []
        self.device = device
        
        # Load clean images (label=0)
        for p in clean_dir.glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.samples.append((p, 0))
        
        # Load adversarial images (label=1)
        for p in adv_dir.glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.samples.append((p, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load image tensor on specified device
        x = load_image_tensor(path, IMAGE_SIZE, self.device).squeeze(0)
        return x, torch.tensor(label, dtype=torch.float32).to(self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--adv-dir", type=Path, default=Path("data/adversarial"))
    parser.add_argument("--output", type=Path, default=Path("models/detector.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to train on. If None, auto-selects GPU if available.")
    args = parser.parse_args()

    # Auto-select device or use user choice
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataset on the selected device
    dataset = CleanAdvDataset(args.clean_dir, args.adv_dir, device=device)
    if len(dataset) == 0:
        raise SystemExit("No images found. Add samples to data/clean/ and data/adversarial/.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = AdversarialDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, labels in loader:
            # Images and labels are already on the correct device
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={total_loss / len(loader):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved weights to {args.output}")


if __name__ == "__main__":
    main()
