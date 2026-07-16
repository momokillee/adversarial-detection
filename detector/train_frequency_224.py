"""Train the frequency-domain detector on 224x224 clean and adversarial data."""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 224


class ImageLabelDataset(Dataset):
    """Simple dataset that loads images from disk on demand."""

    def __init__(self, paths, label, device):
        self.samples = [(path, label) for path in paths]
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_image_tensor(path, IMAGE_SIZE, self.device).squeeze(0)
        return x, torch.tensor(label, dtype=torch.float32).to(self.device)


def collect_files(root: Path, prefix: str = None, limit: int = None):
    files = sorted(root.glob("*"))
    files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if prefix is not None:
        files = [p for p in files if p.name.startswith(prefix)]

    if limit is not None:
        files = files[:limit]

    return files


def main():
    parser = argparse.ArgumentParser(description="Train the 224x224 frequency detector")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/diverse_clean_224"))
    parser.add_argument("--adv-dir", type=Path, default=Path("data/diverse_adversarial_224"))
    parser.add_argument("--output", type=Path, default=Path("models/detector_frequency_224.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    clean_files = collect_files(args.clean_dir, limit=500)
    fgsm_files = collect_files(args.adv_dir, prefix="fgsm_", limit=500)
    pgd_files = collect_files(args.adv_dir, prefix="pgd_", limit=500)
    adv_files = fgsm_files + pgd_files

    clean_dataset = ImageLabelDataset(clean_files, 0, device)
    adv_dataset = ImageLabelDataset(adv_files, 1, device)
    dataset = ConcatDataset([clean_dataset, adv_dataset])

    if len(dataset) == 0:
        raise SystemExit("No image samples found for training.")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = FrequencyDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_accuracy={val_accuracy:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
