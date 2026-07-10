"""Train the frequency-domain detector on clean vs adversarial images."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from detector.frequency_model import FrequencyDetector
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
    parser.add_argument("--output", type=Path, default=Path("models/detector_frequency.pt"))
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

    total_samples = len(dataset)
    val_size = int(total_samples * 0.2)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

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
            loss = criterion(model(images), labels)
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
                predictions = (torch.sigmoid(logits) >= 0.5).float()
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
    print(f"Saved weights to {args.output}")

    # Compute confusion matrix on validation set
    print("\n" + "="*60)
    print("VALIDATION SET CONFUSION MATRIX")
    print("="*60)

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            prob = torch.sigmoid(logits)
            predictions = (prob > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = [int(p) for p in all_predictions]
    all_labels = [int(l) for l in all_labels]

    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives (clean correctly identified):     {tn}")
    print(f"False Positives (clean labelled as adversarial): {fp}")
    print(f"False Negatives (adversarial labelled as clean): {fn}")
    print(f"True Positives (adversarial correctly identified): {tp}")
    print()

    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
