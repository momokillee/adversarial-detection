"""Train the dual-domain detector on clean vs adversarial images."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from detector.dual_model import DualDomainDetector
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
SPLIT_SEED = 42


class CleanAdvDataset(Dataset):
    """Binary dataset: clean (label=0) vs adversarial (label=1) images."""

    def __init__(self, clean_dir: Path, adv_dir: Path, device: torch.device = torch.device("cpu")):
        self.device = device

        self.clean_samples = []
        for p in clean_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.clean_samples.append((p, 0))

        self.adv_samples = []
        for p in adv_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.adv_samples.append((p, 1))

        # clean samples first, then adversarial -- indices below rely on this order
        self.samples = self.clean_samples + self.adv_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_image_tensor(path, IMAGE_SIZE, self.device).squeeze(0)
        return x, torch.tensor(label, dtype=torch.float32).to(self.device)


def stratified_split(dataset: CleanAdvDataset, val_fraction: float = 0.2, seed: int = SPLIT_SEED):
    """Split clean and adversarial samples separately (each val_fraction),
    then combine -- guarantees both classes appear proportionally in both
    train and validation regardless of overall class imbalance."""
    n_clean = len(dataset.clean_samples)
    n_adv = len(dataset.adv_samples)

    clean_indices = list(range(n_clean))
    adv_indices = list(range(n_clean, n_clean + n_adv))

    rng = random.Random(seed)
    rng.shuffle(clean_indices)
    rng.shuffle(adv_indices)

    clean_val_size = int(n_clean * val_fraction)
    adv_val_size = int(n_adv * val_fraction)

    val_indices = clean_indices[:clean_val_size] + adv_indices[:adv_val_size]
    train_indices = clean_indices[clean_val_size:] + adv_indices[adv_val_size:]

    print(f"Found {n_clean} clean images, {n_adv} adversarial images")
    print(f"Train: {len(train_indices)} total "
          f"({n_clean - clean_val_size} clean, {n_adv - adv_val_size} adversarial)")
    print(f"Val:   {len(val_indices)} total "
          f"({clean_val_size} clean, {adv_val_size} adversarial)")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--adv-dir", type=Path, default=Path("data/adversarial"))
    parser.add_argument("--output", type=Path, default=Path("models/detector_dual.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    dataset = CleanAdvDataset(args.clean_dir, args.adv_dir, device=device)
    if len(dataset) == 0:
        raise SystemExit(f"No images found. Check {args.clean_dir} and {args.adv_dir}.")

    train_dataset, val_dataset = stratified_split(dataset)

    # Save the exact validation file paths + labels for later strict evaluation
    val_split_path = args.output.parent / (args.output.stem + "_val_split.json")
    val_manifest = [
        {"path": str(dataset.samples[i][0]), "label": dataset.samples[i][1]}
        for i in val_dataset.indices
    ]
    with open(val_split_path, "w") as f:
        json.dump(val_manifest, f, indent=2)
    print(f"Saved validation split manifest to {val_split_path} ({len(val_manifest)} files)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DualDomainDetector().to(device)
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

    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives (clean correctly identified):     {tn}")
    print(f"False Positives (clean labelled as adversarial): {fp}")
    print(f"False Negatives (adversarial labelled as clean): {fn}")
    print(f"True Positives (adversarial correctly identified): {tp}")
    print()

    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()