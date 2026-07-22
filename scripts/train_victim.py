#!/usr/bin/env python3
"""Train a simple multi-class victim classifier on the labeled CIFAR-10-like dataset."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from attacks.victim_model import VictimCNN
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
CLASS_ORDER = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class LabeledVictimDataset(Dataset):
    def __init__(self, root_dir: Path, labels: Dict[str, int], device: torch.device):
        self.samples: List[Tuple[Path, int]] = []
        self.device = device
        for class_name in CLASS_ORDER:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                continue
            class_label = labels[class_name]
            for image_path in sorted(class_dir.glob("*")):
                if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    self.samples.append((image_path, class_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        x = load_image_tensor(path, IMAGE_SIZE, self.device).squeeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def split_indices(num_items: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    indices = list(range(num_items))
    rng.shuffle(indices)
    split_idx = int(num_items * (1.0 - val_fraction))
    return indices[:split_idx], indices[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VictimCNN on the labeled CIFAR-10-like dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/clean_labeled/victim_train"))
    parser.add_argument("--labels", type=Path, default=Path("data/clean_labeled/labels.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/victim.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true", help="Run a short 2-epoch sanity check")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not args.labels.exists():
        raise SystemExit(f"Labels file not found: {args.labels}")
    labels = json.loads(args.labels.read_text())

    dataset = LabeledVictimDataset(args.data_dir, labels, device=device)
    if len(dataset) == 0:
        raise SystemExit(f"No images found in {args.data_dir}")

    train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VictimCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    epochs = 2 if args.quick else args.epochs
    best_val_acc = 0.0
    best_state_dict = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = correct / total if total else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if not args.quick:
            scheduler.step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if best_state_dict is None:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, args.output)
    print(f"Saved weights to {args.output}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
