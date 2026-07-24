"""Diagnose the pixel detector's collapse -- check raw probability outputs."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector.model import AdversarialDetector
from utils.preprocess import load_image_tensor

DEVICE = torch.device("cpu")
model = AdversarialDetector().to(DEVICE)
model.load_state_dict(torch.load("models/detector.pt", map_location=DEVICE))
model.eval()

clean_paths = list(Path("data/clean_labeled/attack_source").rglob("*"))[:5]
adv_paths = list(Path("data/adversarial_all_v2/fgsm").rglob("*"))
adv_paths = [p for p in adv_paths if p.is_file()][:5]

print("Clean samples (expect low prob):")
for p in clean_paths:
    if not p.is_file(): continue
    x = load_image_tensor(p, 64, DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    print(f"  {p.name}: prob={prob:.4f}")

print("FGSM adversarial samples (expect high prob):")
for p in adv_paths:
    x = load_image_tensor(p, 64, DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    print(f"  {p.name}: prob={prob:.4f}")