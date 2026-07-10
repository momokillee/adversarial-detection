import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from detector.frequency_model import FrequencyDetector
from utils.preprocess import load_image_tensor


device = torch.device("cpu")
IMAGE_SIZE = 64

# Load detector
model = FrequencyDetector().to(device)
model.load_state_dict(torch.load(REPO_ROOT / "models" / "detector_frequency.pt", map_location=device))
model.eval()

# Test on 5 clean images
clean_dir = REPO_ROOT / "data" / "clean"
clean_paths = list(clean_dir.glob("*.png"))[:5] + list(clean_dir.glob("*.jpg"))[:5]
print("=== CLEAN IMAGES ===")
for p in clean_paths[:5]:
    img = load_image_tensor(p, IMAGE_SIZE, device)
    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()
    print(f"{p.name}: prob={prob:.6f} -> {'Adversarial' if prob > 0.5 else 'Clean'}")

# Test on 5 adversarial images
adv_dir = REPO_ROOT / "data" / "adversarial"
adv_paths = list(adv_dir.glob("*.png"))[:5] + list(adv_dir.glob("*.jpg"))[:5]
print("\n=== ADVERSARIAL IMAGES ===")
for p in adv_paths[:5]:
    img = load_image_tensor(p, IMAGE_SIZE, device)
    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()
    print(f"{p.name}: prob={prob:.6f} -> {'Adversarial' if prob > 0.5 else 'Clean'}")

# Check label distribution in dataset
clean_count = len(list(clean_dir.glob("*.png")) + list(clean_dir.glob("*.jpg")))
adv_count = len(list(adv_dir.glob("*.png")) + list(adv_dir.glob("*.jpg")))
print(f"\n=== DATASET COUNTS ===")
print(f"Clean images: {clean_count}")
print(f"Adversarial images: {adv_count}")
print(f"Class ratio: {adv_count/clean_count:.2f}x more adversarial")
