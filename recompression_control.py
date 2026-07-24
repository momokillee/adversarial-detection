"""Control test: re-save clean images through the identical load/save
pipeline used for adversarial images, but with ZERO perturbation applied.
If detectors flag these as adversarial, that indicates they're partly
keying on recompression artifacts rather than true perturbation.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.dual_model import DualDomainDetector
from detector.frequency_model import FrequencyDetector
from utils.preprocess import denormalize_tensor, load_image_tensor, save_tensor_image

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
CONTROL_DIR = Path("data/recompression_control")

DETECTORS = {
    "dual": ("models/detector_dual.pt", DualDomainDetector, "models/detector_dual_val_split.json"),
    "frequency": ("models/detector_frequency.pt", FrequencyDetector, "models/detector_frequency_val_split.json"),
}


def load_detector(path, model_class):
    model = model_class().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    # Use the same clean validation images that were held out during training
    with open("models/detector_dual_val_split.json") as f:
        manifest = json.load(f)
    clean_paths = [Path(m["path"]) for m in manifest if m["label"] == 0]

    print(f"Re-saving {len(clean_paths)} clean validation images through the "
          f"identical pipeline (load -> tensor -> denormalize -> save), no perturbation applied.")

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    control_paths = []
    for src_path in clean_paths:
        image = load_image_tensor(src_path, IMAGE_SIZE, DEVICE, normalize=True)
        # No perturbation -- straight round-trip through the save pipeline
        to_save = denormalize_tensor(image)
        out_path = CONTROL_DIR / src_path.name
        save_tensor_image(to_save, out_path)
        control_paths.append(out_path)

    print(f"Saved {len(control_paths)} control images to {CONTROL_DIR}\n")

    for name, (path, cls, _) in DETECTORS.items():
        model = load_detector(path, cls)
        false_positives = 0
        for p in control_paths:
            x = load_image_tensor(p, IMAGE_SIZE, DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()
            if prob > 0.5:
                false_positives += 1
        rate = false_positives / len(control_paths) * 100
        print(f"{name:10s}: {false_positives}/{len(control_paths)} control images "
              f"flagged as adversarial ({rate:.1f}%)")
        if rate > 5:
            print(f"           WARNING: high false-positive rate on unperturbed, "
                  f"resaved images -- suggests recompression artifact confound")
        else:
            print(f"           Low/no false positives -- detector likely keying on "
                  f"real perturbation, not save artifacts")


if __name__ == "__main__":
    main()