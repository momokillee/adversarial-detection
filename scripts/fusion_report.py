"""Generate a compact report comparing the baseline detectors with the calibrated fusion detector."""

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.frequency_model import FrequencyDetector
from scripts.mahalanobis_detector import FeatureExtractor, fit_mahalanobis, load_images_from_folder, mahalanobis_score
from scripts.calibrated_fusion_detector import (
    compute_features,
    load_frequency_detector,
    load_resnet18,
    build_mahalanobis_model,
    split_paths,
    gather_categories,
)
from utils.preprocess import load_image_tensor

IMAGE_SIZE = 64
DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    categories = gather_categories()
    category_map = {name: paths for name, paths, _ in categories}
    freq_model = load_frequency_detector(ROOT / "models" / "detector_frequency.pt", DEVICE)
    resnet = load_resnet18(DEVICE)
    extractor = FeatureExtractor(resnet)

    clean_paths = category_map["Clean"]
    train_clean_paths, _ = split_paths(clean_paths, split_ratio=0.2, seed=7)
    class_means, precision = build_mahalanobis_model(train_clean_paths, resnet, extractor, DEVICE)

    clean_train_maha_scores = []
    with torch.no_grad():
        for path in train_clean_paths:
            img = load_image_tensor(path, IMAGE_SIZE, DEVICE).to(DEVICE)
            _ = resnet(img)
            feat = extractor.get().squeeze(0)
            clean_train_maha_scores.append(float(mahalanobis_score(feat, class_means, precision)))
    maha_mean = float(np.mean(clean_train_maha_scores))
    maha_std = float(np.std(clean_train_maha_scores)) + 1e-8

    for name, paths, expected in categories:
        feats = compute_features(paths, freq_model, resnet, extractor, class_means, precision, maha_mean, maha_std, DEVICE)
        print(name, len(paths), expected)


if __name__ == "__main__":
    main()
