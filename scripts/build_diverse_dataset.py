"""Build a 50k diverse clean image dataset using Tiny ImageNet, STL-10, and SVHN."""

import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image
import torch
from torchvision.datasets import STL10, SVHN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = Path("data/diverse_clean_224")
RAW_DIR = Path("data/raw")
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_ZIP = RAW_DIR / "tiny-imagenet-200.zip"
IMAGE_SIZE = 224


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_pil_image(image: Image.Image, path: Path) -> bool:
    try:
        img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, format="PNG")
        return True
    except Exception as exc:
        print(f"Failed saving {path}: {exc}")
        return False


def download_tiny_imagenet(timeout: int = 300) -> int:
    if not TINY_IMAGENET_ZIP.exists():
        print(f"Downloading Tiny ImageNet to {TINY_IMAGENET_ZIP} ...")
        response = requests.get(TINY_IMAGENET_URL, stream=True, timeout=timeout)
        response.raise_for_status()
        with TINY_IMAGENET_ZIP.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    saved = 0
    with zipfile.ZipFile(TINY_IMAGENET_ZIP, "r") as archive:
        image_names = [
            name
            for name in archive.namelist()
            if name.startswith("tiny-imagenet-200/train/") and name.endswith(".JPEG")
        ]
        image_names = sorted(image_names)
        for entry in image_names[:20000]:
            try:
                with archive.open(entry) as fp:
                    image = Image.open(fp).convert("RGB")
                    out_path = OUTPUT_DIR / f"imagenet_{saved + 1:05d}.png"
                    if save_pil_image(image, out_path):
                        saved += 1
                        if saved % 1000 == 0:
                            print(f"Tiny ImageNet: saved {saved}/20000")
            except Exception as exc:
                print(f"Skipped Tiny ImageNet entry {entry}: {exc}")
                continue
    return saved


def build_stl10() -> int:
    saved = 0
    stl10 = STL10(root=str(RAW_DIR), split="unlabeled", download=True)
    for idx in range(min(20000, len(stl10))):
        try:
            image = stl10[idx][0]
            out_path = OUTPUT_DIR / f"stl10_{idx + 1:05d}.png"
            if save_pil_image(image, out_path):
                saved += 1
                if saved % 1000 == 0:
                    print(f"STL-10: saved {saved}/20000")
        except Exception as exc:
            print(f"Skipped STL-10 sample {idx + 1}: {exc}")
            continue
    return saved


def build_svhn() -> int:
    saved = 0
    svhn = SVHN(root=str(RAW_DIR), split="train", download=True)
    for idx in range(min(10000, len(svhn))):
        try:
            image = svhn[idx][0]
            out_path = OUTPUT_DIR / f"svhn_{idx + 1:05d}.png"
            if save_pil_image(image, out_path):
                saved += 1
                if saved % 1000 == 0:
                    print(f"SVHN: saved {saved}/10000")
        except Exception as exc:
            print(f"Skipped SVHN sample {idx + 1}: {exc}")
            continue
    return saved


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")
    ensure_dirs()
    start_time = time.time()

    results = {}
    print("Starting Tiny ImageNet build...")
    results["imagenet"] = download_tiny_imagenet(timeout=300)

    print("Starting STL-10 build...")
    results["stl10"] = build_stl10()

    print("Starting SVHN build...")
    results["svhn"] = build_svhn()

    total = sum(results.values())
    elapsed = time.time() - start_time

    print("\nDataset build summary:")
    for source, count in results.items():
        print(f"  {source}: {count}")
    print(f"  total: {total}")
    print(f"  elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
