#!/usr/bin/env python3
"""Download a labeled CIFAR-10-like subset from a GitHub-hosted mirror.

The script reads the repository tree from the remote GitHub repo, discovers
images under the standard CIFAR-10 class folders, and saves them into:

    data/clean_labeled/<class_name>/imgXXX.png

It also writes a labels.json file mapping class names to integer labels in
standard CIFAR-10 order.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.error import URLError

from PIL import Image

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
DEFAULT_REPO = "YoongiKim/CIFAR-10-images"
DEFAULT_SUBSET_SIZE = 200


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def get_default_branch(repo: str) -> str:
    data = fetch_json(f"https://api.github.com/repos/{repo}")
    return data.get("default_branch", "master")


def list_repo_files(repo: str, branch: str) -> List[str]:
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    data = fetch_json(tree_url)
    return [entry["path"] for entry in data.get("tree", []) if entry.get("type") == "blob"]


def find_class_files(repo_files: Sequence[str], class_name: str, split: str) -> List[str]:
    target_class = class_name.lower()
    matches: List[str] = []
    for path in repo_files:
        parts = [part.lower() for part in Path(path).parts]
        if split != "all":
            if not parts or parts[0] != split.lower():
                continue
        suffix = Path(path).suffix.lower()
        if suffix in IMAGE_EXTENSIONS and target_class in parts:
            matches.append(path)
    return sorted(matches)


def download_image_bytes(url: str, max_attempts: int = 4) -> bytes:
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            if attempt == max_attempts - 1:
                raise
            wait_time = 2 ** (attempt + 1)
            print(f"Download failed for {url}: {exc}. Retrying in {wait_time}s ({attempt + 1}/{max_attempts})...")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed to download {url}")


def save_png_from_bytes(data: bytes, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(data)) as image:
        if image.mode in {"RGBA", "LA", "P"}:
            image = image.convert("RGB")
        image.save(dest_path, format="PNG")


def populate_split_directories(
    class_name: str,
    selected_files: Sequence[str],
    output_dir: Path,
    branch: str,
    repo: str,
    rng: random.Random,
    overwrite: bool,
    failed_downloads: List[tuple[str, str]],
) -> Dict[str, int]:
    victim_train_dir = output_dir / "victim_train" / class_name
    attack_source_dir = output_dir / "attack_source" / class_name
    victim_train_dir.mkdir(parents=True, exist_ok=True)
    attack_source_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for split_dir in (victim_train_dir, attack_source_dir):
            for existing_file in split_dir.glob("*.png"):
                existing_file.unlink()

    shuffled_files = list(selected_files)
    rng.shuffle(shuffled_files)
    split_index = int(len(shuffled_files) * 0.75)
    train_files = shuffled_files[:split_index]
    attack_files = shuffled_files[split_index:]

    train_counter = len(list(victim_train_dir.glob("*.png")))
    attack_counter = len(list(attack_source_dir.glob("*.png")))

    for rel_path in train_files:
        dest_name = f"img_{train_counter:03d}.png"
        dest_path = victim_train_dir / dest_name
        train_counter += 1

        if not overwrite and dest_path.exists():
            continue

        time.sleep(0.15)
        raw_url = (
            "https://raw.githubusercontent.com/"
            f"{repo}/{branch}/{urllib.parse.quote(rel_path, safe='/')}"
        )
        try:
            image_bytes = download_image_bytes(raw_url)
        except Exception as exc:  # pragma: no cover - defensive fallback
            failed_downloads.append((class_name, Path(rel_path).name))
            print(f"Warning: failed to download {class_name}/{Path(rel_path).name}: {exc}")
            continue
        save_png_from_bytes(image_bytes, dest_path)

    for rel_path in attack_files:
        dest_name = f"img_{attack_counter:03d}.png"
        dest_path = attack_source_dir / dest_name
        attack_counter += 1

        if not overwrite and dest_path.exists():
            continue

        time.sleep(0.15)
        raw_url = (
            "https://raw.githubusercontent.com/"
            f"{repo}/{branch}/{urllib.parse.quote(rel_path, safe='/')}"
        )
        try:
            image_bytes = download_image_bytes(raw_url)
        except Exception as exc:  # pragma: no cover - defensive fallback
            failed_downloads.append((class_name, Path(rel_path).name))
            print(f"Warning: failed to download {class_name}/{Path(rel_path).name}: {exc}")
            continue
        save_png_from_bytes(image_bytes, dest_path)

    victim_count = len(list(victim_train_dir.glob("*.png")))
    attack_count = len(list(attack_source_dir.glob("*.png")))
    return {"victim_train": victim_count, "attack_source": attack_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a subset of labeled CIFAR-10 images")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--output-dir", default="data/clean_labeled")
    parser.add_argument("--per-class", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_ORDER:
        flat_dir = output_dir / class_name
        if flat_dir.exists():
            shutil.rmtree(flat_dir)

    branch = args.branch or get_default_branch(args.repo)
    print(f"Using repository: {args.repo}")
    print(f"Branch: {branch}")
    print(f"Output directory: {output_dir}")
    print(f"Using split: {args.split}")

    repo_files = list_repo_files(args.repo, branch)
    labels: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    split_counts: Dict[str, Dict[str, int]] = {}
    failed_downloads: List[tuple[str, str]] = []
    rng = random.Random(42)

    for idx, class_name in enumerate(CLASS_ORDER):
        labels[class_name] = idx

        candidates = find_class_files(repo_files, class_name, args.split)
        if not candidates:
            print(f"Warning: no image files found for class '{class_name}' in split '{args.split}'")
            counts[class_name] = 0
            split_counts[class_name] = {"victim_train": 0, "attack_source": 0, "written": 0}
            continue

        selected = candidates[: args.per_class]
        split_counts[class_name] = populate_split_directories(
            class_name,
            selected,
            output_dir,
            branch,
            args.repo,
            rng,
            args.overwrite,
            failed_downloads,
        )
        counts[class_name] = split_counts[class_name]["victim_train"] + split_counts[class_name]["attack_source"]

    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as fp:
        json.dump(labels, fp, indent=2, sort_keys=True)

    total_images = sum(counts.values())
    print("\nSummary:")
    print(f"Total images: {total_images}")
    for class_name in CLASS_ORDER:
        per_class = counts[class_name]
        split_info = split_counts.get(class_name, {"victim_train": 0, "attack_source": 0})
        print(
            f" - {class_name}: {per_class} total "
            f"(victim_train={split_info['victim_train']}, attack_source={split_info['attack_source']})"
        )
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Labels written to: {labels_path}")


if __name__ == "__main__":
    main()
