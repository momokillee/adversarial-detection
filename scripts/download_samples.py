"""Download or synthesize sample images into data/clean/ for testing."""

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from PIL import Image, ImageDraw

URLS = [
    "https://picsum.photos/id/101/224/224",
    "https://picsum.photos/id/102/224/224",
    "https://picsum.photos/id/103/224/224",
    "https://picsum.photos/id/104/224/224",
    "https://picsum.photos/id/106/224/224",
    "https://picsum.photos/id/107/224/224",
    "https://picsum.photos/id/108/224/224",
    "https://picsum.photos/id/109/224/224",
]


def synthesize(path: Path, index: int) -> None:
    """Create a simple RGB test image when download fails."""
    img = Image.new("RGB", (224, 224), color=(40 + index * 20, 80, 120 + index * 10))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 184, 184], fill=(200, 150, 50))
    draw.rectangle([90, 90, 134, 134], fill=(30, 30, 180))
    img.save(path, format="JPEG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(1, args.count + 1):
        dest = args.output_dir / f"sample_{i:02d}.jpg"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"Skip existing {dest}")
            saved += 1
            continue

        url = URLS[(i - 1) % len(URLS)] if i <= len(URLS) else None
        if url:
            try:
                print(f"Downloading {url} -> {dest}")
                urlretrieve(url, dest)
                saved += 1
                continue
            except (HTTPError, URLError, TimeoutError) as e:
                print(f"Download failed ({e}); synthesizing {dest}")

        synthesize(dest, i)
        print(f"Synthesized {dest}")
        saved += 1

    print(f"Done. {saved} images in {args.output_dir}")


if __name__ == "__main__":
    main()
