"""Test connectivity / download availability for torchvision datasets and Tiny ImageNet URL.

Prints status for CIFAR-10, STL-10, SVHN, and Tiny ImageNet HEAD.
Cleans up the temporary download folder after the test.
"""
from pathlib import Path
import shutil
import sys
import time

TEST_DIR = Path("data/connection_test")
TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

print("Starting connection tests...")

# ensure clean test dir
if TEST_DIR.exists():
    try:
        shutil.rmtree(TEST_DIR)
    except Exception as e:
        print(f"Failed to remove existing {TEST_DIR}: {e}")
        sys.exit(1)

TEST_DIR.mkdir(parents=True, exist_ok=True)

# Helper to run a dataset download test
def test_torchvision_dataset(name, ctor, *args, **kwargs):
    try:
        print(f"Testing {name}...", end=" ")
        start = time.time()
        ds = ctor(*args, **kwargs)
        elapsed = time.time() - start
        try:
            count = len(ds)
        except Exception:
            # some datasets (SVHN) may not implement __len__ before download; check files instead
            count = None
        if count is not None:
            print(f"OK ({count} samples) [{elapsed:.1f}s]")
        else:
            print(f"OK (downloaded) [{elapsed:.1f}s]")
    except Exception as e:
        print(f"FAILED ({e})")

# Run CIFAR-10
try:
    from torchvision.datasets import CIFAR10
    test_torchvision_dataset("CIFAR-10", CIFAR10, str(TEST_DIR / "cifar10"), train=True, download=True)
except Exception as e:
    print(f"CIFAR-10: FAILED (import or download error: {e})")

# Run STL-10 (unlabeled)
try:
    from torchvision.datasets import STL10
    test_torchvision_dataset("STL-10 (unlabeled)", STL10, str(TEST_DIR / "stl10"), split="unlabeled", download=True)
except Exception as e:
    print(f"STL-10: FAILED (import or download error: {e})")

# Run SVHN (train)
try:
    from torchvision.datasets import SVHN
    test_torchvision_dataset("SVHN", SVHN, str(TEST_DIR / "svhn"), split="train", download=True)
except Exception as e:
    print(f"SVHN: FAILED (import or download error: {e})")

# Tiny ImageNet HEAD request
try:
    print("Testing Tiny ImageNet URL (HEAD)...", end=" ")
    # use urllib to avoid external deps
    import urllib.request
    req = urllib.request.Request(TINY_URL, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as resp:
        code = resp.getcode()
        length = resp.getheader('Content-Length') or 'unknown'
        print(f"OK (status={code}, content-length={length})")
except Exception as e:
    print(f"Tiny ImageNet: FAILED ({e})")

# Cleanup
try:
    shutil.rmtree(TEST_DIR)
    print(f"Cleaned up {TEST_DIR}")
except Exception as e:
    print(f"Cleanup failed: {e}")

print("Connection tests complete.")
