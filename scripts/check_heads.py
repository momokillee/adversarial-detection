"""Check HEAD availability for dataset URLs and report HTTP status and content-length.

If HEAD is not supported, fall back to a small ranged GET to fetch headers.
"""
import urllib.request
import urllib.error
from urllib.parse import urlparse

URLS = [
    ("CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"),
    ("STL-10", "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"),
    ("SVHN", "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"),
    ("Tiny ImageNet", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"),
    ("MNIST", "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"),
]


def try_head(url, timeout=30):
    req = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            length = resp.getheader('Content-Length')
            return code, length
    except Exception as e:
        raise


def try_range_get(url, timeout=30):
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            length = resp.getheader('Content-Length')
            return code, length
    except Exception as e:
        raise


if __name__ == '__main__':
    for name, url in URLS:
        try:
            try:
                code, length = try_head(url)
                method = 'HEAD'
            except Exception as e_head:
                # fallback to Range GET
                try:
                    code, length = try_range_get(url)
                    method = 'Range-GET'
                except Exception as e_get:
                    print(f"{name}: FAILED - {e_get}")
                    continue

            if length is None:
                size_mb = 'unknown'
            else:
                try:
                    size_mb = float(length) / (1024 * 1024)
                    size_mb = f"{size_mb:.1f} MB"
                except Exception:
                    size_mb = length
            print(f"{name}: OK - HTTP {code} - Size: {size_mb} (via {method})")
        except Exception as e:
            print(f"{name}: FAILED - {e}")
