# Adversarial Detection

Detect adversarial image attacks (FGSM, PGD, etc.) on vision models.

## Project structure

```
adversarial-detection/
├── data/
│   ├── clean/          # original test images
│   └── adversarial/    # generated attack samples
├── models/             # saved weights
├── attacks/            # FGSM, PGD attack scripts
├── detector/           # detection model code
├── notebooks/          # Jupyter experiments
├── app/                # Gradio demo UI
├── .env                # API keys (never commit)
├── .gitignore
└── README.md
```

## Setup

```bash
cd adversarial-detection
python3 -m venv .venv          # virtualenv folder — NOT the .env secrets file
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_numpy.py  # must show numpy 1.x + tensor.numpy(): OK
# Optional: edit .env for API keys (separate from .venv/)
```

**NumPy note:** PyTorch 2.2.x needs `numpy<2`. If you see ABI warnings, you're using global pyenv Python (numpy 2.x) instead of `.venv`. Always activate `.venv` first.

## Quick start (full pipeline)

```bash
export PYTHONPATH=.
python scripts/download_samples.py
python attacks/fgsm.py
python attacks/pgd.py
python detector/train.py --epochs 15
python app/demo.py
```

## Usage

1. Place test images in `data/clean/` (or run `scripts/download_samples.py`).
2. Generate attacks: `python attacks/fgsm.py` or `python attacks/pgd.py`.
3. Train or load a detector in `detector/`, save weights to `models/`.
4. Run the demo: `python app/demo.py`.

## Notebooks

Experiments and visualizations live in `notebooks/`.
