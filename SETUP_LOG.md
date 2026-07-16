# Setup log — adversarial-detection

This document records all changes and actions taken when the project was initialized and organized into the target structure.

**Date:** June 4, 2026  
**Starting state:** Empty directory (`adversarial-detection/` had no files)  
**Goal:** Match the requested folder layout with starter code for attacks, detection, training, and a Gradio demo.

---

## Actions performed

1. **Inspected the workspace** — Confirmed the repo folder was empty (no existing files to move or rename).
2. **Created the directory tree** — All folders from the target structure, plus placeholder files where needed.
3. **Added configuration files** — `.gitignore`, `.env`, `README.md`, `requirements.txt`.
4. **Scaffolded Python modules** — Attack scripts (FGSM, PGD), detector model + training, Gradio app.
5. **Updated README** — Removed a mistaken `cp .env .env` line; replaced with a note to edit `.env` directly.

No files were moved from elsewhere (there was nothing to migrate). No git commits were made.

---

## Target structure (achieved)

```
adversarial-detection/
│
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

Additional files added beyond the minimal tree (for usability):

- `requirements.txt` — Python dependencies
- `SETUP_LOG.md` — This file

---

## Files created (full list)

| Path | Purpose |
|------|---------|
| `.gitignore` | Ignores `.env`, Python cache, venvs, Jupyter checkpoints, IDE files |
| `.env` | Template for API keys (commented placeholders; not committed) |
| `README.md` | Project overview, structure, setup, and usage commands |
| `requirements.txt` | `torch`, `torchvision`, `numpy`, `Pillow`, `gradio`, `jupyter` |
| `data/clean/.gitkeep` | Keeps empty `clean/` folder in git |
| `data/adversarial/.gitkeep` | Keeps empty `adversarial/` folder in git |
| `models/.gitkeep` | Keeps empty `models/` folder in git |
| `notebooks/.gitkeep` | Placeholder for future Jupyter notebooks |
| `attacks/__init__.py` | Package marker for `attacks` |
| `attacks/fgsm.py` | FGSM attack CLI → writes to `data/adversarial/` |
| `attacks/pgd.py` | PGD attack CLI → writes to `data/adversarial/` |
| `detector/__init__.py` | Exports `AdversarialDetector` |
| `detector/model.py` | Small CNN binary classifier (clean vs adversarial) |
| `detector/train.py` | Training script; saves `models/detector.pt` |
| `app/demo.py` | Gradio UI for image upload + prediction |

---

## File-by-file summary

### `.gitignore`

- Excludes `.env` so secrets are never committed.
- Standard Python, Jupyter, and IDE ignore patterns.
- Commented optional rules for large `models/` and `data/` artifacts.

### `.env`

- Comment-only template (`OPENAI_API_KEY`, `HUGGINGFACE_TOKEN` placeholders).
- User fills in values locally; file stays out of version control.

### `README.md`

- Describes project purpose and folder layout.
- Setup: venv, `pip install -r requirements.txt`.
- Usage flow: images → attacks → train detector → run demo.

### `attacks/fgsm.py`

- Loads ResNet18 (ImageNet weights).
- Reads images from `data/clean/` (default).
- Applies FGSM with configurable `--epsilon` (default `0.03`).
- Saves adversarial images as `fgsm_<original_name>` in `data/adversarial/`.

### `attacks/pgd.py`

- Same model and input/output paths as FGSM.
- Iterative PGD with `--epsilon`, `--alpha`, `--steps` (defaults: 0.03, 0.01, 10).
- Saves as `pgd_<original_name>` in `data/adversarial/`.

### `detector/model.py`

- `AdversarialDetector`: 3 conv blocks + linear head, single logit output.
- `predict_proba()` applies sigmoid for inference.

### `detector/train.py`

- `CleanAdvDataset`: label `0` for `data/clean/`, `1` for `data/adversarial/`.
- Resizes to 64×64, BCE loss, Adam optimizer.
- Default output: `models/detector.pt`.
- Exits with error if no images are found.

### `app/demo.py`

- Loads `models/detector.pt` if present (otherwise untrained weights).
- Gradio image upload → "Clean" or "Adversarial" with confidence.

---

## Suggested workflow (not run automatically)

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add test images
#    Place .jpg / .png files in data/clean/

# 3. Generate attacks
python attacks/fgsm.py
python attacks/pgd.py

# 4. Train detector
python detector/train.py

# 5. Run demo
python app/demo.py
```

---

## What was not done

- No git repository initialized (unless you do that separately).
- No sample images downloaded or committed.
- No pretrained `models/detector.pt` (training required after data exists).
- No Jupyter notebooks created in `notebooks/` (folder only).
- No CI, tests, or Docker configuration.

---

## Edits after initial creation

| File | Change |
|------|--------|
| `README.md` | Replaced incorrect `cp .env .env` with note to edit `.env` |
| `SETUP_LOG.md` | Added (this document) per user request |

---

## Next steps you might take

1. Add images to `data/clean/` and run attack scripts.
2. Train the detector and verify `models/detector.pt` loads in the app.
3. Add experiment notebooks under `notebooks/`.
4. Initialize git: `git init`, first commit (`.env` will stay ignored).
5. Tune model architecture, hyperparameters, or attack parameters as needed.

---

## Phase 2 — Pipeline run (June 5, 2026)

### Actions performed

1. **Downloaded 8 sample images** into `data/clean/` via `scripts/download_samples.py` (Picsum + PIL fallback).
2. **Created `.venv`** (partial `pip install` timed out; global pyenv Python used for runs).
3. **Removed `torchvision` dependency** — pyenv build lacks `_lzma`, which broke `torchvision` imports. Added `utils/preprocess.py` (PIL-only) and `attacks/victim_model.py` (small victim CNN).
4. **Ran FGSM + PGD** — 16 adversarial images in `data/adversarial/` (8 FGSM + 8 PGD).
5. **Trained detector** — 20 epochs → `models/detector.pt` (~377 KB).
6. **Added automation** — `scripts/run_pipeline.sh`, `notebooks/01_pipeline_overview.ipynb`.
7. **Pinned `numpy<2`** in `requirements.txt` to avoid torch/numpy ABI warnings.

### Commands used

```bash
cd /Users/albin/Documents/adversarial-detection
export PYTHONPATH=.
python3 scripts/download_samples.py
python3 attacks/fgsm.py
python3 attacks/pgd.py
python3 detector/train.py --epochs 20
# Demo (manual): python3 app/demo.py
```

Or: `bash scripts/run_pipeline.sh`

### Current data state

| Location | Count |
|----------|-------|
| `data/clean/` | 8 images (`sample_01.jpg` … `sample_08.jpg`) |
| `data/adversarial/` | 16 images (`fgsm_*`, `pgd_*`) |
| `models/detector.pt` | Trained weights (present) |

### New / updated files

| Path | Change |
|------|--------|
| `scripts/download_samples.py` | Download + synthesize fallback images |
| `scripts/run_pipeline.sh` | One-shot pipeline script |
| `utils/preprocess.py` | PIL → tensor without torchvision |
| `attacks/victim_model.py` | Victim CNN for attacks |
| `attacks/fgsm.py`, `attacks/pgd.py` | Use victim model + 64×64 tensors |
| `detector/train.py`, `app/demo.py` | No torchvision |
| `notebooks/01_pipeline_overview.ipynb` | Visualize + score samples |
| `requirements.txt` | Dropped torchvision; `numpy<2`; added matplotlib |

### Environment notes for your machine

- **`.venv` vs `.env`**: `.venv/` is the Python virtualenv; `.env` is only API keys (a file, not a folder). Activate with `source .venv/bin/activate` after `python -m venv .venv && pip install -r requirements.txt`.
- **NumPy warning**: Run `pip install "numpy>=1.24,<2"` in your active environment to silence torch/numpy ABI warnings.
- **pyenv `_lzma`**: If you reinstall Python via pyenv with `xz` installed, `torchvision` can be re-added later; the project no longer requires it.
