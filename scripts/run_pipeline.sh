#!/usr/bin/env bash
# Run the full adversarial-detection pipeline from the project root.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

echo "==> Environment check"
"$PYTHON" scripts/check_numpy.py || {
  echo "Fix: source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
}

echo "==> Sample images"
"$PYTHON" scripts/download_samples.py

echo "==> FGSM attacks"
"$PYTHON" attacks/fgsm.py

echo "==> PGD attacks"
"$PYTHON" attacks/pgd.py

echo "==> Train detector"
"$PYTHON" detector/train.py --epochs 20

echo "==> Done. Weights: models/detector.pt"
echo "    Launch demo: python3 app/demo.py"
