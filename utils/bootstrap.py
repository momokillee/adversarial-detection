"""Re-exec with project .venv when global Python has incompatible packages."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def ensure_project_venv() -> None:
    """Use .venv/bin/python if it exists and we're not already running it."""
    if not VENV_PYTHON.is_file():
        return
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])
