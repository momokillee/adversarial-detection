"""Runtime environment check for NumPy / PyTorch compatibility."""

import json
import sys
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / ".cursor" / "debug-01f780.log"
SESSION_ID = "01f780"


def _log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    # #region agent log
    payload = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


def main() -> int:
    _log("B", "check_numpy.py:main", "python interpreter", {
        "executable": sys.executable,
        "version": sys.version,
    })

    # Hypothesis A: numpy 2.x installed globally
    try:
        import numpy as np
        np_ver = np.__version__
        np_major = int(np_ver.split(".")[0])
        np_ok = np_major < 2
        _log("A", "check_numpy.py:numpy", "numpy import ok", {
            "version": np_ver,
            "path": np.__file__,
            "major": np_major,
            "compatible_with_torch_1x_abi": np_ok,
        })
    except Exception as e:
        _log("C", "check_numpy.py:numpy", "numpy import failed", {"error": str(e)})
        np_ok = False
        np_ver = None

    # Hypothesis A/D: torch numpy bridge
    torch_ok = False
    torch_ver = None
    array_api_ok = False
    try:
        import torch
        torch_ver = torch.__version__
        _log("A", "check_numpy.py:torch", "torch import ok", {"version": torch_ver})

        t = torch.tensor([1.0, 2.0, 3.0])
        try:
            arr = t.numpy()
            array_api_ok = True
            _log("D", "check_numpy.py:tensor_numpy", "torch tensor -> numpy ok", {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            })
        except Exception as e:
            _log("D", "check_numpy.py:tensor_numpy", "torch tensor -> numpy FAILED", {
                "error": str(e),
            })
        torch_ok = True
    except Exception as e:
        _log("A", "check_numpy.py:torch", "torch import failed", {"error": str(e)})

    # Hypothesis B: requirements pin not applied
    req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    req_pin = None
    if req_path.exists():
        for line in req_path.read_text().splitlines():
            if line.strip().lower().startswith("numpy"):
                req_pin = line.strip()
                break
    _log("B", "check_numpy.py:requirements", "requirements numpy pin", {
        "pin": req_pin,
        "installed_version": np_ver,
        "pin_satisfied": np_ok if np_ver else False,
    })

    # Hypothesis C: venv incomplete vs global pyenv
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    venv_exists = venv_python.exists()
    using_venv = str(venv_python.resolve()) == str(Path(sys.executable).resolve()) if venv_exists else False
    _log("C", "check_numpy.py:venv", "venv state", {
        "venv_python_exists": venv_exists,
        "using_venv": using_venv,
        "active_executable": sys.executable,
    })

    # Summary
    issues = []
    if np_ver and not np_ok:
        issues.append(f"numpy {np_ver} is 2.x but torch 2.2.x needs numpy 1.x ABI")
    if torch_ok and not array_api_ok:
        issues.append("torch._ARRAY_API bridge broken (tensor.numpy() fails)")
    if not using_venv and venv_exists:
        issues.append("project .venv exists but is not activated")
    if venv_exists:
        try:
            import importlib.util
            spec = importlib.util.find_spec("numpy")
            if spec is None:
                issues.append(".venv has no numpy installed")
        except Exception:
            pass

    _log("E", "check_numpy.py:summary", "check complete", {
        "issues": issues,
        "healthy": len(issues) == 0,
        "numpy_version": np_ver,
        "torch_version": torch_ver,
        "array_api_ok": array_api_ok,
    })

    print("=== NumPy / PyTorch environment check ===")
    print(f"Python:  {sys.executable}")
    print(f"NumPy:   {np_ver or 'NOT INSTALLED'}")
    print(f"PyTorch: {torch_ver or 'NOT INSTALLED'}")
    print(f"tensor.numpy(): {'OK' if array_api_ok else 'BROKEN'}")
    print(f"requirements pin: {req_pin}")
    if issues:
        print("\nISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nFix: activate .venv and run: pip install -r requirements.txt")
        print("  Or downgrade global numpy: pip install 'numpy>=1.24,<2'")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
