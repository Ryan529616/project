#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PY_T210="/home/ryan529/miniconda3/envs/pytorch_cu130_py311_t210/bin/python"
DEFAULT_PY_OLD="/home/ryan529/miniconda3/envs/pytorch_cu130_py311/bin/python"

if [[ -n "${TRAIN_PYTHON:-}" ]]; then
  PYTHON_BIN="$TRAIN_PYTHON"
elif [[ -x "$DEFAULT_PY_T210" ]]; then
  PYTHON_BIN="$DEFAULT_PY_T210"
else
  PYTHON_BIN="$DEFAULT_PY_OLD"
fi

CFG_PATH="${1:-${CFG_PATH:-$ROOT_DIR/configs/cfg_hardware_aware_bdd100k_v1.json}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CFG_PATH" ]]; then
  echo "[error] cfg not found: $CFG_PATH" >&2
  exit 1
fi

echo "[check] python=$PYTHON_BIN"
echo "[check] cfg=$CFG_PATH"

CFG_PATH_ENV="$CFG_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

def _safe_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[check] {name}={ver}")
        return mod
    except Exception as e:
        print(f"[check] {name} import failed: {type(e).__name__}: {e}")
        return None

print(f"[check] python={sys.version.split()[0]}")
torch = _safe_import("torch")
_safe_import("torchvision")
_safe_import("torchaudio")
_safe_import("triton")

cfg_path = os.environ["CFG_PATH_ENV"]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
trainopt = cfg.get("trainopt", {}) if isinstance(cfg, dict) else {}
print(
    f"[check] cfg.compile_detector={bool(trainopt.get('compile_detector', False))} "
    f"cfg.compile_mode={trainopt.get('compile_mode', 'default')}"
)

if torch is None:
    raise SystemExit(3)

ok = bool(torch.cuda.is_available())
print(f"[check] torch.cuda.is_available={ok}")
if ok:
    try:
        print(f"[check] gpu0={torch.cuda.get_device_name(0)}")
        print(f"[check] cuda_build={torch.version.cuda}")
        print(f"[check] bf16={torch.cuda.is_bf16_supported()}")
    except Exception as e:
        print(f"[check] cuda probe warning: {type(e).__name__}: {e}")

torch_mm = ".".join(str(torch.__version__).split("+")[0].split(".")[:2])
if trainopt.get("compile_detector") and torch_mm == "2.9":
    print("[check][warn] compile enabled on torch 2.9.x; consider torch 2.10+ if compile errors occur.")
PY
