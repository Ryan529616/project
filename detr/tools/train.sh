#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${ROOT}/configs/train_detr_rggb.json"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x "/home/ryan529/miniconda3/envs/pytorch_cu130_py311_t210/bin/python" ]]; then
  PY="/home/ryan529/miniconda3/envs/pytorch_cu130_py311_t210/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "[error] python interpreter not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 127
fi

"${PY}" "${ROOT}/train_detr_rggb.py" --cfg "${CFG}" "$@"
