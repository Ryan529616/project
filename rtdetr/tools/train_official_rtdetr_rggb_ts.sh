#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${ROOT}/runs"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${RUN_TAG:-official_r50_6x_coco}"
RUN_DIR="${RUNS_DIR}/${TAG}_${TS}"
LOG_FILE="${RUN_DIR}/codex_launch.log"
CMD_FILE="${RUN_DIR}/launch_cmd.txt"

mkdir -p "${RUN_DIR}" "${RUN_DIR}/logs"

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

CFG="${TRAIN_CFG:-${ROOT}/configs/train_rtdetr_rggb.json}"
PRETRAIN_MODE="${PRETRAIN_MODE:-checkpoint}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-official:r50_6x_coco}"
PRETRAIN_KEY="${PRETRAIN_KEY:-ema.module}"
PRETRAIN_SCOPE="${PRETRAIN_SCOPE:-all}"
PRETRAIN_CACHE_DIR="${PRETRAIN_CACHE_DIR:-/tmp/torch_hub_checkpoints}"
BACKBONE="${BACKBONE:-resnet50}"
STRICT_PRETRAIN="${STRICT_PRETRAIN:-1}"

CMD=(
  "${PY}" "${ROOT}/train_rtdetr_rggb.py"
  --cfg "${CFG}"
  --out "${RUN_DIR}"
  --backbone "${BACKBONE}"
  --pretrain-mode "${PRETRAIN_MODE}"
  --pretrain-ckpt "${PRETRAIN_CKPT}"
  --pretrain-key "${PRETRAIN_KEY}"
  --pretrain-scope "${PRETRAIN_SCOPE}"
  --pretrain-cache-dir "${PRETRAIN_CACHE_DIR}"
)

if [[ "${STRICT_PRETRAIN}" == "1" ]]; then
  CMD+=(--no-pretrain-suffix-match)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "RUN_DIR=${RUN_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "CFG=${CFG}"
  echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
  echo "PRETRAIN_KEY=${PRETRAIN_KEY}"
  echo "PRETRAIN_SCOPE=${PRETRAIN_SCOPE}"
  echo "PRETRAIN_CACHE_DIR=${PRETRAIN_CACHE_DIR}"
  echo "BACKBONE=${BACKBONE}"
  echo "STRICT_PRETRAIN=${STRICT_PRETRAIN}"
  printf "CMD="
  printf "%q " "${CMD[@]}"
  echo
} | tee -a "${CMD_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
