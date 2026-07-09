#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MODE="${MODE:-prediction}"
CONFIG="${CONFIG:-configs/${MODE}.yaml}"
COMBINED_PATH="${COMBINED_PATH:-data/interaction_multi_train_combined.npz}"
NUM_GPUS="${NUM_GPUS:-}"
LAUNCHER="${LAUNCHER:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  COMBINED_PATH="$1"
  shift
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$COMBINED_PATH" ]]; then
  echo "Missing combined dataset: $COMBINED_PATH" >&2
  echo "Set COMBINED_PATH in .env or pass it as the first argument." >&2
  exit 1
fi

mkdir -p results runs

if [[ -n "$NUM_GPUS" && "$NUM_GPUS" != "0" && "$NUM_GPUS" != "1" ]]; then
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NUM_GPUS" train_combined.py \
    --launcher torchrun \
    --config "$CONFIG" \
    --combined_path "$COMBINED_PATH" \
    "$@"
else
  "$PYTHON_BIN" train_combined.py \
    --launcher "$LAUNCHER" \
    --config "$CONFIG" \
    --combined_path "$COMBINED_PATH" \
    "$@"
fi
