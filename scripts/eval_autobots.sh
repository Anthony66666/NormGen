#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MODEL_PATH="${1:-${AUTOBOTS_MODEL_PATH:-}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  shift
fi

AUTOBOTS_ROOT="${AUTOBOTS_ROOT:-../AutoBots}"
AUTOBOTS_DATASET_DIR="${AUTOBOTS_DATASET_DIR:-autobots_data/normgen_generated}"

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "Missing AutoBots checkpoint." >&2
  echo "Usage: bash scripts/eval_autobots.sh /path/to/best_models_fde.pth [extra eval args]" >&2
  exit 1
fi

if [[ ! -d "$AUTOBOTS_ROOT" ]]; then
  echo "Missing AutoBots repo: $AUTOBOTS_ROOT" >&2
  exit 1
fi

if [[ ! -f "$AUTOBOTS_DATASET_DIR/val_dataset.hdf5" ]]; then
  echo "Missing val_dataset.hdf5 under $AUTOBOTS_DATASET_DIR." >&2
  exit 1
fi

cd "$AUTOBOTS_ROOT"
python evaluate.py \
  --models-path "$MODEL_PATH" \
  --dataset-path "$ROOT_DIR/$AUTOBOTS_DATASET_DIR" \
  --batch-size "${AUTOBOTS_EVAL_BATCH_SIZE:-16}" \
  "$@"
