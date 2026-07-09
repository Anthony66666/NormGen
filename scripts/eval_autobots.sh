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
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTOBOTS_NUM_WORKERS="${AUTOBOTS_NUM_WORKERS:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/server_workspace/matplotlib}"
export AUTOBOTS_NUM_WORKERS MPLCONFIGDIR
MODEL_PATH="$(realpath "$MODEL_PATH")"

if [[ "$AUTOBOTS_DATASET_DIR" = /* ]]; then
  AUTOBOTS_DATASET_PATH="$AUTOBOTS_DATASET_DIR"
else
  AUTOBOTS_DATASET_PATH="$ROOT_DIR/$AUTOBOTS_DATASET_DIR"
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "Missing AutoBots checkpoint." >&2
  echo "Usage: bash scripts/eval_autobots.sh /path/to/best_models_fde.pth [extra eval args]" >&2
  exit 1
fi

if [[ ! -d "$AUTOBOTS_ROOT" ]]; then
  echo "Missing AutoBots repo: $AUTOBOTS_ROOT" >&2
  exit 1
fi
AUTOBOTS_ROOT_PATH="$(realpath "$AUTOBOTS_ROOT")"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$AUTOBOTS_DATASET_PATH/val_dataset.hdf5" ]]; then
  echo "Missing val_dataset.hdf5 under $AUTOBOTS_DATASET_PATH." >&2
  exit 1
fi

# AutoBots load_config mishandles absolute checkpoint paths unless cwd is /.
cd /
"$PYTHON_BIN" "$ROOT_DIR/tools/run_autobots_with_worker_patch.py" "$AUTOBOTS_ROOT_PATH/evaluate.py" \
  --models-path "$MODEL_PATH" \
  --dataset-path "$AUTOBOTS_DATASET_PATH" \
  --batch-size "${AUTOBOTS_EVAL_BATCH_SIZE:-16}" \
  "$@"
