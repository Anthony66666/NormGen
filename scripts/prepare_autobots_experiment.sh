#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_NPZ="${1:-${NORMGEN_TRAIN_NPZ:-${NORMGEN_NPZ:-}}}"
REAL_VAL_NPZ="${REAL_VAL_NPZ:-}"
AUTOBOTS_DATASET_DIR="${AUTOBOTS_DATASET_DIR:-autobots_data/normgen_experiment}"
SAMPLE_KEY="${SAMPLE_KEY:-conditional_samples}"
MODE_INDEX="${MODE_INDEX:--1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TRAIN_MAX_SCENES="${TRAIN_MAX_SCENES:--1}"
VAL_MAX_SCENES="${VAL_MAX_SCENES:--1}"
MIN_AGENTS="${MIN_AGENTS:-2}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  shift
fi

if [[ -z "$GENERATED_NPZ" || ! -f "$GENERATED_NPZ" ]]; then
  echo "Missing generated NormGen NPZ." >&2
  echo "Usage: bash scripts/prepare_autobots_experiment.sh /path/to/generated_samples.npz" >&2
  echo "Optional: REAL_VAL_NPZ=/path/to/real_val_combined.npz for strict real validation." >&2
  exit 1
fi

if [[ "$CLEAN_OUTPUT" == "1" || "$CLEAN_OUTPUT" == "true" || "$CLEAN_OUTPUT" == "True" ]]; then
  if [[ -z "$AUTOBOTS_DATASET_DIR" || "$AUTOBOTS_DATASET_DIR" == "/" || "$AUTOBOTS_DATASET_DIR" == "." || "$AUTOBOTS_DATASET_DIR" == "$ROOT_DIR" ]]; then
    echo "Refusing to clean unsafe AUTOBOTS_DATASET_DIR: $AUTOBOTS_DATASET_DIR" >&2
    exit 1
  fi
  rm -rf -- "$AUTOBOTS_DATASET_DIR"
fi
mkdir -p "$AUTOBOTS_DATASET_DIR"

MAP_ARGS=()
if [[ -n "${INTERACTION_MAPS_ROOT:-}" && -d "${INTERACTION_MAPS_ROOT:-}" ]]; then
  MAP_ARGS+=(--maps-root "$INTERACTION_MAPS_ROOT" --map-copy-mode symlink)
else
  MAP_ARGS+=(--map-copy-mode dummy --allow-dummy-maps)
fi

MAX_TRAIN_ARGS=()
if [[ "$TRAIN_MAX_SCENES" != "-1" ]]; then
  MAX_TRAIN_ARGS+=(--max-scenes "$TRAIN_MAX_SCENES")
fi

MAX_VAL_ARGS=()
if [[ "$VAL_MAX_SCENES" != "-1" ]]; then
  MAX_VAL_ARGS+=(--max-scenes "$VAL_MAX_SCENES")
fi

if [[ -n "$REAL_VAL_NPZ" ]]; then
  if [[ ! -f "$REAL_VAL_NPZ" ]]; then
    echo "REAL_VAL_NPZ does not exist: $REAL_VAL_NPZ" >&2
    exit 1
  fi

  echo "[prepare] generated train -> $AUTOBOTS_DATASET_DIR/train_dataset.hdf5"
  "$PYTHON_BIN" tools/convert_normgen_to_autobots.py \
    --input-npz "$GENERATED_NPZ" \
    --output-dir "$AUTOBOTS_DATASET_DIR" \
    --source samples \
    --sample-key "$SAMPLE_KEY" \
    --mode-index "$MODE_INDEX" \
    --split-name train \
    --val-ratio 0.0 \
    --min-agents "$MIN_AGENTS" \
    "${MAX_TRAIN_ARGS[@]}" \
    "${MAP_ARGS[@]}" \
    "$@"

  echo "[prepare] real validation -> $AUTOBOTS_DATASET_DIR/val_dataset.hdf5"
  "$PYTHON_BIN" tools/convert_normgen_to_autobots.py \
    --input-npz "$REAL_VAL_NPZ" \
    --output-dir "$AUTOBOTS_DATASET_DIR" \
    --source combined \
    --split-name val \
    --val-ratio 0.0 \
    --min-agents "$MIN_AGENTS" \
    "${MAX_VAL_ARGS[@]}" \
    "${MAP_ARGS[@]}"
else
  echo "[prepare] REAL_VAL_NPZ not set; splitting generated data into train/val for debugging." >&2
  echo "[prepare] For paper evaluation, set REAL_VAL_NPZ to a real preprocessed combined NPZ." >&2
  "$PYTHON_BIN" tools/convert_normgen_to_autobots.py \
    --input-npz "$GENERATED_NPZ" \
    --output-dir "$AUTOBOTS_DATASET_DIR" \
    --source samples \
    --sample-key "$SAMPLE_KEY" \
    --mode-index "$MODE_INDEX" \
    --val-ratio "$VAL_RATIO" \
    --min-agents "$MIN_AGENTS" \
    "${MAX_TRAIN_ARGS[@]}" \
    "${MAP_ARGS[@]}" \
    "$@"
fi

echo "[ok] AutoBots dataset ready: $AUTOBOTS_DATASET_DIR"
