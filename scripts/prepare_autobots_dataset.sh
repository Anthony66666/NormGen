#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

INPUT_NPZ="${1:-${NORMGEN_NPZ:-}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  shift
fi

if [[ -z "$INPUT_NPZ" || ! -f "$INPUT_NPZ" ]]; then
  echo "Missing NormGen NPZ." >&2
  echo "Usage: bash scripts/prepare_autobots_dataset.sh /path/to/normgen_output.npz [extra converter args]" >&2
  exit 1
fi

AUTOBOTS_DATASET_DIR="${AUTOBOTS_DATASET_DIR:-autobots_data/normgen_generated}"
INTERACTION_MAPS_ROOT="${INTERACTION_MAPS_ROOT:-}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SAMPLE_KEY="${SAMPLE_KEY:-conditional_samples}"
MODE_INDEX="${MODE_INDEX:--1}"

MAP_ARGS=()
if [[ -n "$INTERACTION_MAPS_ROOT" && -d "$INTERACTION_MAPS_ROOT" ]]; then
  MAP_ARGS+=(--maps-root "$INTERACTION_MAPS_ROOT" --map-copy-mode symlink)
else
  MAP_ARGS+=(--map-copy-mode dummy --allow-dummy-maps)
fi

python tools/convert_normgen_to_autobots.py \
  --input-npz "$INPUT_NPZ" \
  --output-dir "$AUTOBOTS_DATASET_DIR" \
  --sample-key "$SAMPLE_KEY" \
  --mode-index "$MODE_INDEX" \
  --val-ratio "$VAL_RATIO" \
  "${MAP_ARGS[@]}" \
  "$@"
