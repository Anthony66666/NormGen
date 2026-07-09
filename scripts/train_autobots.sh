#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

AUTOBOTS_ROOT="${AUTOBOTS_ROOT:-../AutoBots}"
AUTOBOTS_DATASET_DIR="${AUTOBOTS_DATASET_DIR:-autobots_data/normgen_generated}"
AUTOBOTS_SAVE_DIR="${AUTOBOTS_SAVE_DIR:-$ROOT_DIR/autobots_runs}"
EXP_ID="${EXP_ID:-normgen}"
USE_MAP_LANES="${USE_MAP_LANES:-0}"

if [[ "$AUTOBOTS_DATASET_DIR" = /* ]]; then
  AUTOBOTS_DATASET_PATH="$AUTOBOTS_DATASET_DIR"
else
  AUTOBOTS_DATASET_PATH="$ROOT_DIR/$AUTOBOTS_DATASET_DIR"
fi

if [[ ! -d "$AUTOBOTS_ROOT" ]]; then
  echo "Missing AutoBots repo: $AUTOBOTS_ROOT" >&2
  echo "Set AUTOBOTS_ROOT in .env." >&2
  exit 1
fi

if [[ ! -f "$AUTOBOTS_DATASET_PATH/train_dataset.hdf5" || ! -f "$AUTOBOTS_DATASET_PATH/val_dataset.hdf5" ]]; then
  echo "Missing AutoBots train/val HDF5 files under $AUTOBOTS_DATASET_PATH." >&2
  echo "Run scripts/prepare_autobots_dataset.sh first." >&2
  exit 1
fi

MAP_FLAGS=()
if [[ "$USE_MAP_LANES" == "1" || "$USE_MAP_LANES" == "true" || "$USE_MAP_LANES" == "True" ]]; then
  MAP_FLAGS+=(--use-map-lanes True)
fi

mkdir -p "$AUTOBOTS_SAVE_DIR"

cd "$AUTOBOTS_ROOT"
python train.py \
  --exp-id "$EXP_ID" \
  --dataset interaction-dataset \
  --model-type Autobot-Joint \
  --dataset-path "$AUTOBOTS_DATASET_PATH" \
  --save-dir "$AUTOBOTS_SAVE_DIR" \
  --num-modes "${AUTOBOTS_NUM_MODES:-6}" \
  --hidden-size "${AUTOBOTS_HIDDEN_SIZE:-128}" \
  --num-encoder-layers "${AUTOBOTS_ENCODER_LAYERS:-2}" \
  --num-decoder-layers "${AUTOBOTS_DECODER_LAYERS:-2}" \
  --dropout "${AUTOBOTS_DROPOUT:-0.1}" \
  --entropy-weight "${AUTOBOTS_ENTROPY_WEIGHT:-40.0}" \
  --kl-weight "${AUTOBOTS_KL_WEIGHT:-20.0}" \
  --use-FDEADE-aux-loss True \
  --tx-hidden-size "${AUTOBOTS_TX_HIDDEN_SIZE:-384}" \
  --batch-size "${AUTOBOTS_BATCH_SIZE:-64}" \
  --learning-rate "${AUTOBOTS_LR:-0.00075}" \
  --learning-rate-sched ${AUTOBOTS_LR_SCHED:-10 20 30 40 50} \
  --num-epochs "${AUTOBOTS_EPOCHS:-50}" \
  "${MAP_FLAGS[@]}" \
  "$@"
