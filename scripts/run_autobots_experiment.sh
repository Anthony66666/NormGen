#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

GENERATED_NPZ="${1:-${NORMGEN_TRAIN_NPZ:-${NORMGEN_NPZ:-}}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  shift
fi

if [[ -z "$GENERATED_NPZ" || ! -f "$GENERATED_NPZ" ]]; then
  echo "Missing generated NormGen NPZ." >&2
  echo "Usage: bash scripts/run_autobots_experiment.sh /path/to/generated_samples.npz [extra train args]" >&2
  exit 1
fi

AUTOBOTS_SAVE_DIR="${AUTOBOTS_SAVE_DIR:-$ROOT_DIR/autobots_runs}"
EXP_ID="${EXP_ID:-normgen_$(date +%Y%m%d_%H%M%S)}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"
export AUTOBOTS_SAVE_DIR EXP_ID

if [[ "$RUN_PREPARE" == "1" || "$RUN_PREPARE" == "true" || "$RUN_PREPARE" == "True" ]]; then
  NORMGEN_TRAIN_NPZ="$GENERATED_NPZ" bash scripts/prepare_autobots_experiment.sh
fi

if [[ "$RUN_TRAIN" == "1" || "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "True" ]]; then
  bash scripts/train_autobots.sh "$@"
fi

if [[ "$RUN_EVAL" != "1" && "$RUN_EVAL" != "true" && "$RUN_EVAL" != "True" ]]; then
  echo "[ok] AutoBots experiment finished without eval. EXP_ID=$EXP_ID"
  exit 0
fi

if [[ -z "$EVAL_CHECKPOINT" ]]; then
  mapfile -t RESULT_DIRS < <(find "$AUTOBOTS_SAVE_DIR/results/interaction-dataset" -maxdepth 1 -type d -name "*_${EXP_ID}_s*" 2>/dev/null | sort)
  if [[ "${#RESULT_DIRS[@]}" -eq 0 ]]; then
    echo "Could not find AutoBots result directory for EXP_ID=$EXP_ID under $AUTOBOTS_SAVE_DIR." >&2
    echo "Set EVAL_CHECKPOINT=/path/to/checkpoint.pth and rerun with RUN_PREPARE=0 RUN_TRAIN=0." >&2
    exit 1
  fi

  RESULT_DIR="${RESULT_DIRS[-1]}"
  if [[ -f "$RESULT_DIR/best_models_fde.pth" ]]; then
    EVAL_CHECKPOINT="$RESULT_DIR/best_models_fde.pth"
  else
    mapfile -t MODEL_CKPTS < <(find "$RESULT_DIR" -maxdepth 1 -type f -name "models_*.pth" | sort -V)
    if [[ "${#MODEL_CKPTS[@]}" -eq 0 ]]; then
      echo "No eval checkpoint found under $RESULT_DIR." >&2
      echo "AutoBots only writes models_*.pth every 10 epochs, and best_models_fde.pth only if FDE improves below its internal threshold." >&2
      echo "Use AUTOBOTS_EPOCHS>=11 or set EVAL_CHECKPOINT=/path/to/checkpoint.pth." >&2
      exit 1
    fi
    EVAL_CHECKPOINT="${MODEL_CKPTS[-1]}"
  fi
fi

echo "[eval] checkpoint=$EVAL_CHECKPOINT"
bash scripts/eval_autobots.sh "$EVAL_CHECKPOINT"
echo "[ok] AutoBots experiment finished. EXP_ID=$EXP_ID"
