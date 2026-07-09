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
AUTOBOTS_ROOT="${AUTOBOTS_ROOT:-../AutoBots}"
SERVER_WORK_DIR="${SERVER_WORK_DIR:-$ROOT_DIR/server_workspace}"
SMOKE_DATASET_DIR="${SMOKE_DATASET_DIR:-$SERVER_WORK_DIR/autobots_smoke_data}"
SMOKE_MAX_SCENES="${SMOKE_MAX_SCENES:-4}"
INPUT_NPZ="${1:-${NORMGEN_NPZ:-}}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$SERVER_WORK_DIR/matplotlib}"
export MPLCONFIGDIR

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  shift
fi

if [[ -z "$INPUT_NPZ" || ! -f "$INPUT_NPZ" ]]; then
  echo "Missing NormGen NPZ." >&2
  echo "Usage: bash scripts/smoke_autobots_pipeline.sh /path/to/normgen_output.npz [extra converter args]" >&2
  exit 1
fi

if [[ ! -d "$AUTOBOTS_ROOT" ]]; then
  echo "Missing AutoBots repo: $AUTOBOTS_ROOT" >&2
  echo "Set AUTOBOTS_ROOT in .env or clone AutoBots next to this repo." >&2
  exit 1
fi

mkdir -p "$SERVER_WORK_DIR" "$MPLCONFIGDIR"

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

missing = []
for name in ("h5py", "matplotlib", "numpy", "pyproj", "pyquaternion", "scipy", "sklearn", "torch"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

try:
    importlib.import_module("cv2")
except Exception as exc:
    missing.append(f"opencv-python/cv2: {exc}")

if missing:
    print("Missing AutoBots dependencies:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("Install them with: pip install -r requirements-autobots.txt", file=sys.stderr)
    raise SystemExit(1)
PY

MAP_ARGS=()
if [[ -n "${INTERACTION_MAPS_ROOT:-}" && -d "${INTERACTION_MAPS_ROOT:-}" ]]; then
  MAP_ARGS+=(--maps-root "$INTERACTION_MAPS_ROOT" --map-copy-mode symlink)
else
  MAP_ARGS+=(--map-copy-mode dummy --allow-dummy-maps)
fi

"$PYTHON_BIN" tools/convert_normgen_to_autobots.py \
  --input-npz "$INPUT_NPZ" \
  --output-dir "$SMOKE_DATASET_DIR" \
  --sample-key "${SAMPLE_KEY:-conditional_samples}" \
  --mode-index "${MODE_INDEX:--1}" \
  --val-ratio "${VAL_RATIO:-0.25}" \
  --max-scenes "$SMOKE_MAX_SCENES" \
  "${MAP_ARGS[@]}" \
  "$@"

cd "$AUTOBOTS_ROOT"
PYTHONPATH="$AUTOBOTS_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$SMOKE_DATASET_DIR" <<'PY'
import sys
from datasets.interaction_dataset.dataset import InteractionDataset

dataset_root = sys.argv[1]
for split in ("train", "val"):
    dataset = InteractionDataset(
        dset_path=dataset_root,
        split_name=split,
        evaluation=False,
        use_map_lanes=False,
    )
    item = dataset[0]
    shapes = [getattr(value, "shape", None) for value in item]
    print(f"[ok] {split}: len={len(dataset)} item_shapes={shapes}")
PY

echo "[ok] AutoBots smoke pipeline passed: $SMOKE_DATASET_DIR"
