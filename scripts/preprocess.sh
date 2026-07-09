#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

INTERACTION_ROOT="${INTERACTION_ROOT:-}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed_${SPLIT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  INTERACTION_ROOT="$1"
  shift
fi

if [[ -z "$INTERACTION_ROOT" || ! -d "$INTERACTION_ROOT" ]]; then
  echo "Missing INTERACTION dataset root." >&2
  echo "Set INTERACTION_ROOT in .env or pass it as the first argument." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" data_preprocess.py \
  --root "$INTERACTION_ROOT" \
  --split "$SPLIT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
