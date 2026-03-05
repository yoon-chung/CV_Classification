#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMBER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${MEMBER_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python executable not found: ${PYTHON_BIN}" >&2
  exit 127
fi

DATE="${1:-2026-02-27}"
MODEL_GROUP="${2:-ensemble3}"
METHOD="${3:-tune}"
VERSION="${4:-v2}"
FOLDS="${5:-5}"

DIRECTION="ensemble_weight_${VERSION}"
TRAIN_CONFIG="configs/tune/train_${VERSION}.yaml"

if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "Missing train config: ${TRAIN_CONFIG}" >&2
  echo "Use VERSION=v1 or VERSION=v2" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/eda_report.py

"${PYTHON_BIN}" scripts/train.py \
  --date "${DATE}" \
  --model "${MODEL_GROUP}" \
  --method "${METHOD}" \
  --direction "${DIRECTION}" \
  --folds "${FOLDS}" \
  --data-config configs/base/data.yaml \
  --model-config configs/base/model.yaml \
  --train-config "${TRAIN_CONFIG}"
