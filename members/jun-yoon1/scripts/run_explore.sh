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
METHOD="${3:-explore}"
DIRECTION="${4:-aug_low_f1_v1}"
FOLDS="${5:-3}"

"${PYTHON_BIN}" scripts/eda_report.py

"${PYTHON_BIN}" scripts/train.py \
  --date "${DATE}" \
  --model "${MODEL_GROUP}" \
  --method "${METHOD}" \
  --direction "${DIRECTION}" \
  --folds "${FOLDS}" \
  --data-config configs/base/data.yaml \
  --model-config configs/base/model.yaml \
  --train-config configs/base/train.yaml
