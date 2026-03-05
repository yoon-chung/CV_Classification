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
RANDOM_TRIALS="${5:-3000}"

DIRECTION="ensemble_weight_${VERSION}"
EXP_DIR="experiments/${DATE}/${MODEL_GROUP}/${METHOD}/${DIRECTION}"

"${PYTHON_BIN}" scripts/search_ensemble_weights.py \
  --exp-dir "${EXP_DIR}" \
  --random-trials "${RANDOM_TRIALS}"

"${PYTHON_BIN}" scripts/infer_ensemble.py \
  --exp-dir "${EXP_DIR}" \
  --data-config configs/base/data.yaml
