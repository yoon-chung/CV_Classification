#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CV_PRJ_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export CV_PRJ_ROOT="${PROJECT_ROOT}"

if [[ ! -f "${SCRIPT_DIR}/common_env.sh" ]]; then
  echo "Error: missing ${SCRIPT_DIR}/common_env.sh" >&2
  exit 1
fi

source "${SCRIPT_DIR}/common_env.sh"

export MEMBER_ID="${MEMBER_ID:-$(resolve_member_id_from_script "${BASH_SOURCE[0]}")}"
init_member_workspace_paths "${PROJECT_ROOT}" "${MEMBER_ID}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="$(resolve_python_bin "${PYTHON_BIN:-}")"

PRESET="${1:-default}"
if [[ $# -ge 1 ]]; then
  shift
fi

RUNNER_PROFILE="${RUNNER_PROFILE:-local_proxy}"

export SLACK_NOTIFY="${SLACK_NOTIFY:-0}"
SMOKE_REQUIRE_CUDA="${SMOKE_REQUIRE_CUDA:-0}"
export CVDC_REQUIRE_CUDA="${SMOKE_REQUIRE_CUDA}"

DATE="$(date +%F)"
RUN_ID="${RUN_ID:-smoke-tune-$(date +%H%M%S)}"
SMOKE_SAVE_DIR="${SMOKE_SAVE_DIR:-${RUNS_DIR}/_smoke}"
SMOKE_RUN_DIR="${SMOKE_SAVE_DIR}/${RUNNER_PROFILE}/date-${DATE}/runid-${RUN_ID}__tune__${PRESET}"

SMOKE_KEEP="${SMOKE_KEEP:-0}"
SMOKE_KEEP_ON_FAIL="${SMOKE_KEEP_ON_FAIL:-1}"

# Pre-check CUDA
"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[smoke:tune] CVDC_REQUIRE_CUDA=' + str(os.environ.get('CVDC_REQUIRE_CUDA'))); print('[smoke:tune] CUDA_VISIBLE_DEVICES=' + str(os.environ.get('CUDA_VISIBLE_DEVICES'))); print('[smoke:tune] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[smoke:tune] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"

set +e
"${PYTHON_BIN}" -m experiment \
  scenario=local \
  runner_profile="${RUNNER_PROFILE}" \
  runner.dummy_data=true \
  kind=tune \
  tune="${PRESET}" \
  model.pretrained=false \
  train.num_workers=0 \
  train.epochs=1 \
  pipeline.stop_after=eval \
  wandb.mode=disabled \
  upload.s3.enabled=false \
  "hydra.run.dir=${SMOKE_RUN_DIR}" \
  "$@"
RC=$?
set -e

echo "[smoke:tune] run_dir=${SMOKE_RUN_DIR}"

if [[ ! -f "${SMOKE_RUN_DIR}/train.json" || ! -f "${SMOKE_RUN_DIR}/eval.json" ]]; then
  echo "[smoke:tune] missing train.json or eval.json; keeping artifacts for debugging" >&2
  SMOKE_KEEP=1
fi

if [[ "${SMOKE_KEEP}" != "1" ]]; then
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path("${SMOKE_RUN_DIR}")
eval_path = root / "eval.json"
if not eval_path.exists():
    raise SystemExit("missing eval.json")

data = json.loads(eval_path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit("eval.json is not an object")
if "macro_f1" not in data and "val/macro_f1" not in data:
    metrics = data.get("metrics")
    if not (isinstance(metrics, dict) and ("macro_f1" in metrics or "val/macro_f1" in metrics)):
        raise SystemExit("macro_f1 missing in eval.json")
print("eval.json macro_f1 OK")
PY
  if [[ $? -ne 0 ]]; then
    echo "[smoke:tune] invalid eval.json; keeping artifacts for debugging" >&2
    SMOKE_KEEP=1
  fi
fi

if [[ ${RC} -ne 0 ]]; then
  echo "[smoke:tune] FAILED (rc=${RC})" >&2
  if [[ "${SMOKE_KEEP_ON_FAIL}" != "1" ]]; then
    echo "[smoke:tune] cleanup_on_fail enabled; removing ${SMOKE_RUN_DIR}" >&2
    rm -rf "${SMOKE_RUN_DIR}" || true
  else
    echo "[smoke:tune] keeping artifacts for debugging (set SMOKE_KEEP_ON_FAIL=0 to remove)" >&2
  fi
  exit ${RC}
fi

if [[ "${SMOKE_KEEP}" == "1" ]]; then
  echo "[smoke:tune] success; keeping artifacts (SMOKE_KEEP=1)"
  exit 0
fi

case "${SMOKE_RUN_DIR}" in
  "${RUNS_DIR}"/_smoke/*) ;;
  *)
    echo "[smoke:tune] ERROR: cleanup guard triggered - SMOKE_RUN_DIR is not under RUNS_DIR/_smoke/" >&2
    echo "[smoke:tune] SMOKE_RUN_DIR=${SMOKE_RUN_DIR}, RUNS_DIR=${RUNS_DIR}" >&2
    exit 2
    ;;
esac

echo "[smoke:tune] success; cleaning up ${SMOKE_RUN_DIR}"
rm -rf "${SMOKE_RUN_DIR}"
