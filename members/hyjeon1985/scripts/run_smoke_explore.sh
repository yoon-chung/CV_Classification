#!/usr/bin/env bash
#
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

PRESET="${1:-team_focus}"
if [[ $# -ge 1 ]]; then
  shift
fi

RUNNER_PROFILE="${RUNNER_PROFILE:-local_proxy}"

export SLACK_NOTIFY="${SLACK_NOTIFY:-0}"
export CVDC_REQUIRE_CUDA="${CVDC_REQUIRE_CUDA:-1}"

DATE="$(date +%F)"
RUN_ID="${RUN_ID:-smoke-$(date +%H%M%S)}"
SMOKE_SAVE_DIR="${SMOKE_SAVE_DIR:-${RUNS_DIR}/_smoke}"
SMOKE_RUN_DIR="${SMOKE_SAVE_DIR}/${RUNNER_PROFILE}/date-${DATE}/runid-${RUN_ID}__${PRESET}"

SMOKE_KEEP="${SMOKE_KEEP:-0}"
SMOKE_KEEP_ON_FAIL="${SMOKE_KEEP_ON_FAIL:-1}"

# Pre-check CUDA
"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[smoke] CVDC_REQUIRE_CUDA=' + str(os.environ.get('CVDC_REQUIRE_CUDA'))); print('[smoke] CUDA_VISIBLE_DEVICES=' + str(os.environ.get('CUDA_VISIBLE_DEVICES'))); print('[smoke] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[smoke] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"

set +e
"${PYTHON_BIN}" -m experiment \
  scenario=local \
  runner_profile="${RUNNER_PROFILE}" \
  runner.dummy_data=true \
  kind=explore \
  explore="${PRESET}" \
  explore.max_jobs=1 \
  explore.selection.topk=1 \
  train.epochs=1 \
  pipeline.stop_after=eval \
  wandb.mode=disabled \
  upload.s3.enabled=false \
  "hydra.run.dir=${SMOKE_RUN_DIR}" \
  "$@"
RC=$?
set -e

echo "[smoke] run_dir=${SMOKE_RUN_DIR}"

SUMMARY_PATH="${SMOKE_RUN_DIR}/explore_summary.json"
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "[smoke] missing explore_summary.json; keeping artifacts for debugging" >&2
  SMOKE_KEEP=1
fi

if [[ "${SMOKE_KEEP}" != "1" ]]; then
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path("${SMOKE_RUN_DIR}")
children = root / "children"
eval_paths = list(children.rglob("eval.json")) if children.exists() else []
if not eval_paths:
    raise SystemExit("no child eval.json found")

ok = False
for p in eval_paths:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(data, dict) and "macro_f1" in data:
        ok = True
        break
if not ok:
    raise SystemExit("macro_f1 missing in all child eval.json")
print("eval.json macro_f1 OK")
PY
  if [[ $? -ne 0 ]]; then
    echo "[smoke] missing eval.json macro_f1; keeping artifacts for debugging" >&2
    SMOKE_KEEP=1
  fi
fi

if [[ ${RC} -ne 0 ]]; then
  echo "[smoke] FAILED (rc=${RC})" >&2
  if [[ "${SMOKE_KEEP_ON_FAIL}" != "1" ]]; then
    echo "[smoke] cleanup_on_fail enabled; removing ${SMOKE_RUN_DIR}" >&2
    rm -rf "${SMOKE_RUN_DIR}" || true
  else
    echo "[smoke] keeping artifacts for debugging (set SMOKE_KEEP_ON_FAIL=0 to remove)" >&2
  fi
  exit ${RC}
fi

if [[ "${SMOKE_KEEP}" == "1" ]]; then
  echo "[smoke] success; keeping artifacts (SMOKE_KEEP=1)"
  exit 0
fi

# Guard: only cleanup if SMOKE_RUN_DIR is under RUNS_DIR/_smoke/
case "${SMOKE_RUN_DIR}" in
  "${RUNS_DIR}"/_smoke/*) ;;
  *)
    echo "[smoke] ERROR: cleanup guard triggered - SMOKE_RUN_DIR is not under RUNS_DIR/_smoke/" >&2
    echo "[smoke] SMOKE_RUN_DIR=${SMOKE_RUN_DIR}, RUNS_DIR=${RUNS_DIR}" >&2
    exit 2
    ;;
esac

echo "[smoke] success; cleaning up ${SMOKE_RUN_DIR}"
rm -rf "${SMOKE_RUN_DIR}"
