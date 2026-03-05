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

if [[ ! -f "${ROOT_DIR}/src/experiment/__main__.py" ]]; then
  echo "Error: invalid workspace root (${ROOT_DIR})" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin "${PYTHON_BIN:-}")"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Error: python/python3 executable not found" >&2
  exit 1
fi

QUEUE_ID="${QUEUE_ID:-quickcheck-cloud}"
RUN_ID="${RUN_ID:-quickcheck-$(date +%H%M%S)}"
RUN_DUMMY="${RUN_DUMMY:-true}"
EPOCHS="${EPOCHS:-1}"
STOP_AFTER="${STOP_AFTER:-upload}"
S3_DRY_RUN="${S3_DRY_RUN:-0}"
WORKFLOW_PURPOSE="${WORKFLOW_PURPOSE:-explore}"
SCENARIO="${SCENARIO:-local}"
RUNNER_PROFILE="${RUNNER_PROFILE:-local_proxy}"
KIND="${KIND:-explore}"

RUN_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/queue_id=${QUEUE_ID}/run_id=${RUN_ID}"

cd "${PROJECT_ROOT}"
S3_DRY_RUN="${S3_DRY_RUN}" WORKFLOW_PURPOSE="${WORKFLOW_PURPOSE}" "${PYTHON_BIN}" -m experiment \
  scenario=${SCENARIO} \
  runner_profile=${RUNNER_PROFILE} \
  kind=${KIND} \
  "runner.run_id=${RUN_ID}" \
  runner.resume=false \
  "pipeline.stop_after=${STOP_AFTER}" \
  "runner.dummy_data=${RUN_DUMMY}" \
  "train.epochs=${EPOCHS}"

echo "[quickcheck] run_dir=${RUN_DIR}"
if [[ "${STOP_AFTER}" == "upload" ]]; then
  echo "[quickcheck] upload_manifest=${RUN_DIR}/reports/upload_manifest.json"
  echo "[quickcheck] s3_export=${RUN_DIR}/exports/s3/manifest.json"
  echo "[quickcheck] wandb_export=${RUN_DIR}/exports/wandb/summary.json"
else
  echo "[quickcheck] upload outputs are skipped when STOP_AFTER=${STOP_AFTER}"
fi
