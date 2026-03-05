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

export SCENARIO="${SCENARIO:-cloud}"
export RUNNER_PROFILE="${RUNNER_PROFILE:-hpc_proxy}"
if [[ "${CVDC_REQUIRE_CUDA:-1}" != "1" ]]; then
  echo "[warn] CVDC_REQUIRE_CUDA was set to '${CVDC_REQUIRE_CUDA}'. Overriding to '1' for team tune safety."
fi
export CVDC_REQUIRE_CUDA="1"
export SLACK_NOTIFY="${SLACK_NOTIFY:-1}"
export NUM_WORKERS="${NUM_WORKERS:-10}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | awk '/^GPU/{c++} END{print c+0}' || true)
    GPU_COUNT=${GPU_COUNT//[[:space:]]/}
  else
    GPU_COUNT=0
  fi
  if [[ -z "${GPU_COUNT}" ]] || (( GPU_COUNT <= 0 )); then
    export CUDA_VISIBLE_DEVICES="0"
  else
    CVD=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES="${CVD}"
  fi
else
  GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ' | wc -w)
  GPU_COUNT=${GPU_COUNT//[[:space:]]/}
fi

"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[pre] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[pre] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"

echo "=========================================="
echo "Starting TEAM TUNE"
echo "=========================================="
echo "Scenario: ${SCENARIO}"
echo "Runner: ${RUNNER_PROFILE}"
echo "Tune preset: team_tune"
echo "W&B Project: ${WANDB_PROJECT:-<from config/env>}"
echo "W&B Group: ${WANDB_GROUP:-<from config/env>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "Expected sweep jobs: 64"
echo "=========================================="

"${PYTHON_BIN}" -m experiment \
  scenario="${SCENARIO}" \
  runner_profile="${RUNNER_PROFILE}" \
  kind=tune \
  tune=team_tune \
  train.num_workers="${NUM_WORKERS}" \
  train.amp=true \
  "$@"
