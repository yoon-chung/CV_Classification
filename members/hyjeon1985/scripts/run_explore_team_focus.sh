#!/usr/bin/env bash
# Run team_focus exploration on cloud scenario with W&B and Slack notifications
# S3 upload disabled

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

# Cloud scenario configuration
export SCENARIO="cloud"
export RUNNER_PROFILE="${RUNNER_PROFILE:-hpc_proxy}"

export CVDC_REQUIRE_CUDA="${CVDC_REQUIRE_CUDA:-1}"

# Slack notifications (requires SLACK_WEBHOOK_URL in .env)
export SLACK_NOTIFY="${SLACK_NOTIFY:-1}"

# GPU detection and concurrency configuration (preflight alignment)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Detect GPU count using nvidia-smi
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | awk '/^GPU/{c++} END{print c+0}')
    GPU_COUNT=${GPU_COUNT//[[:space:]]/}
    if [[ -z "${GPU_COUNT}" ]] || (( GPU_COUNT <= 0 )); then
        export CUDA_VISIBLE_DEVICES="0"
        GPU_COUNT=1
    else
        # Create a comma-separated list: 0,1,...,N-1
        CVD=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES="${CVD}"
    fi
else
    # Respect user-provided CUDA_VISIBLE_DEVICES
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ' | wc -w)
    GPU_COUNT=${GPU_COUNT//[[:space:]]/}
fi

# Set MAX_CONCURRENCY based on GPU count
if (( GPU_COUNT < 2 )); then
    export MAX_CONCURRENCY=1
else
    DEFAULT_CONCURRENCY=$(( (GPU_COUNT * 75) / 100 ))
    if (( DEFAULT_CONCURRENCY < 1 )); then
        DEFAULT_CONCURRENCY=1
    fi
    export MAX_CONCURRENCY="${MAX_CONCURRENCY:-${DEFAULT_CONCURRENCY}}"
fi

export NUM_WORKERS="${NUM_WORKERS:-10}"

# Disable S3 upload
export S3_ENABLED="false"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[pre] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[pre] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"
echo "=========================================="
echo "Starting TEAM FOCUS exploration"
echo "=========================================="
echo "Scenario: ${SCENARIO}"
echo "Runner: ${RUNNER_PROFILE}"
echo "W&B Project: ${WANDB_PROJECT:-<from config/env>}"
echo "W&B Group: ${WANDB_GROUP:-<from config/env>}"
echo "Slack Notify: ${SLACK_NOTIFY}"
echo "S3 Upload: disabled"
echo "Max Concurrency: ${MAX_CONCURRENCY}"
echo "=========================================="

"${PYTHON_BIN}" -m experiment \
  scenario=cloud \
  runner_profile="${RUNNER_PROFILE}" \
  kind=explore \
  explore=team_focus \
  runner.max_concurrency="${MAX_CONCURRENCY}" \
  train.num_workers="${NUM_WORKERS}" \
  train.amp=true \
  "$@"
