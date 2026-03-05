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
export RUNNER_PROFILE="${RUNNER_PROFILE:-hpc_confirm}"
if [[ "${CVDC_REQUIRE_CUDA:-1}" != "1" ]]; then
  echo "[warn] CVDC_REQUIRE_CUDA was set to '${CVDC_REQUIRE_CUDA}'. Overriding to '1' for team final solve safety."
fi
export CVDC_REQUIRE_CUDA="1"
export SLACK_NOTIFY="${SLACK_NOTIFY:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi

"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[pre] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[pre] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"

DATE="$(date +%Y%m%d)"
STAMP="$(date +%H%M%S)"
RUN_DATE_TAG="$(date +%Y-%m-%d)"
export WANDB_GROUP="${WANDB_GROUP:-team_solve_final__date=${DATE}__ts=${STAMP}}"

echo "=========================================="
echo "Starting TEAM SOLVE FINAL (3 outputs, soft-vote only)"
echo "=========================================="
echo "Scenario: ${SCENARIO}"
echo "Runner: ${RUNNER_PROFILE}"
echo "W&B Project: ${WANDB_PROJECT:-<from config/env>}"
echo "W&B Group: ${WANDB_GROUP}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "=========================================="

COMMON_ARGS=(
  scenario="${SCENARIO}"
  runner_profile="${RUNNER_PROFILE}"
  kind=solve
  solve=team_solve_final
  train.num_workers="${NUM_WORKERS}"
  train.amp=true
)

RUN_BASE="${RUN_ID_BASE:-solve-team-final-${DATE}-${STAMP}}"
MATRIX=(
  top1:base:top1_base
  top1:convnext_384:top1_convnext_384
  top2:base:top2_base
)
echo "Solve final matrix: ${MATRIX[*]}"

RUN_DIR_TOP1_BASE=""
RUN_DIR_TOP1_384=""
RUN_DIR_TOP2_BASE=""

for idx in "${!MATRIX[@]}"; do
  triplet="${MATRIX[$idx]}"
  candidate="${triplet%%:*}"
  remain="${triplet#*:}"
  backbone="${remain%%:*}"
  key="${remain##*:}"
  if [[ -z "${candidate}" || -z "${backbone}" || -z "${key}" ]]; then
    echo "[solve-final] invalid matrix item='${triplet}'" >&2
    exit 2
  fi

  run_id="${RUN_BASE}-$((idx + 1))-${candidate}-${backbone}"
  run_dir="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/run_id=${run_id}"
  echo "[solve-final] starting key=${key} candidate=${candidate} backbone=${backbone} run_id=${run_id}"

  RUN_ID="${run_id}" "${PYTHON_BIN}" -m experiment \
    "${COMMON_ARGS[@]}" \
    solve_candidate="${candidate}" \
    solve_backbone="${backbone}" \
    "$@"

  if [[ "${key}" == "top1_base" ]]; then
    RUN_DIR_TOP1_BASE="${run_dir}"
  elif [[ "${key}" == "top1_convnext_384" ]]; then
    RUN_DIR_TOP1_384="${run_dir}"
  elif [[ "${key}" == "top2_base" ]]; then
    RUN_DIR_TOP2_BASE="${run_dir}"
  fi
done

if [[ -z "${RUN_DIR_TOP1_BASE}" || -z "${RUN_DIR_TOP1_384}" || -z "${RUN_DIR_TOP2_BASE}" ]]; then
  echo "[solve-final] missing one or more run directories for final submission build." >&2
  exit 3
fi

ENSEMBLE_OUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/ensemble/${RUN_BASE}"
mkdir -p "${ENSEMBLE_OUT_DIR}"
"${PYTHON_BIN}" -m experiment.solve.ensemble_metrics \
  --run-dir "${RUN_DIR_TOP1_BASE}" \
  --run-dir "${RUN_DIR_TOP1_384}" \
  --run-dir "${RUN_DIR_TOP2_BASE}" \
  --out-dir "${ENSEMBLE_OUT_DIR}"
echo "[solve-final] ensemble analysis saved to ${ENSEMBLE_OUT_DIR}"

FINAL_OUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/final_submissions/${RUN_BASE}"
mkdir -p "${FINAL_OUT_DIR}"
"${PYTHON_BIN}" -m experiment.solve.final_builder \
  --model "top1_base=${RUN_DIR_TOP1_BASE}" \
  --model "top1_convnext_384=${RUN_DIR_TOP1_384}" \
  --model "top2_base=${RUN_DIR_TOP2_BASE}" \
  --single-key top1_convnext_384 \
  --pair-keys top1_base top1_convnext_384 \
  --triple-keys top1_base top1_convnext_384 top2_base \
  --weight-mode val_macro_f1 \
  --output-single submission_1_single_top1_convnext_384.csv \
  --output-pair-soft submission_2_pair_soft_top1_base_top1_convnext_384.csv \
  --output-triple-soft submission_3_triple_soft_top1_base_top1_convnext_384_top2_base.csv \
  --out-dir "${FINAL_OUT_DIR}"
echo "[solve-final] final submissions saved to ${FINAL_OUT_DIR}"

echo "[solve-final] completed."
