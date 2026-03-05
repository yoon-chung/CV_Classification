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
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CVDC_REQUIRE_CUDA="1"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi

"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[pre] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[pre] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"

SEEDS_RAW="${MULTISEED_SEEDS:-42,52,62}"
SEEDS=()
for tok in ${SEEDS_RAW//,/ }; do
  tok="${tok//[[:space:]]/}"
  if [[ -n "${tok}" ]]; then
    SEEDS+=("${tok}")
  fi
done
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "Error: MULTISEED_SEEDS is empty." >&2
  exit 2
fi

MATRIX="${MULTISEED_MATRIX:-top1:convnext_384}"
if [[ "${MATRIX}" == *","* ]]; then
  echo "Error: MULTISEED_MATRIX currently supports a single candidate:backbone pair, got '${MATRIX}'." >&2
  exit 2
fi

CANDIDATE="${MATRIX%%:*}"
BACKBONE="${MATRIX##*:}"
if [[ -z "${CANDIDATE}" || -z "${BACKBONE}" || "${CANDIDATE}" == "${BACKBONE}" ]]; then
  echo "Error: MULTISEED_MATRIX must be 'candidate:backbone', got '${MATRIX}'." >&2
  exit 2
fi

DATE="$(date +%Y%m%d)"
STAMP="$(date +%H%M%S)"
RUN_DATE_TAG="$(date +%Y-%m-%d)"
RUN_BASE_PREFIX="${MULTISEED_RUN_BASE_PREFIX:-solve-team-fulltrain-multiseed-${DATE}-${STAMP}}"
WANDB_GROUP_BASE="${MULTISEED_WANDB_GROUP_BASE:-team_solve_fulltrain_multiseed__date=${DATE}__ts=${STAMP}}"
BUILD_FINAL="${MULTISEED_BUILD_FINAL:-1}"

echo "=========================================="
echo "Starting TEAM SOLVE MULTISEED (full-train only)"
echo "=========================================="
echo "Scenario: ${SCENARIO}"
echo "Runner: ${RUNNER_PROFILE}"
echo "Seed list: ${SEEDS[*]}"
echo "Matrix: ${MATRIX}"
echo "W&B Group Base: ${WANDB_GROUP_BASE}"
echo "Build Final Ensemble: ${BUILD_FINAL}"
echo "=========================================="

RUN_DIRS=()
KEYS=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  key="seed${seed}"
  run_base="${RUN_BASE_PREFIX}-seed${seed}"
  wandb_group="${WANDB_GROUP_BASE}__seed=${seed}"
  run_id="${run_base}-1-${CANDIDATE}-${BACKBONE}"
  run_dir="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/run_id=${run_id}"

  echo "[multiseed] start seed=${seed} run_id_base_fulltrain=${run_base}"
  SOLVE_SKIP_HOLDOUT=1 \
  SOLVE_FULL_TRAIN_AFTER_HOLDOUT=1 \
  SOLVE_MATRIX="${MATRIX}" \
  RUN_ID_BASE_FULLTRAIN="${run_base}" \
  WANDB_GROUP="${wandb_group}" \
  bash "${SCRIPT_DIR}/run_solve_team.sh" \
    train.seed="${seed}" \
    "$@"

  RUN_DIRS+=("${run_dir}")
  KEYS+=("${key}")
done

if [[ "${BUILD_FINAL}" != "1" ]]; then
  echo "[multiseed] MULTISEED_BUILD_FINAL=${BUILD_FINAL}; skipping final ensemble build."
  exit 0
fi

if [[ ${#RUN_DIRS[@]} -ne 3 ]]; then
  echo "[multiseed] automatic final submission build requires exactly 3 seeds. current count=${#RUN_DIRS[@]} -> skipped." >&2
  exit 0
fi

FINAL_BASE="${MULTISEED_FINAL_BASE:-solve-team-final-multiseed-${DATE}-${STAMP}}"
ENSEMBLE_OUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/ensemble/${FINAL_BASE}"
FINAL_OUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/final_submissions/${FINAL_BASE}"
mkdir -p "${ENSEMBLE_OUT_DIR}" "${FINAL_OUT_DIR}"

ENSEMBLE_ARGS=()
for run_dir in "${RUN_DIRS[@]}"; do
  ENSEMBLE_ARGS+=(--run-dir "${run_dir}")
done
"${PYTHON_BIN}" -m experiment.solve.ensemble_metrics \
  "${ENSEMBLE_ARGS[@]}" \
  --out-dir "${ENSEMBLE_OUT_DIR}"

MODEL_ARGS=()
for i in "${!RUN_DIRS[@]}"; do
  MODEL_ARGS+=(--model "${KEYS[$i]}=${RUN_DIRS[$i]}")
done

"${PYTHON_BIN}" -m experiment.solve.final_builder \
  "${MODEL_ARGS[@]}" \
  --single-key "${KEYS[0]}" \
  --pair-keys "${KEYS[0]}" "${KEYS[1]}" \
  --triple-keys "${KEYS[0]}" "${KEYS[1]}" "${KEYS[2]}" \
  --weight-mode val_macro_f1 \
  --output-single "submission_1_single_${KEYS[0]}.csv" \
  --output-pair-soft "submission_2_pair_soft_${KEYS[0]}_${KEYS[1]}.csv" \
  --output-triple-soft "submission_3_triple_soft_${KEYS[0]}_${KEYS[1]}_${KEYS[2]}.csv" \
  --out-dir "${FINAL_OUT_DIR}"

echo "[multiseed] completed."
echo "[multiseed] ensemble dir: ${ENSEMBLE_OUT_DIR}"
echo "[multiseed] final dir: ${FINAL_OUT_DIR}"
