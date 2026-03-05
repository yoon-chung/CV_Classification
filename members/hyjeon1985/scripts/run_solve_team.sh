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
  echo "[warn] CVDC_REQUIRE_CUDA was set to '${CVDC_REQUIRE_CUDA}'. Overriding to '1' for team solve safety."
fi
export CVDC_REQUIRE_CUDA="1"
export SLACK_NOTIFY="${SLACK_NOTIFY:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SOLVE_SKIP_HOLDOUT="${SOLVE_SKIP_HOLDOUT:-0}"
export SOLVE_FULL_TRAIN_AFTER_HOLDOUT="${SOLVE_FULL_TRAIN_AFTER_HOLDOUT:-1}"

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
fi

"${PYTHON_BIN}" -c "import os, sys, torch; req=str(os.environ.get('CVDC_REQUIRE_CUDA','0')).strip()=='1'; print('[pre] torch.cuda.is_available=' + str(torch.cuda.is_available())); print('[pre] torch.cuda.device_count=' + str(torch.cuda.device_count())); sys.exit(0 if (not req or torch.cuda.is_available()) else 2)"

DATE="$(date +%Y%m%d)"
STAMP="$(date +%H%M%S)"
export WANDB_GROUP="${WANDB_GROUP:-team_solve__date=${DATE}__ts=${STAMP}}"

echo "=========================================="
echo "Starting TEAM SOLVE (top1-3 x 2 backbones, optional multi-fold)"
echo "=========================================="
echo "Scenario: ${SCENARIO}"
echo "Runner: ${RUNNER_PROFILE}"
echo "W&B Project: ${WANDB_PROJECT:-<from config/env>}"
echo "W&B Group: ${WANDB_GROUP}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "SKIP_HOLDOUT: ${SOLVE_SKIP_HOLDOUT}"
echo "FULL_TRAIN_AFTER_HOLDOUT: ${SOLVE_FULL_TRAIN_AFTER_HOLDOUT}"
echo "=========================================="

COMMON_ARGS=(
  scenario="${SCENARIO}"
  runner_profile="${RUNNER_PROFILE}"
  kind=solve
  train.num_workers="${NUM_WORKERS}"
  train.amp=true
)

RUN_BASE="${RUN_ID_BASE:-solve-team-${DATE}-${STAMP}}"
RUN_DATE_TAG="$(date +%Y-%m-%d)"
if [[ -n "${SOLVE_MATRIX:-}" ]]; then
  IFS=',' read -r -a MATRIX <<< "${SOLVE_MATRIX}"
else
  MATRIX=(
    top1:base
    top1:convnext_384
    top2:base
    top2:convnext_384
    top3:base
    top3:convnext_384
  )
fi
if [[ -n "${SOLVE_FOLDS:-}" ]]; then
  IFS=',' read -r -a FOLDS <<< "${SOLVE_FOLDS}"
else
  FOLDS=(0)
fi
echo "Solve matrix: ${MATRIX[*]}"
echo "Solve folds: ${FOLDS[*]}"

HOLDOUT_RUN_DIRS=()
holdout_count=0
if [[ "${SOLVE_SKIP_HOLDOUT}" != "1" ]]; then
  for fold in "${FOLDS[@]}"; do
    for pair in "${MATRIX[@]}"; do
      candidate="${pair%%:*}"
      backbone="${pair##*:}"
      if [[ -z "${candidate}" || -z "${backbone}" ]]; then
        echo "[solve] invalid pair='${pair}' (expected candidate:backbone)" >&2
        exit 2
      fi
      holdout_count=$((holdout_count + 1))
      run_id="${RUN_BASE}-${holdout_count}-f${fold}-${candidate}-${backbone}"
      echo "[solve:holdout] starting fold=${fold} candidate=${candidate} backbone=${backbone} run_id=${run_id}"
      run_dir="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/run_id=${run_id}"
      RUN_ID="${run_id}" "${PYTHON_BIN}" -m experiment \
        "${COMMON_ARGS[@]}" \
        solve=team_solve \
        solve_candidate="${candidate}" \
        solve_backbone="${backbone}" \
        "$@" \
        "split.fold_index=${fold}"
      HOLDOUT_RUN_DIRS+=("${run_dir}")
    done
  done
fi

if [[ ${#HOLDOUT_RUN_DIRS[@]} -gt 0 ]]; then
  ENSEMBLE_OUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/ensemble/${RUN_BASE}"
  mkdir -p "${ENSEMBLE_OUT_DIR}"
  ENSEMBLE_ARGS=()
  for run_dir in "${HOLDOUT_RUN_DIRS[@]}"; do
    ENSEMBLE_ARGS+=(--run-dir "${run_dir}")
  done
  "${PYTHON_BIN}" -m experiment.solve.ensemble_metrics \
    "${ENSEMBLE_ARGS[@]}" \
    --out-dir "${ENSEMBLE_OUT_DIR}"
  echo "[solve:holdout] ensemble analysis saved to ${ENSEMBLE_OUT_DIR}"
else
  echo "[solve:holdout] skipped."
fi

if [[ "${SOLVE_FULL_TRAIN_AFTER_HOLDOUT}" == "1" ]]; then
  FULLTRAIN_BASE="${RUN_ID_BASE_FULLTRAIN:-solve-team-fulltrain-${DATE}-${STAMP}}"
  echo "[solve:fulltrain] starting full-train runs | base=${FULLTRAIN_BASE}"
  FULLTRAIN_RUN_DIRS=()
  fulltrain_count=0
  for pair in "${MATRIX[@]}"; do
    candidate="${pair%%:*}"
    backbone="${pair##*:}"
    if [[ -z "${candidate}" || -z "${backbone}" ]]; then
      echo "[solve:fulltrain] invalid pair='${pair}' (expected candidate:backbone)" >&2
      exit 2
    fi
    fulltrain_count=$((fulltrain_count + 1))
    run_id="${FULLTRAIN_BASE}-${fulltrain_count}-${candidate}-${backbone}"
    echo "[solve:fulltrain] starting candidate=${candidate} backbone=${backbone} run_id=${run_id}"
    run_dir="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/run_id=${run_id}"
    RUN_ID="${run_id}" "${PYTHON_BIN}" -m experiment \
      "${COMMON_ARGS[@]}" \
      solve=team_solve \
      solve_candidate="${candidate}" \
      solve_backbone="${backbone}" \
      "$@" \
      "split.strategy=full_train" \
      "split.fold_index=0" \
      "train.early_stop.enabled=false"
    FULLTRAIN_RUN_DIRS+=("${run_dir}")
  done

  FULLTRAIN_ENSEMBLE_OUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${RUN_DATE_TAG}/ensemble/${FULLTRAIN_BASE}"
  mkdir -p "${FULLTRAIN_ENSEMBLE_OUT_DIR}"
  FULLTRAIN_ENSEMBLE_ARGS=()
  for run_dir in "${FULLTRAIN_RUN_DIRS[@]}"; do
    FULLTRAIN_ENSEMBLE_ARGS+=(--run-dir "${run_dir}")
  done
  "${PYTHON_BIN}" -m experiment.solve.ensemble_metrics \
    "${FULLTRAIN_ENSEMBLE_ARGS[@]}" \
    --out-dir "${FULLTRAIN_ENSEMBLE_OUT_DIR}"
  echo "[solve:fulltrain] ensemble analysis saved to ${FULLTRAIN_ENSEMBLE_OUT_DIR}"
fi

echo "[solve] completed all presets."
