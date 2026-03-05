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

usage() {
  cat <<'EOF'
Usage:
  bash members/hyjeon1985/scripts/archive_team_solve.sh [options]

Options:
  --date YYYY-MM-DD          Source date tag under outputs (default: today)
  --runner-profile NAME      Runner profile under outputs (default: hpc_confirm)
  --run-base NAME            Run base without "run_id=" (ex: solve-team-final-20260303-101010)
                             If omitted, auto-selects latest base from ensemble/final_submissions.
  --kind auto|solve-team|solve-team-final|team_solve|team_solve_final
                             Archive namespace under archive/date=.../solve (default: auto)
  --dry-run                  Print resolved paths and exit
  -h, --help                 Show this help

Examples:
  bash members/hyjeon1985/scripts/archive_team_solve.sh
  bash members/hyjeon1985/scripts/archive_team_solve.sh --date 2026-03-03 --run-base solve-team-20260303-152256
  bash members/hyjeon1985/scripts/archive_team_solve.sh --run-base solve-team-final-20260303-220501 --kind team_solve_final
EOF
}

ARCHIVE_DATE="$(date +%Y-%m-%d)"
RUNNER_PROFILE="${RUNNER_PROFILE:-hpc_confirm}"
RUN_BASE="${RUN_BASE:-}"
ARCHIVE_KIND="auto"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      ARCHIVE_DATE="${2:-}"
      shift 2
      ;;
    --runner-profile)
      RUNNER_PROFILE="${2:-}"
      shift 2
      ;;
    --run-base)
      RUN_BASE="${2:-}"
      shift 2
      ;;
    --kind)
      ARCHIVE_KIND="${2:-auto}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${MEMBER_ID:-}" ]]; then
  export MEMBER_ID="$(resolve_member_id_from_script "${BASH_SOURCE[0]}")"
fi
init_member_workspace_paths "${PROJECT_ROOT}" "${MEMBER_ID}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="$(resolve_python_bin "${PYTHON_BIN:-}")"

DATE_OUTPUT_DIR="${RUNS_DIR}/${RUNNER_PROFILE}/date=${ARCHIVE_DATE}"
if [[ ! -d "${DATE_OUTPUT_DIR}" ]]; then
  echo "Error: output date dir not found: ${DATE_OUTPUT_DIR}" >&2
  exit 3
fi

auto_pick_run_base() {
  local date_output_dir="$1"
  local latest=""
  latest="$(
    find "${date_output_dir}/ensemble" "${date_output_dir}/final_submissions" \
      -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
      -printf '%T@ %f\n' \
      | sort -nr \
      | head -n1 \
      | awk '{print $2}'
  )"
  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
    return 0
  fi

  # Fallback: infer by run_id directory names.
  latest="$(
    find "${date_output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'run_id=*' \
      -printf '%T@ %f\n' \
      | sort -nr \
      | head -n1 \
      | awk '{print $2}' \
      | sed 's/^run_id=//'
  )"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  # Strip trailing "-N-..." pattern to get base.
  if [[ "${latest}" =~ ^(.+)-[0-9]+-[^-]+-[^-]+$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' "${latest}"
  fi
}

if [[ -z "${RUN_BASE}" ]]; then
  RUN_BASE="$(auto_pick_run_base "${DATE_OUTPUT_DIR}")" || {
    echo "Error: failed to auto-detect RUN_BASE from ${DATE_OUTPUT_DIR}" >&2
    exit 4
  }
fi

if [[ "${ARCHIVE_KIND}" == "auto" ]]; then
  if [[ "${RUN_BASE}" == solve-team-final-* ]]; then
    ARCHIVE_KIND="solve-team-final"
  else
    ARCHIVE_KIND="solve-team"
  fi
fi

if [[ "${ARCHIVE_KIND}" == "team_solve" ]]; then
  ARCHIVE_KIND="solve-team"
elif [[ "${ARCHIVE_KIND}" == "team_solve_final" ]]; then
  ARCHIVE_KIND="solve-team-final"
fi

if [[ "${ARCHIVE_KIND}" != "solve-team" && "${ARCHIVE_KIND}" != "solve-team-final" ]]; then
  echo "Error: --kind must be auto|solve-team|solve-team-final|team_solve|team_solve_final, got: ${ARCHIVE_KIND}" >&2
  exit 5
fi

mapfile -t RUN_DIRS < <(
  find "${DATE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name "run_id=${RUN_BASE}-*" | sort
)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  # Fallback for run bases whose outputs are aggregated under ensemble/final_submissions
  # (e.g. solve-team-final-multiseed-...).
  mapfile -t RUN_DIRS < <(
    "${PYTHON_BIN}" - <<'PY' "${DATE_OUTPUT_DIR}" "${RUN_BASE}"
import json
import sys
from pathlib import Path

date_output_dir = Path(sys.argv[1])
run_base = sys.argv[2]
paths = []

ensemble_runs = date_output_dir / "ensemble" / run_base / "ensemble_runs.json"
if ensemble_runs.exists():
    try:
        payload = json.loads(ensemble_runs.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    run_dir = row.get("run_dir")
                    if isinstance(run_dir, str) and run_dir.strip():
                        paths.append(run_dir.strip())
    except Exception:
        pass

final_summary = date_output_dir / "final_submissions" / run_base / "final_submission_summary.json"
if final_summary.exists():
    try:
        payload = json.loads(final_summary.read_text(encoding="utf-8"))
        models = payload.get("models", {})
        if isinstance(models, dict):
            for run_dir in models.values():
                if isinstance(run_dir, str) and run_dir.strip():
                    paths.append(run_dir.strip())
    except Exception:
        pass

seen = set()
for item in sorted(paths):
    p = Path(item)
    if p.exists() and p.is_dir():
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            print(s)
PY
  )
fi
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  if [[ -d "${DATE_OUTPUT_DIR}/ensemble/${RUN_BASE}" || -d "${DATE_OUTPUT_DIR}/final_submissions/${RUN_BASE}" ]]; then
    echo "[archive] warning: no per-run directories found; archiving ensemble/final_submissions only for run_base=${RUN_BASE}" >&2
  else
    echo "Error: no run dirs matched run_id=${RUN_BASE}-* under ${DATE_OUTPUT_DIR}" >&2
    exit 6
  fi
fi

ARCHIVE_ROOT="${ROOT_DIR}/archive/date=${ARCHIVE_DATE}/solve/${ARCHIVE_KIND}/run_id=${RUN_BASE}"
ARCHIVE_RUNS_DIR="${ARCHIVE_ROOT}/runs"
ARCHIVE_ENSEMBLE_DIR="${ARCHIVE_ROOT}/ensemble"
ARCHIVE_FINAL_SUB_DIR="${ARCHIVE_ROOT}/final_submissions"

echo "[archive] date=${ARCHIVE_DATE}"
echo "[archive] runner_profile=${RUNNER_PROFILE}"
echo "[archive] run_base=${RUN_BASE}"
echo "[archive] kind=${ARCHIVE_KIND}"
echo "[archive] source=${DATE_OUTPUT_DIR}"
echo "[archive] target=${ARCHIVE_ROOT}"
echo "[archive] run_count=${#RUN_DIRS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[archive] runs:\n'
  printf '  - %s\n' "${RUN_DIRS[@]}"
  exit 0
fi

mkdir -p "${ARCHIVE_RUNS_DIR}" "${ARCHIVE_ENSEMBLE_DIR}" "${ARCHIVE_FINAL_SUB_DIR}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
  fi
}

for run_dir in "${RUN_DIRS[@]}"; do
  run_name="$(basename "${run_dir}")"
  out_dir="${ARCHIVE_RUNS_DIR}/${run_name}"
  mkdir -p "${out_dir}"

  copy_if_exists "${run_dir}/prep.json" "${out_dir}/prep.json"
  copy_if_exists "${run_dir}/train.json" "${out_dir}/train.json"
  copy_if_exists "${run_dir}/eval.json" "${out_dir}/eval.json"
  copy_if_exists "${run_dir}/infer.json" "${out_dir}/infer.json"
  copy_if_exists "${run_dir}/submission.json" "${out_dir}/submission.json"
  copy_if_exists "${run_dir}/predictions.csv" "${out_dir}/predictions.csv"
  copy_if_exists "${run_dir}/submission.csv" "${out_dir}/submission.csv"

  copy_if_exists "${run_dir}/.hydra/config.yaml" "${out_dir}/hydra/config.yaml"
  copy_if_exists "${run_dir}/.hydra/overrides.yaml" "${out_dir}/hydra/overrides.yaml"
  copy_if_exists "${run_dir}/.hydra/hydra.yaml" "${out_dir}/hydra/hydra.yaml"

  copy_if_exists "${run_dir}/artifacts/eval/class_metrics.csv" "${out_dir}/artifacts/eval/class_metrics.csv"
  copy_if_exists "${run_dir}/artifacts/eval/val_predictions.csv" "${out_dir}/artifacts/eval/val_predictions.csv"
  copy_if_exists "${run_dir}/artifacts/infer/predictions_with_confidence.csv" "${out_dir}/artifacts/infer/predictions_with_confidence.csv"
  copy_if_exists "${run_dir}/artifacts/infer/predictions_proba.npz" "${out_dir}/artifacts/infer/predictions_proba.npz"
done

if [[ -d "${DATE_OUTPUT_DIR}/ensemble/${RUN_BASE}" ]]; then
  cp -f "${DATE_OUTPUT_DIR}/ensemble/${RUN_BASE}"/* "${ARCHIVE_ENSEMBLE_DIR}/" 2>/dev/null || true
fi
if [[ -d "${DATE_OUTPUT_DIR}/final_submissions/${RUN_BASE}" ]]; then
  cp -f "${DATE_OUTPUT_DIR}/final_submissions/${RUN_BASE}"/* "${ARCHIVE_FINAL_SUB_DIR}/" 2>/dev/null || true
fi

RUN_DIRS_JOINED="$(printf '%s\n' "${RUN_DIRS[@]}")"
export RUN_DIRS_JOINED
export ARCHIVE_ROOT
export DATE_OUTPUT_DIR
export RUN_BASE
export ARCHIVE_KIND
export ARCHIVE_DATE
export RUNNER_PROFILE

"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

archive_root = Path(os.environ["ARCHIVE_ROOT"])
run_dirs = [Path(p) for p in os.environ.get("RUN_DIRS_JOINED", "").splitlines() if p.strip()]

rows = []
for run_dir in run_dirs:
    run_name = run_dir.name
    eval_path = run_dir / "eval.json"
    payload = {}
    if eval_path.exists():
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    row = {
        "run_id": run_name,
        "macro_f1": payload.get("macro_f1", ""),
        "val_loss": payload.get("val/loss", ""),
        "overfit_gap": payload.get("selection/overfit_gap", ""),
        "error_rate": payload.get("selection/error_rate", ""),
        "class_f1_std": payload.get("selection/class_f1_std", ""),
        "high_conf_wrong_rate": payload.get("selection/high_conf_wrong_rate", ""),
        "confidence_mean": payload.get("selection/confidence_mean", ""),
        "low_margin_rate": payload.get("selection/low_margin_rate", ""),
    }
    rows.append(row)

summary_path = archive_root / "runs_summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as fp:
    fieldnames = [
        "run_id",
        "macro_f1",
        "val_loss",
        "overfit_gap",
        "error_rate",
        "class_f1_std",
        "high_conf_wrong_rate",
        "confidence_mean",
        "low_margin_rate",
    ]
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

archived_file_count = sum(1 for _ in archive_root.rglob("*") if _.is_file())
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": {
        "outputs_date_dir": os.environ["DATE_OUTPUT_DIR"],
        "run_base": os.environ["RUN_BASE"],
        "runner_profile": os.environ["RUNNER_PROFILE"],
        "run_glob": f"run_id={os.environ['RUN_BASE']}-*",
    },
    "archived": {
        "archive_root": str(archive_root),
        "kind": os.environ["ARCHIVE_KIND"],
        "run_count": len(run_dirs),
        "archived_file_count": archived_file_count,
        "include": [
            "runs/*/{prep,train,eval,infer,submission}.json",
            "runs/*/{predictions.csv,submission.csv}",
            "runs/*/.hydra/{config,overrides,hydra}.yaml",
            "runs/*/artifacts/eval/{class_metrics.csv,val_predictions.csv}",
            "runs/*/artifacts/infer/predictions_with_confidence.csv",
            "runs/*/artifacts/infer/predictions_proba.npz",
            "ensemble/*",
            "final_submissions/*",
            "runs_summary.csv",
        ],
    },
    "notes": [
        "Hardware/system metadata files are excluded.",
        "Heavy model artifacts (checkpoints/binaries) are excluded.",
    ],
}
manifest_path = archive_root / "archive_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

SOLVE_README="${ROOT_DIR}/archive/date=${ARCHIVE_DATE}/solve/README.md"
mkdir -p "$(dirname "${SOLVE_README}")"
if [[ ! -f "${SOLVE_README}" ]]; then
  cat > "${SOLVE_README}" <<'EOF'
# 제출/추론(solve) 아카이브

- `solve-team/`: solve 기본 매트릭스 아카이브
- `solve-team-final/`: 최종 제출 조합 아카이브
EOF
fi

if [[ ! -f "${ROOT_DIR}/archive/date=${ARCHIVE_DATE}/solve/solve-team-final/README.md" ]]; then
  mkdir -p "${ROOT_DIR}/archive/date=${ARCHIVE_DATE}/solve/solve-team-final"
  cat > "${ROOT_DIR}/archive/date=${ARCHIVE_DATE}/solve/solve-team-final/README.md" <<'EOF'
# 최종 제출(solve-team-final) 아카이브

최종 제출용 solve 결과를 보관합니다.
EOF
fi

echo "[archive] scrubbing absolute paths in archived text files"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scrub_archive_paths.py" \
  --root "${ARCHIVE_ROOT}" \
  --repo-root "${PROJECT_ROOT}" >/dev/null

echo "[archive] completed: ${ARCHIVE_ROOT}"
