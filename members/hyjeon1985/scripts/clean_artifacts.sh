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

DRY_RUN="${DRY_RUN:-0}"
KEEP_RECENT_RUNS="${KEEP_RECENT_RUNS:-0}"

echo "[clean] ROOT_DIR=${ROOT_DIR}"
echo "[clean] RUNS_DIR=${RUNS_DIR}"
echo "[clean] LOG_DIR=${LOG_DIR}"
echo "[clean] CACHE_DIR=${CACHE_DIR} (kept)"

remove_path() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] rm -rf $p"
  else
    rm -rf "$p"
    echo "[clean] removed: $p"
  fi
}

remove_glob() {
  local pattern="$1"
  local matched=0
  shopt -s nullglob
  local paths=( $pattern )
  shopt -u nullglob
  if [[ ${#paths[@]} -eq 0 ]]; then
    return 0
  fi
  matched=1
  if [[ "$DRY_RUN" == "1" ]]; then
    local p
    for p in "${paths[@]}"; do
      echo "[dry-run] rm -rf $p"
    done
  else
    rm -rf -- "${paths[@]}"
    echo "[clean] removed pattern: $pattern"
  fi
}

if [[ "$KEEP_RECENT_RUNS" == "0" ]]; then
  remove_path "${RUNS_DIR}"
else
  if [[ -d "${RUNS_DIR}" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] keep recent runs enabled; pruning skipped"
    else
      echo "[clean] KEEP_RECENT_RUNS=${KEEP_RECENT_RUNS} is not implemented as prune; skipping RUNS_DIR cleanup"
    fi
  fi
fi

remove_path "${LOG_DIR}"

if [[ -d "${ROOT_DIR}/tmp" ]]; then
  remove_path "${ROOT_DIR}/tmp"
fi

echo "[clean] done (cache preserved)"
