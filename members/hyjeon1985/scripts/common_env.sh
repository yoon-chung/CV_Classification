#!/usr/bin/env bash

detect_arch() {
  uname -m 2>/dev/null || echo unknown
}

resolve_member_id_from_script() {
  local script_path="${1:-}"
  local marker="/members/"
  local tail=""

  if [[ -z "${script_path}" ]]; then
    if [[ -n "${MEMBER_ID:-}" ]]; then
      printf "%s\n" "${MEMBER_ID}"
      return 0
    fi
    return 1
  fi

  case "${script_path}" in
    /*) ;;
    *) script_path="$(pwd)/${script_path#./}" ;;
  esac
  if [[ "${script_path}" == *"${marker}"* ]]; then
    tail="${script_path#*${marker}}"
    printf "%s\n" "${tail%%/*}"
    return 0
  fi

  if [[ -n "${MEMBER_ID:-}" ]]; then
    printf "%s\n" "${MEMBER_ID}"
    return 0
  fi
  return 1
}

_resolve_from_root() {
  local root_dir="${1:-}"
  local raw_value="${2:-}"
  local default_rel="${3:-}"

  if [[ -z "${raw_value}" ]]; then
    raw_value="${default_rel}"
  fi

  case "${raw_value}" in
    /*) printf "%s\n" "${raw_value}" ;;
    ./*) printf "%s\n" "${root_dir}/${raw_value#./}" ;;
    *) printf "%s\n" "${root_dir}/${raw_value}" ;;
  esac
}

_load_member_env_once() {
  local root_dir="${1:-}"
  if [[ -z "${root_dir}" ]]; then
    return 1
  fi
  if [[ "${_MEMBER_ENV_LOADED:-0}" == "1" ]]; then
    return 0
  fi

  local env_raw="${MEMBER_ENV_FILE:-${root_dir}/.env}"
  local env_path=""
  case "${env_raw}" in
    /*) env_path="${env_raw}" ;;
    ./*) env_path="${root_dir}/${env_raw#./}" ;;
    *) env_path="${root_dir}/${env_raw}" ;;
  esac

  export MEMBER_ENV_FILE="${env_path}"
  if [[ -f "${env_path}" ]]; then
    set -a
    source "${env_path}"
    set +a
  fi
  export _MEMBER_ENV_LOADED=1
}

init_member_workspace_paths() {
  local project_root="${1:-}"
  local member_id="${2:-}"
  local raw_root="${ROOT_DIR:-}"
  local raw_config_dir="${CONFIG_DIR:-}"

  if [[ -z "${project_root}" || -z "${member_id}" ]]; then
    return 1
  fi

  if [[ -z "${raw_root}" ]]; then
    raw_root="members/${member_id}"
  fi

  local resolved_root=""
  resolved_root="$(_resolve_from_root "${project_root}" "${raw_root}" "members/${member_id}")"
  ROOT_DIR="${resolved_root}"
  export ROOT_DIR

  _load_member_env_once "${ROOT_DIR}" || return 1
  ROOT_DIR="${resolved_root}"
  export ROOT_DIR

  RUNS_DIR="$(_resolve_from_root "${ROOT_DIR}" "${RUNS_DIR:-}" "outputs")"
  local raw_log_dir="${LOG_DIR:-logs}"
  if [[ -z "${raw_log_dir}" ]]; then
    LOG_DIR="${RUNS_DIR}/logs"
  elif [[ "${raw_log_dir}" = /* ]]; then
    LOG_DIR="${raw_log_dir}"
  else
    LOG_DIR="${RUNS_DIR}/${raw_log_dir}"
  fi
  CONFIG_DIR="$(_resolve_from_root "${ROOT_DIR}" "${raw_config_dir}" "configs")"
  DOCS_DIR="$(_resolve_from_root "${ROOT_DIR}" "${DOCS_DIR:-}" "docs")"
  DATA_DIR="$(_resolve_from_root "${ROOT_DIR}" "${DATA_DIR:-}" "data")"
  CACHE_DIR="$(_resolve_from_root "${ROOT_DIR}" "${CACHE_DIR:-}" "cache")"

  HF_HOME="$(_resolve_from_root "${ROOT_DIR}" "${HF_HOME:-}" "${CACHE_DIR}/huggingface")"
  HF_HUB_CACHE="$(_resolve_from_root "${ROOT_DIR}" "${HF_HUB_CACHE:-}" "${HF_HOME}/hub")"
  if [[ -z "${TRANSFORMERS_CACHE:-}" ]]; then
    TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
  else
    TRANSFORMERS_CACHE="$(_resolve_from_root "${ROOT_DIR}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}")"
  fi
  HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

  if [[ -n "${TORCH_HOME:-}" ]]; then
    TORCH_HOME="$(_resolve_from_root "${ROOT_DIR}" "${TORCH_HOME}" "cache/torch")"
  fi
  if [[ -n "${TMPDIR:-}" ]]; then
    TMPDIR="$(_resolve_from_root "${ROOT_DIR}" "${TMPDIR}" "tmp")"
  fi

  if [[ -z "${WANDB_CACHE_DIR:-}" ]]; then
    WANDB_CACHE_DIR="${CACHE_DIR}/wandb"
  else
    WANDB_CACHE_DIR="$(_resolve_from_root "${ROOT_DIR}" "${WANDB_CACHE_DIR}" "cache/wandb")"
  fi
  export RUNS_DIR LOG_DIR CONFIG_DIR DOCS_DIR DATA_DIR CACHE_DIR WANDB_CACHE_DIR
  export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
  export TORCH_HOME TMPDIR
  export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

resolve_python_bin() {
  local requested="${1:-}"
  if [[ -n "$requested" ]]; then
    echo "$requested"
    return 0
  fi

  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi

  return 1
}

load_dotenv() {
  local env_file="${1:-.env}"
  if [[ ! -f "$env_file" ]]; then
    return 0
  fi

  set -a
  source "$env_file"
  set +a
}

print_subshell_activate_note() {
  local venv_dir="${1:-.venv}"
  echo "[NOTE] This script was executed in a subshell."
  echo "[NOTE] Activate the environment in your current shell with:"
  echo "source ${venv_dir}/bin/activate"
}
