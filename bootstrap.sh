#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL_RAW="${TORCH_INDEX_URL:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL_RAW:-https://download.pytorch.org/whl/cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.0}"
ARCH_REQUIREMENTS_FILE="${ARCH_REQUIREMENTS_FILE:-requirements.aarch64.txt}"
VERIFY_ARCH_OVERLAY="${VERIFY_ARCH_OVERLAY:-0}"
RUN_VERIFY=1
INSTALL_TORCH=1
ALLOW_NON310=0
USE_UV="1"
ARCH="unknown"

log_info() {
  echo "[INFO] $*"
}

log_warn() {
  echo "[WARN] $*" >&2
}

log_error() {
  echo "[ERROR] $*" >&2
}

verify_torch_cuda_build() {
  local py_bin="$1"
  "$py_bin" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"[ERROR] torch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"[INFO] torch_version={torch.__version__}")
print(f"[INFO] torch_cuda_version={torch.version.cuda}")
print(f"[INFO] torch_cuda_built={torch.backends.cuda.is_built()}")
print(f"[INFO] torch_cuda_available={torch.cuda.is_available()}")

if torch.version.cuda is None or not torch.backends.cuda.is_built():
    print(
        "[ERROR] CPU-only torch detected. Reinstall torch/torchvision from CUDA wheel index.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

verify_aarch64_overlay_versions() {
  local py_bin="$1"
  local req_file="$2"
  "$py_bin" - "$req_file" <<'PY'
import re
import sys
from pathlib import Path

req_path = Path(sys.argv[1])
if not req_path.exists():
    print(f"[ERROR] Missing overlay file: {req_path}", file=sys.stderr)
    raise SystemExit(1)

expected = {}
for line in req_path.read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    m = re.match(r"^(torch|torchvision|torchaudio)==(.+)$", s)
    if m:
        expected[m.group(1)] = m.group(2)

for key in ("torch", "torchvision", "torchaudio"):
    if key not in expected:
        print(f"[ERROR] {req_path} must pin {key}==<version>", file=sys.stderr)
        raise SystemExit(1)

import torch
import torchvision
try:
    import torchaudio
except Exception as exc:
    print(f"[ERROR] torchaudio import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

installed = {
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__.split("+")[0],
    "torchaudio": torchaudio.__version__.split("+")[0],
}

for pkg, want in expected.items():
    got = installed[pkg]
    if got != want:
        print(f"[ERROR] {pkg} version mismatch: expected {want}, got {got}", file=sys.stderr)
        raise SystemExit(1)

print("[INFO] aarch64 overlay versions verified")
for pkg in ("torch", "torchvision", "torchaudio"):
    print(f"[INFO] {pkg}={installed[pkg]}")
PY
}

print_subshell_activate_note() {
  local venv_dir="${1:-.venv}"
  echo "[NOTE] This script was executed in a subshell."
  echo "[NOTE] Activate the environment in your current shell with:"
  echo "source ${venv_dir}/bin/activate"
}

usage() {
  cat <<'EOF'
Usage: ./bootstrap.sh [options]

Options:
  --python-bin <path>     Python executable to use (default: python3.10, then python3)
  --venv-dir <path>       Virtualenv directory (default: .venv)
  --use-uv                Use uv for Python/venv/pip workflow (default)
  --no-uv                 Disable uv and use stdlib venv/pip
  --torch-index-url <url> Torch wheel index URL (default: cu118 index)
  --torch-version <ver>   Torch version (default: 2.1.0)
  --torchvision-version <ver> Torchvision version (default: 0.16.0)
  --arch-req-file <path>   aarch64-specific requirements file (default: requirements.aarch64.txt)
  --skip-torch            Skip torch/torchvision install step
  --force-torch           Force torch install even on unsupported arch/index combos
  --allow-non310          Allow Python other than 3.10 (not competition-parity)
  --no-verify             Skip environment verification
  -h, --help              Show this help

Examples:
  ./bootstrap.sh
  ./bootstrap.sh --python-bin python3.10
  ./bootstrap.sh
  ./bootstrap.sh --venv-dir .venv-server --no-verify
  ./bootstrap.sh --allow-non310 --skip-torch
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./bootstrap.sh
EOF
}

FORCE_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --use-uv)
      USE_UV="1"
      shift
      ;;
    --no-uv)
      USE_UV="0"
      shift
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torchvision-version)
      TORCHVISION_VERSION="$2"
      shift 2
      ;;
    --arch-req-file)
      ARCH_REQUIREMENTS_FILE="$2"
      shift 2
      ;;
    --skip-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --force-torch)
      FORCE_TORCH=1
      shift
      ;;
    --allow-non310)
      ALLOW_NON310=1
      shift
      ;;
    --no-verify)
      RUN_VERIFY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "requirements.txt" ]]; then
  echo "[ERROR] Run this script from team-repo root directory." >&2
  exit 1
fi
ARCH="$(uname -m 2>/dev/null || echo unknown)"

if [[ "$ARCH" == "aarch64" ]] && [[ -z "$TORCH_INDEX_URL_RAW" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
  log_info "aarch64 detected: default TORCH_INDEX_URL -> $TORCH_INDEX_URL"
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  log_info "uv not found. Installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    log_error "uv install requires curl or wget."
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    log_error "uv installation completed but uv is not in PATH."
    log_error "Try: export PATH=\"$HOME/.local/bin:$PATH\""
    exit 1
  fi
}

patch_venv_activate() {
  local activate_file="$VENV_DIR/bin/activate"
  if [[ ! -f "$activate_file" ]]; then
    log_warn "activate script not found: $activate_file"
    return 0
  fi

  "$VENV_PY" - "$activate_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)

lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
lines = [ln for ln in lines if "CVDC_BOOTSTRAP_PYTHONPATH" not in ln]

block = [
    "\n",
    "if [ -z \"${CVDC_BOOTSTRAP_PYTHONPATH:-}\" ]; then\n",
    "  export CVDC_BOOTSTRAP_PYTHONPATH=1\n",
    "  _CVDC_BOOTSTRAP_ROOT=\"\"\n",
    "  if command -v git >/dev/null 2>&1; then\n",
    "    _CVDC_BOOTSTRAP_ROOT=\"$(git -C \"$(dirname \"$VIRTUAL_ENV\")\" rev-parse --show-toplevel 2>/dev/null || true)\"\n",
    "  fi\n",
    "  if [ -z \"$_CVDC_BOOTSTRAP_ROOT\" ]; then\n",
    "    _CVDC_BOOTSTRAP_ROOT=\"$(dirname \"$VIRTUAL_ENV\")\"\n",
    "  fi\n",
    "  export PYTHONPATH=\"$_CVDC_BOOTSTRAP_ROOT${PYTHONPATH:+:$PYTHONPATH}\"\n",
    "  unset _CVDC_BOOTSTRAP_ROOT\n",
    "fi\n",
]

path.write_text("".join(lines + block), encoding="utf-8")
PY
}

MEMBER_ID="${MEMBER_ID:-}"
if [[ -n "$MEMBER_ID" ]] && [[ -d "members/$MEMBER_ID" ]]; then
  MEMBER_ENV_TEMPLATE="members/$MEMBER_ID/.env.template"
  MEMBER_ENV_FILE="members/$MEMBER_ID/.env"
  if [[ ! -f "$MEMBER_ENV_FILE" ]] && [[ -f "$MEMBER_ENV_TEMPLATE" ]]; then
    cp "$MEMBER_ENV_TEMPLATE" "$MEMBER_ENV_FILE"
    log_info "Created $MEMBER_ENV_FILE from $MEMBER_ENV_TEMPLATE"
  fi
else
  log_info "MEMBER_ID not set. Skip personal .env creation."
  log_info "Use: MEMBER_ID=<github_id> bash bootstrap.sh"
fi

if [[ "$USE_UV" == "1" ]]; then
  ensure_uv
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif [[ "$USE_UV" != "0" ]]; then
    uv python install 3.10
    PYTHON_BIN="$(uv python find 3.10)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    echo "[WARN] python3.10 not found. Falling back to python3." >&2
  else
    echo "[ERROR] python executable not found." >&2
    exit 1
  fi
fi

if [[ "$USE_UV" == "1" ]] && ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] --use-uv was specified but uv is not installed." >&2
  exit 1
fi

echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Venv: $VENV_DIR"
echo "[INFO] uv mode: $USE_UV"
echo "[INFO] Architecture: $ARCH"

PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "[INFO] Python version: $PY_VER"

if [[ "$PY_VER" != "3.10" ]] && [[ "$ALLOW_NON310" != "1" ]]; then
  if [[ "$USE_UV" == "1" ]]; then
    uv python install 3.10
    PYTHON_BIN="$(uv python find 3.10)"
    PY_VER="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    echo "[INFO] Switched Python via uv: $PYTHON_BIN ($PY_VER)"
  fi
fi

if [[ "$PY_VER" != "3.10" ]] && [[ "$ALLOW_NON310" != "1" ]]; then
  echo "[ERROR] This project targets Python 3.10 for competition parity." >&2
  echo "[ERROR] Detected Python $PY_VER." >&2
  echo "[ERROR] Use --python-bin python3.10 or rerun with --allow-non310." >&2
  exit 1
fi

if [[ "$PY_VER" != "3.10" ]] && [[ "$INSTALL_TORCH" == "1" ]]; then
  echo "[WARN] Python $PY_VER may not have wheel support for torch==$TORCH_VERSION at $TORCH_INDEX_URL." >&2
  echo "[WARN] If install fails, rerun with --skip-torch and install compatible torch manually." >&2
fi

if [[ "$ARCH" == "aarch64" ]] && [[ "$FORCE_TORCH" != "1" ]]; then
  echo "[INFO] aarch64 detected: skipping torch/torchvision direct install step." >&2
  echo "[INFO] Will require arch-specific overlay: $ARCH_REQUIREMENTS_FILE" >&2
  INSTALL_TORCH=0
fi

if [[ "$USE_UV" == "1" ]]; then
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
else
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [[ "$USE_UV" == "1" ]]; then
  uv pip install --python "$VENV_PY" --upgrade pip setuptools wheel
else
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
fi

if [[ "$INSTALL_TORCH" == "1" ]]; then
  echo "[INFO] Installing torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION from: $TORCH_INDEX_URL"
  if [[ "$USE_UV" == "1" ]]; then
    uv pip install --python "$VENV_PY" --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  else
    "$VENV_PIP" install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  fi
else
  echo "[INFO] Skipping torch install (--skip-torch)"
fi

if [[ "$ARCH" == "aarch64" ]] && [[ "$FORCE_TORCH" != "1" ]]; then
  if [[ ! -f "$ARCH_REQUIREMENTS_FILE" ]]; then
    echo "[ERROR] Missing required aarch64 overlay file: $ARCH_REQUIREMENTS_FILE" >&2
    echo "[ERROR] Create this local file (gitignored) with compatible torch/torchvision pins." >&2
    exit 1
  fi

  echo "[INFO] Installing aarch64 overlay requirements: $ARCH_REQUIREMENTS_FILE"
  if [[ "$USE_UV" == "1" ]]; then
    uv pip install --python "$VENV_PY" --index-url "$TORCH_INDEX_URL" -r "$ARCH_REQUIREMENTS_FILE"
  else
    "$VENV_PIP" install --index-url "$TORCH_INDEX_URL" -r "$ARCH_REQUIREMENTS_FILE"
  fi
fi

TMP_REQ="$VENV_DIR/.requirements.no_torch.txt"
export BOOTSTRAP_VENV_DIR="$VENV_DIR"
"$VENV_PY" - <<'PY'
from pathlib import Path
import os

src = Path("requirements.txt")
dst = Path(os.environ["BOOTSTRAP_VENV_DIR"]) / ".requirements.no_torch.txt"

lines = src.read_text(encoding="utf-8").splitlines()
filtered = []
for line in lines:
    s = line.strip()
    if s.startswith("torch==") or s.startswith("torchvision=="):
        continue
    filtered.append(line)

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(filtered) + "\n", encoding="utf-8")
PY

echo "[INFO] Installing remaining requirements"
if [[ "$USE_UV" == "1" ]]; then
  uv pip install --python "$VENV_PY" -r "$TMP_REQ"
else
  "$VENV_PIP" install -r "$TMP_REQ"
fi

if [[ "$INSTALL_TORCH" == "1" ]]; then
  echo "[INFO] Verifying torch CUDA build"
  verify_torch_cuda_build "$VENV_PY"
fi

if [[ "$VERIFY_ARCH_OVERLAY" == "1" ]] && [[ -f "$ARCH_REQUIREMENTS_FILE" ]]; then
  echo "[INFO] Verifying overlay versions from: $ARCH_REQUIREMENTS_FILE"
  verify_aarch64_overlay_versions "$VENV_PY" "$ARCH_REQUIREMENTS_FILE"
fi

patch_venv_activate

if [[ "$RUN_VERIFY" == "1" ]]; then
  if [[ -n "${MEMBER_ID}" ]] && [[ -f "members/${MEMBER_ID}/scripts/verify_env.sh" ]]; then
    echo "[INFO] Running environment check (non-strict by default)"
    VERIFY_STRICT="${VERIFY_STRICT:-0}" PYTHON_BIN="$VENV_PY" bash "members/${MEMBER_ID}/scripts/verify_env.sh"
  elif [[ -z "${MEMBER_ID}" ]]; then
    echo "[NOTE] MEMBER_ID not set. Skipping environment verification."
    echo "[NOTE] Set MEMBER_ID=<github_id> to run member-specific verification."
  fi
fi

echo "[DONE] Bootstrap completed."
echo "[NEXT] Activate environment: source $VENV_DIR/bin/activate"
echo "[NEXT] After activation, PYTHONPATH is set to repo root."

if [[ "${BASH_SOURCE[0]}" == "$0" ]] && type print_subshell_activate_note >/dev/null 2>&1; then
  print_subshell_activate_note "$VENV_DIR"
fi
