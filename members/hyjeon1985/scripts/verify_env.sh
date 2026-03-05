#!/usr/bin/env bash
set -euo pipefail

VERIFY_STRICT="${VERIFY_STRICT:-0}"

EXPECTED_PYTHON="3.10.13"
EXPECTED_PIP="23.2.1"
EXPECTED_UBUNTU="20.04.6"
EXPECTED_GPU_MIN_MEM_MIB="24000"
EXPECTED_TORCH="2.1.0"
EXPECTED_TORCH_CUDA="11.8"

PYTHON_BIN="${PYTHON_BIN:-}"

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

check_equals() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    echo "[OK] $name: $actual"
  else
    echo "[WARN] $name mismatch: actual='$actual' expected='$expected'"
    if [[ "$VERIFY_STRICT" == "1" ]]; then
      fail "$name mismatch"
    fi
  fi
}

check_ge() {
  local name="$1"
  local actual="$2"
  local expected_min="$3"
  if [[ "$actual" =~ ^[0-9]+$ ]] && [[ "$actual" -ge "$expected_min" ]]; then
    echo "[OK] $name: $actual (>= $expected_min)"
  else
    echo "[WARN] $name below minimum: actual='$actual' expected_min='$expected_min'"
    if [[ "$VERIFY_STRICT" == "1" ]]; then
      fail "$name below minimum"
    fi
  fi
}

echo "== Environment Rehearsal Check =="

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    fail "python executable not found"
  fi
fi

python_ver="$($PYTHON_BIN --version 2>&1 | awk '{print $2}')"
check_equals "python" "$python_ver" "$EXPECTED_PYTHON"

pip_ver="$($PYTHON_BIN -m pip --version | awk '{print $2}')"
check_equals "pip" "$pip_ver" "$EXPECTED_PIP"

ubuntu_ver="$($PYTHON_BIN - <<'PY'
import re
from pathlib import Path
text = Path('/etc/os-release').read_text(encoding='utf-8', errors='ignore')
m = re.search(r'VERSION="([0-9]+\.[0-9]+\.[0-9]+)', text)
print(m.group(1) if m else 'unknown')
PY
)"
check_equals "ubuntu" "$ubuntu_ver" "$EXPECTED_UBUNTU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "$VERIFY_STRICT" == "1" ]]; then
    fail "nvidia-smi not found"
  fi
  echo "[WARN] nvidia-smi not found"
else
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | awk '{print $1}')"
  check_ge "gpu_count" "$gpu_count" "1"
  gpu_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  check_ge "gpu_memory_mib" "$gpu_mem_mib" "$EXPECTED_GPU_MIN_MEM_MIB"
fi

if $PYTHON_BIN -c "import torch" >/dev/null 2>&1; then
  $PYTHON_BIN - <<'PY'
import torch
print(f"[INFO] torch: {torch.__version__}")
print(f"[INFO] torch_cuda_available: {torch.cuda.is_available()}")
print(f"[INFO] torch_cuda_runtime: {torch.version.cuda}")
print(f"[INFO] torch_device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"[INFO] torch_gpu_name: {torch.cuda.get_device_name(0)}")
PY

  torch_ver="$($PYTHON_BIN - <<'PY'
import torch
print(torch.__version__)
PY
)"
  check_equals "torch" "$torch_ver" "$EXPECTED_TORCH"

  torch_cuda_rt="$($PYTHON_BIN - <<'PY'
import torch
print(torch.version.cuda)
PY
)"
  check_equals "torch_cuda_runtime" "$torch_cuda_rt" "$EXPECTED_TORCH_CUDA"

  torch_cuda_available="$($PYTHON_BIN - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
  check_equals "torch_cuda_available" "$torch_cuda_available" "1"

  torch_device_count="$($PYTHON_BIN - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  check_ge "torch_device_count" "$torch_device_count" "1"
else
  echo "[WARN] torch import failed"
  if [[ "$VERIFY_STRICT" == "1" ]]; then
    fail "torch import failed"
  fi
fi

echo "[DONE] Environment check completed."
