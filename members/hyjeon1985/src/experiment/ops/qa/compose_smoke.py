"""Compose smoke test - verify Hydra config loads successfully"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_compose_smoke(
    member_root: Path | None = None, python_bin: str | None = None
) -> int:
    if member_root is None:
        member_root = Path(__file__).resolve().parents[4]

    if python_bin is None:
        python_bin = sys.executable

    src_path = member_root / "src"

    cmd = [
        python_bin,
        "-m",
        "experiment",
        "--cfg",
        "job",
    ]

    env = dict(os.environ)
    env["ROOT_DIR"] = str(member_root)
    env["PYTHONPATH"] = str(src_path) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    try:
        result = subprocess.run(
            cmd,
            cwd=str(member_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("Compose smoke: PASSED")
            return 0

        print("Compose smoke: FAILED", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("Compose smoke: TIMEOUT", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Compose smoke: ERROR - {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_compose_smoke())
