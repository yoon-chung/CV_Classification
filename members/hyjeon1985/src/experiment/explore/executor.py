from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from experiment.context import RuntimeContext
from experiment.explore.planner import ExploreItem
from experiment.ops.logger import get_logger


class ExploreExecutor:
    """Runs child experiment processes for explore orchestration."""

    def __init__(
        self,
        ctx: RuntimeContext | None = None,
        *,
        root_dir: Path | None = None,
        python_bin: str | None = None,
        queue_id: str | None = None,
        job_timeout_sec: int = 3600,
    ):
        self.ctx = ctx
        self.python_bin = python_bin or os.environ.get("PYTHON_BIN") or "python"
        env_root = os.environ.get("ROOT_DIR")
        if root_dir is not None:
            self.root_dir = root_dir
        elif env_root and Path(env_root).is_absolute():
            self.root_dir = Path(env_root)
        else:
            self.root_dir = Path(__file__).resolve().parents[3]
        self.queue_id = queue_id or "n/a"
        self.job_timeout_sec = int(job_timeout_sec)
        self.logger = get_logger(__name__)

    def run_child(
        self, *, overrides: list[str], child_run_dir: Path, assigned_gpu: str | None
    ) -> dict[str, Any]:
        cmd = [self.python_bin, "-m", "experiment"] + overrides
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.root_dir / "src")
        env["ROOT_DIR"] = str(self.root_dir)
        env["EXPERIMENT_CHILD"] = "1"

        if assigned_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        timeout_sec = int(self.job_timeout_sec)
        child_run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = child_run_dir / "child_stdout.txt"
        stderr_path = child_run_dir / "child_stderr.txt"
        assigned_gpu_text = env.get("CUDA_VISIBLE_DEVICES")
        started = time.perf_counter()
        timed_out = False
        returncode: int | None = None

        try:
            self.logger.info(
                "Child start | queue_id=%s child_run_dir=%s gpu=%s timeout_sec=%s",
                self.queue_id,
                child_run_dir,
                assigned_gpu_text,
                timeout_sec,
            )
            with (
                stdout_path.open("w", encoding="utf-8") as stdout_fp,
                stderr_path.open("w", encoding="utf-8") as stderr_fp,
            ):
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.root_dir),
                    env=env,
                    stdout=stdout_fp,
                    stderr=stderr_fp,
                    text=True,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._kill_process_group(process=process)
                    returncode = process.returncode
                    self.logger.warning(
                        "Child timeout | queue_id=%s child_run_dir=%s timeout_sec=%s",
                        self.queue_id,
                        child_run_dir,
                        timeout_sec,
                    )
        except Exception as exc:
            elapsed_sec = time.perf_counter() - started
            with stderr_path.open("a", encoding="utf-8") as stderr_fp:
                stderr_fp.write(f"\n[orchestrator_error] {type(exc).__name__}: {exc}\n")
            self.logger.exception(
                "Child launch failed | queue_id=%s child_run_dir=%s",
                self.queue_id,
                child_run_dir,
            )
            return {
                "assigned_gpu": assigned_gpu_text,
                "returncode": -1,
                "timed_out": False,
                "elapsed_sec": float(elapsed_sec),
            }

        elapsed_sec = time.perf_counter() - started
        self.logger.info(
            "Child end | queue_id=%s child_run_dir=%s returncode=%s timed_out=%s elapsed_sec=%.3f",
            self.queue_id,
            child_run_dir,
            returncode,
            timed_out,
            elapsed_sec,
        )
        return {
            "assigned_gpu": assigned_gpu_text,
            "returncode": int(returncode) if isinstance(returncode, int) else -1,
            "timed_out": timed_out,
            "elapsed_sec": float(elapsed_sec),
        }

    def execute_item(self, item: ExploreItem) -> bool:
        result = self.run_child(
            overrides=list(item.overrides)
            + [f"runner.run_id={item.name}", "pipeline.stop_after=eval"],
            child_run_dir=self.root_dir / "outputs" / "explore_executor" / item.name,
            assigned_gpu=None,
        )
        return bool(result.get("returncode") == 0 and not result.get("timed_out"))

    def execute_all(self, items: list[ExploreItem]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for item in items:
            results[item.name] = self.execute_item(item)
        return results

    @staticmethod
    def _kill_process_group(
        process: subprocess.Popen[str], grace_sec: int = 10
    ) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace_sec)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def create_executor(ctx: RuntimeContext) -> ExploreExecutor:
    return ExploreExecutor(ctx=ctx)
