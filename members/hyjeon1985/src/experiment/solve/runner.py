from __future__ import annotations

from experiment.context import RuntimeContext
from experiment.pipeline import run_pipeline


def run_solve_pipeline(*, ctx: RuntimeContext, step: str, stop_after: str) -> None:
    run_pipeline(ctx=ctx, step=step, stop_after=stop_after)
