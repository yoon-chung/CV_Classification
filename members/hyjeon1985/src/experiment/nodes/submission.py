from __future__ import annotations

from pathlib import Path

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]
from experiment.pipeline import register_node  # pyright: ignore[reportMissingImports]
from experiment.solve.submission_writer import (
    build_submission_frame,
    load_predictions,
    write_submission_csv,
)

from .base import load_node_result, save_node_result


@register_node("submission")
def submission_node(ctx: RuntimeContext) -> None:
    infer = load_node_result(ctx, "infer")
    if infer is None:
        raise RuntimeError("infer.json is required before submission node")

    predictions_path = Path(str(infer.get("predictions_csv", "")))
    pred_df = load_predictions(predictions_path)
    submission_df = build_submission_frame(pred_df)
    submission_path = ctx.run_dir / "submission.csv"
    write_submission_csv(submission_df, submission_path)

    result = {
        "node": "submission",
        "status": "completed",
        "submission_csv": str(submission_path),
        "n_rows": int(len(submission_df)),
    }
    save_node_result(ctx, "submission", result)
