from __future__ import annotations

from pathlib import Path

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]
from experiment.pipeline import register_node  # pyright: ignore[reportMissingImports]

from .base import load_node_result, save_node_result


def _is_s3_enabled(ctx: RuntimeContext) -> bool:
    cfg = ctx.cfg if isinstance(ctx.cfg, dict) else {}
    return bool(cfg.get("upload", {}).get("s3", {}).get("enabled", False))


@register_node("upload")
def upload_node(ctx: RuntimeContext) -> None:
    submission = load_node_result(ctx, "submission")
    if submission is None:
        raise RuntimeError("submission.json is required before upload node")

    submission_path = Path(str(submission.get("submission_csv", "")))
    if not submission_path.exists():
        raise FileNotFoundError(f"submission.csv not found: {submission_path}")

    s3_enabled = _is_s3_enabled(ctx)
    upload_called = False
    uploaded = False
    remote_key = None

    if s3_enabled and ctx.upload_backend.is_available():
        upload_called = True
        run_id = "unknown"
        if isinstance(ctx.cfg, dict):
            run_id = str(ctx.cfg.get("runner", {}).get("run_id", "unknown"))
        remote_key = f"{run_id}/submission.csv"
        uploaded = bool(ctx.upload_backend.upload(submission_path, remote_key))

    result = {
        "node": "upload",
        "status": "completed",
        "s3_enabled": s3_enabled,
        "upload_called": upload_called,
        "uploaded": uploaded,
        "remote_key": remote_key,
    }
    save_node_result(ctx, "upload", result)
