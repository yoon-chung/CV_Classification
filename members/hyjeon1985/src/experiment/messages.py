from __future__ import annotations

import re
from typing import Any


def safe_slack_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?<!\w)/(?:[^\s]+)", "<path>", text)
    return text


def format_slack_summary(
    *, run_name: str, run_id: str, profile: str, kind: str, summary_raw: dict[str, Any]
) -> str:
    train = (
        summary_raw.get("train", {})
        if isinstance(summary_raw.get("train"), dict)
        else {}
    )
    eval_ = (
        summary_raw.get("eval", {}) if isinstance(summary_raw.get("eval"), dict) else {}
    )

    parts: list[str] = [
        f"실험 완료 | run_name={run_name}",
        f"profile={profile}",
        f"kind={kind}",
        f"run_id={run_id}",
    ]

    for label, obj in [("train", train), ("eval", eval_)]:
        status = obj.get("status")
        if isinstance(status, str):
            parts.append(f"{label}.status={status}")

    candidate_keys = [
        ("eval", "macro_f1"),
        ("eval", "f1"),
        ("eval", "accuracy"),
        ("eval", "loss"),
        ("train", "loss"),
    ]
    for scope, k in candidate_keys:
        src = eval_ if scope == "eval" else train
        v = (
            src.get("metrics", src).get(k)
            if isinstance(src.get("metrics", src), dict)
            else None
        )
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{scope}.{k}={float(v):.4f}")

    return " | ".join(parts)


def format_slack_explore_summary(
    *,
    run_name: str,
    run_id: str,
    profile: str,
    kind: str,
    explore_summary: dict[str, Any],
) -> str:
    queue_id = explore_summary.get("queue_id")
    best_obj = explore_summary.get("best")
    best: dict[str, Any] = best_obj if isinstance(best_obj, dict) else {}

    best_item_name = best.get("item_name")
    best_macro_f1 = best.get("macro_f1")

    parts: list[str] = [
        f"Explore 완료 | run_name={run_name}",
        f"profile={profile}",
        f"kind={kind}",
        f"run_id={run_id}",
        f"queue_id={queue_id}",
        f"best.item_name={best_item_name}",
    ]
    if isinstance(best_macro_f1, (int, float)) and not isinstance(best_macro_f1, bool):
        parts.append(f"best.macro_f1={float(best_macro_f1):.4f}")
    return " | ".join(parts)
