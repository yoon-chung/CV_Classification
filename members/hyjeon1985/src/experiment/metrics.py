from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_numeric_metrics(value: Any, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}{k}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[key] = float(v)
            elif isinstance(v, bool):
                out[key] = 1.0 if v else 0.0
            elif isinstance(v, dict):
                out.update(flatten_numeric_metrics(v, prefix=f"{key}/"))
        return out

    return out


def status_code(status: Any) -> int:
    if not isinstance(status, str):
        return -1
    s = status.lower()
    if s in {"completed", "success", "ok"}:
        return 1
    if s in {"failed", "error", "exception"}:
        return 0
    return -1


def _first_numeric(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def extract_summary(run_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    train = read_json(run_dir / "train.json") or {}
    eval_ = read_json(run_dir / "eval.json") or {}

    metrics: dict[str, float] = {}
    metrics["status/train"] = float(status_code(train.get("status")))
    metrics["status/eval"] = float(status_code(eval_.get("status")))

    train_metrics = train.get("metrics", train)
    eval_metrics = eval_.get("metrics", eval_)
    metrics.update(flatten_numeric_metrics(train_metrics, prefix="train/"))
    metrics.update(flatten_numeric_metrics(eval_metrics, prefix="eval/"))

    raw = {"train": train, "eval": eval_}
    return metrics, raw


def extract_tune_summary(run_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    train = read_json(run_dir / "train.json") or {}
    eval_ = read_json(run_dir / "eval.json") or {}
    prep = read_json(run_dir / "prep.json") or {}

    metrics: dict[str, float] = {}
    metrics["tune/status/train"] = float(status_code(train.get("status")))
    metrics["tune/status/eval"] = float(status_code(eval_.get("status")))

    macro_f1 = _first_numeric(eval_, ["macro_f1", "val/macro_f1"])
    val_loss = _first_numeric(eval_, ["val/loss"])
    best_val_macro_f1 = _first_numeric(train, ["best_val_macro_f1"])
    overfit_gap = _first_numeric(eval_, ["selection/overfit_gap"])
    if (
        overfit_gap is None
        and isinstance(best_val_macro_f1, float)
        and isinstance(macro_f1, float)
    ):
        overfit_gap = float(best_val_macro_f1 - macro_f1)

    if isinstance(macro_f1, float):
        metrics["tune/val_macro_f1"] = macro_f1
    if isinstance(val_loss, float):
        metrics["tune/val_loss"] = val_loss
    if isinstance(best_val_macro_f1, float):
        metrics["tune/best_val_macro_f1"] = best_val_macro_f1
    if isinstance(overfit_gap, float):
        metrics["tune/overfit_gap"] = overfit_gap

    for src_key, metric_key in [
        ("best_epoch", "tune/best_epoch"),
        ("start_epoch", "tune/start_epoch"),
        ("stop_epoch", "tune/stop_epoch"),
        ("epochs", "tune/epochs_planned"),
        ("train_loss_last", "tune/train_loss_last"),
        ("elapsed_sec", "tune/train_elapsed_sec"),
    ]:
        value = train.get(src_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric_key] = float(value)

    if isinstance(train.get("early_stopped"), bool):
        metrics["tune/early_stopped"] = 1.0 if train["early_stopped"] else 0.0
    if isinstance(train.get("resumed"), bool):
        metrics["tune/resumed"] = 1.0 if train["resumed"] else 0.0

    stop_epoch = train.get("stop_epoch")
    epochs_planned = train.get("epochs")
    if (
        isinstance(stop_epoch, (int, float))
        and isinstance(epochs_planned, (int, float))
        and float(epochs_planned) > 0.0
    ):
        metrics["tune/epoch_utilization"] = float(stop_epoch) / float(epochs_planned)

    split_obj = prep.get("split") if isinstance(prep.get("split"), dict) else {}
    fold_index = split_obj.get("fold_index") if isinstance(split_obj, dict) else None
    if isinstance(fold_index, (int, float)) and not isinstance(fold_index, bool):
        metrics["tune/fold_index"] = float(fold_index)

    raw = {"train": train, "eval": eval_, "prep": prep}
    return metrics, raw


def extract_selection_summary(run_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    train = read_json(run_dir / "train.json") or {}
    eval_ = read_json(run_dir / "eval.json") or {}

    metrics: dict[str, float] = {
        "status/train": float(status_code(train.get("status"))),
        "status/eval": float(status_code(eval_.get("status"))),
    }

    if isinstance(eval_.get("macro_f1"), (int, float)):
        metrics["selection/val_macro_f1"] = float(eval_["macro_f1"])
    elif isinstance(eval_.get("val/macro_f1"), (int, float)):
        metrics["selection/val_macro_f1"] = float(eval_["val/macro_f1"])

    if isinstance(eval_.get("val/loss"), (int, float)):
        metrics["selection/val_loss"] = float(eval_["val/loss"])

    if isinstance(eval_.get("selection/overfit_gap"), (int, float)):
        metrics["selection/overfit_gap"] = float(eval_["selection/overfit_gap"])

    split_cfg = read_json(run_dir / "prep.json") or {}
    split_section = split_cfg.get("split", {}) if isinstance(split_cfg, dict) else {}
    if isinstance(split_section, dict) and isinstance(
        split_section.get("fold_index"), (int, float)
    ):
        metrics["selection/fold_index"] = float(split_section["fold_index"])

    raw = {"train": train, "eval": eval_}
    return metrics, raw


def extract_explore_summary(
    explore_summary: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    supported_keys = {
        "queue_id",
        "planned_items",
        "executed_children",
        "best",
        "stage_summaries",
        "parallelism",
    }
    unsupported_keys = sorted(
        key for key in explore_summary.keys() if key not in supported_keys
    )
    if unsupported_keys:
        warnings.warn(
            "Unsupported keys in explore_summary.json: " + ", ".join(unsupported_keys),
            RuntimeWarning,
            stacklevel=2,
        )

    planned_items = explore_summary.get("planned_items")
    executed_children = explore_summary.get("executed_children")
    best = (
        explore_summary.get("best")
        if isinstance(explore_summary.get("best"), dict)
        else None
    )
    best_macro_f1 = best.get("macro_f1") if isinstance(best, dict) else None

    if not isinstance(planned_items, (int, float)) or isinstance(planned_items, bool):
        raise TypeError(f"Invalid explore_summary.planned_items: {planned_items!r}")
    if not isinstance(executed_children, (int, float)) or isinstance(
        executed_children, bool
    ):
        raise TypeError(
            f"Invalid explore_summary.executed_children: {executed_children!r}"
        )
    if not isinstance(best_macro_f1, (int, float)) or isinstance(best_macro_f1, bool):
        raise TypeError(f"Invalid explore_summary.best.macro_f1: {best_macro_f1!r}")

    metrics: dict[str, float] = {
        "explore/planned_items": float(planned_items),
        "explore/executed_children": float(executed_children),
        "explore/best_macro_f1": float(best_macro_f1),
    }
    raw: dict[str, Any] = {"explore": explore_summary}
    return metrics, raw
