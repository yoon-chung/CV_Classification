from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def read_macro_f1(eval_json_path: Path) -> float | None:
    payload = read_json(eval_json_path)
    if payload is None:
        return None

    if isinstance(payload.get("macro_f1"), (int, float)):
        return float(payload["macro_f1"])
    if isinstance(payload.get("val/macro_f1"), (int, float)):
        return float(payload["val/macro_f1"])

    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        if isinstance(metrics.get("macro_f1"), (int, float)):
            return float(metrics["macro_f1"])
        if isinstance(metrics.get("val/macro_f1"), (int, float)):
            return float(metrics["val/macro_f1"])

    return None


def read_val_loss(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    if isinstance(payload.get("val/loss"), (int, float)):
        return float(payload["val/loss"])
    if isinstance(payload.get("val_loss"), (int, float)):
        return float(payload["val_loss"])
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("val/loss"), (int, float)):
        return float(metrics["val/loss"])
    return None


def read_best_val_macro_f1(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    if isinstance(payload.get("best_val_macro_f1"), (int, float)):
        return float(payload["best_val_macro_f1"])
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(
        metrics.get("best_val_macro_f1"), (int, float)
    ):
        return float(metrics["best_val_macro_f1"])
    return None
