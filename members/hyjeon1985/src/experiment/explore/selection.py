from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from experiment.explore.planner import ExploreItem


@dataclass
class StageScore:
    item: ExploreItem
    score: float
    n_folds: int


def apply_pruning(
    *,
    stage_cfg: dict[str, Any],
    scores: list[StageScore],
    stage_records: list[dict[str, Any]],
) -> list[StageScore]:
    pruning_cfg = stage_cfg.get("pruning", {})
    if not isinstance(pruning_cfg, dict):
        return scores

    enabled = bool(pruning_cfg.get("enabled", False))
    min_value = pruning_cfg.get("min_value")
    if not enabled or min_value is None:
        return scores

    out = list(scores)

    threshold = float(min_value)
    out = [s for s in out if s.score >= threshold]

    best_margin = pruning_cfg.get("best_margin")
    if best_margin is not None and out:
        best_score = max(s.score for s in out)
        margin = float(best_margin)
        out = [s for s in out if s.score >= (best_score - margin)]

    stats = aggregate_item_stats(stage_records=stage_records)

    max_fold_std = pruning_cfg.get("max_fold_std")
    if max_fold_std is not None:
        std_limit = float(max_fold_std)
        out = [
            s
            for s in out
            if float(stats.get(s.item.name, {}).get("fold_std", 0.0)) <= std_limit
        ]

    max_overfit_gap = pruning_cfg.get("max_overfit_gap")
    if max_overfit_gap is not None:
        gap_limit = float(max_overfit_gap)
        out = [
            s
            for s in out
            if (
                stats.get(s.item.name, {}).get("max_overfit_gap") is None
                or float(stats[s.item.name]["max_overfit_gap"]) <= gap_limit
            )
        ]

    max_val_loss = pruning_cfg.get("max_val_loss")
    if max_val_loss is not None:
        loss_limit = float(max_val_loss)
        out = [
            s
            for s in out
            if (
                stats.get(s.item.name, {}).get("mean_val_loss") is None
                or float(stats[s.item.name]["mean_val_loss"]) <= loss_limit
            )
        ]

    return out


def aggregate_item_stats(stage_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_item: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"macro_f1": [], "val_loss": [], "overfit_gap": []}
    )

    for rec in stage_records:
        item_name = rec.get("item_name")
        if not isinstance(item_name, str):
            continue

        macro_f1 = rec.get("macro_f1")
        if isinstance(macro_f1, (int, float)):
            by_item[item_name]["macro_f1"].append(float(macro_f1))

        val_loss = rec.get("val_loss")
        if isinstance(val_loss, (int, float)):
            by_item[item_name]["val_loss"].append(float(val_loss))

        overfit_gap = rec.get("overfit_gap")
        if isinstance(overfit_gap, (int, float)):
            by_item[item_name]["overfit_gap"].append(float(overfit_gap))

    out: dict[str, dict[str, Any]] = {}
    for item_name, vals in by_item.items():
        f1s = vals["macro_f1"]
        losses = vals["val_loss"]
        gaps = vals["overfit_gap"]

        fold_std = 0.0
        if len(f1s) > 1:
            mean = sum(f1s) / len(f1s)
            var = sum((x - mean) ** 2 for x in f1s) / len(f1s)
            fold_std = math.sqrt(var)

        out[item_name] = {
            "fold_std": float(fold_std),
            "mean_val_loss": float(sum(losses) / len(losses)) if losses else None,
            "max_overfit_gap": max(gaps) if gaps else None,
        }
    return out


def select_topk(*, scores: list[StageScore], topk: int) -> list[StageScore]:
    if topk <= 0:
        return scores
    return scores[:topk]
