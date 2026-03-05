from __future__ import annotations

from collections import defaultdict
from typing import Any


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return float(var**0.5)


def rank_candidates(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    tune_cfg = cfg.get("tune", {}) if isinstance(cfg.get("tune"), dict) else {}
    selector_cfg = (
        tune_cfg.get("selector", {})
        if isinstance(tune_cfg.get("selector"), dict)
        else {}
    )

    std_weight = float(selector_cfg.get("std_weight", 0.2))
    overfit_gap_threshold = float(selector_cfg.get("overfit_gap_threshold", 0.03))
    overfit_weight = float(selector_cfg.get("overfit_weight", 1.0))
    fail_weight = float(selector_cfg.get("fail_weight", 0.5))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate_hash = str(row.get("candidate_hash", ""))
        if candidate_hash:
            grouped[candidate_hash].append(row)

    ranked: list[dict[str, Any]] = []
    for candidate_hash, c_rows in grouped.items():
        success_rows = [r for r in c_rows if str(r.get("status")) == "success"]

        f1_values = [_to_float(r.get("macro_f1")) for r in success_rows]
        f1_values = [v for v in f1_values if v is not None]

        loss_values = [_to_float(r.get("val_loss")) for r in success_rows]
        loss_values = [v for v in loss_values if v is not None]

        gap_values = [_to_float(r.get("overfit_gap")) for r in success_rows]
        gap_values = [v for v in gap_values if v is not None]

        elapsed_values = [_to_float(r.get("elapsed_sec")) for r in success_rows]
        elapsed_values = [v for v in elapsed_values if v is not None]

        mean_f1 = _mean(f1_values)
        std_f1 = _std(f1_values)
        mean_val_loss = _mean(loss_values)
        mean_overfit_gap = _mean(gap_values)
        mean_elapsed_sec = _mean(elapsed_values)

        n_total = len(c_rows)
        n_success = len(success_rows)
        n_fail = n_total - n_success
        fail_rate = float(n_fail / n_total) if n_total > 0 else 1.0

        if mean_f1 is None:
            score = -1.0
        else:
            overfit_penalty = (
                max(0.0, float((mean_overfit_gap or 0.0) - overfit_gap_threshold))
                * overfit_weight
            )
            fail_penalty = fail_rate * fail_weight
            score = float(mean_f1 - std_weight * std_f1 - overfit_penalty - fail_penalty)

        folds = sorted(
            {
                int(v)
                for v in (_to_float(r.get("fold_index")) for r in c_rows)
                if v is not None
            }
        )
        seeds = sorted(
            {
                int(v)
                for v in (_to_float(r.get("seed")) for r in c_rows)
                if v is not None
            }
        )

        best_trial_dir = ""
        if success_rows:
            sorted_success = sorted(
                success_rows,
                key=lambda r: _to_float(r.get("macro_f1")) or float("-inf"),
                reverse=True,
            )
            best_trial_dir = str(sorted_success[0].get("trial_dir", ""))

        ranked.append(
            {
                "candidate_hash": candidate_hash,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_val_loss": mean_val_loss,
                "mean_overfit_gap": mean_overfit_gap,
                "mean_elapsed_sec": mean_elapsed_sec,
                "n_total": n_total,
                "n_success": n_success,
                "n_fail": n_fail,
                "fail_rate": fail_rate,
                "score": score,
                "folds": folds,
                "seeds": seeds,
                "best_trial_dir": best_trial_dir,
            }
        )

    ranked.sort(
        key=lambda r: (
            -float(r.get("score") if isinstance(r.get("score"), (int, float)) else -1.0),
            float(r.get("mean_val_loss") if isinstance(r.get("mean_val_loss"), (int, float)) else float("inf")),
            float(r.get("mean_elapsed_sec") if isinstance(r.get("mean_elapsed_sec"), (int, float)) else float("inf")),
            str(r.get("candidate_hash", "")),
        )
    )

    return ranked
