from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_tune_summary(
    *,
    out_path: Path,
    sweep_dir: Path,
    family_id: str,
    cfg: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    tune_cfg = cfg.get("tune", {}) if isinstance(cfg.get("tune"), dict) else {}
    selection_cfg = (
        tune_cfg.get("selection", {})
        if isinstance(tune_cfg.get("selection"), dict)
        else {}
    )
    selector_cfg = (
        tune_cfg.get("selector", {})
        if isinstance(tune_cfg.get("selector"), dict)
        else {}
    )

    topk_n = int(selection_cfg.get("topk", 3) or 0)
    if topk_n < 0:
        topk_n = 0

    topk = [
        str(item.get("candidate_hash", ""))
        for item in ranked_candidates[:topk_n]
        if str(item.get("candidate_hash", ""))
    ]

    summary = {
        "schema_version": 1,
        "family_id": family_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep_dir),
        "objective": {
            "std_weight": float(selector_cfg.get("std_weight", 0.2)),
            "overfit_gap_threshold": float(
                selector_cfg.get("overfit_gap_threshold", 0.03)
            ),
            "overfit_weight": float(selector_cfg.get("overfit_weight", 1.0)),
            "fail_weight": float(selector_cfg.get("fail_weight", 0.5)),
        },
        "candidates": ranked_candidates,
        "topk": topk,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
