from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig  # pyright: ignore[reportMissingImports]

from experiment.context import RuntimeContext
from experiment.pipeline import run_pipeline
from experiment.tune.collector import collect_trial_rows, write_tune_results_csv
from experiment.tune.lineage import compute_family_id
from experiment.tune.report import write_tune_summary
from experiment.tune.selector import rank_candidates


def generate_tune_artifacts(
    *,
    output_root: Path,
    cfg: dict[str, Any],
    results_csv_name: str = "tune_results.csv",
    summary_json_name: str = "tune_summary.json",
) -> dict[str, Any]:
    rows = collect_trial_rows(output_root)

    results_path = output_root / results_csv_name
    write_tune_results_csv(rows, results_path)

    ranked = rank_candidates(rows, cfg)
    family_id = compute_family_id(cfg, sweep_dir_override=output_root)

    summary_path = output_root / summary_json_name
    summary = write_tune_summary(
        out_path=summary_path,
        sweep_dir=output_root,
        family_id=family_id,
        cfg=cfg,
        ranked_candidates=ranked,
    )
    return summary


def run_tune_pipeline(*, ctx: RuntimeContext, step: str, stop_after: str) -> None:
    run_pipeline(ctx=ctx, step=step, stop_after=stop_after)

    mode = "RUN"
    try:
        mode = str(HydraConfig.get().mode).upper()
    except Exception:
        cfg = ctx.cfg if isinstance(ctx.cfg, dict) else {}
        hydra_cfg = cfg.get("hydra", {}) if isinstance(cfg.get("hydra"), dict) else {}
        mode = str(hydra_cfg.get("mode", "RUN")).upper()

    # MULTIRUN aggregation is handled in Hydra callback (on_multirun_end).
    if "MULTIRUN" in mode:
        return

    cfg = ctx.cfg if isinstance(ctx.cfg, dict) else {}
    generate_tune_artifacts(output_root=ctx.run_dir, cfg=cfg)
