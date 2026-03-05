from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig  # pyright: ignore[reportMissingImports]

_DATE_RE = re.compile(r"date[=-](\d{4}-\d{2}-\d{2})")
_SWEEP_RE = re.compile(r"sweep_id[=-]([0-9\-]{6,8})")


def _normalize_hhmmss(value: str) -> str:
    clean = value.strip().replace("_", "-")
    if "-" in clean:
        parts = [p for p in clean.split("-") if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return f"{int(parts[0]):02d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    digits = "".join(ch for ch in clean if ch.isdigit())
    if len(digits) == 6:
        return f"{digits[:2]}-{digits[2:4]}-{digits[4:6]}"
    return clean


def _sanitize_family(value: str) -> str:
    text = value.strip()
    text = re.sub(r"[^a-zA-Z0-9._=-]+", "_", text)
    return text.strip("_") or "tune"


def compute_family_id(cfg: dict[str, Any], *, sweep_dir_override: Path | None = None) -> str:
    runtime_mode: str | None = None
    runtime_sweep_dir: str | None = None
    runtime_job_num: Any = None
    try:
        runtime = HydraConfig.get()
        runtime_mode = str(runtime.mode).upper()
        runtime_sweep_dir = str(runtime.sweep.dir)
        runtime_job_num = runtime.job.num
    except Exception:
        pass

    hydra_cfg = cfg.get("hydra", {}) if isinstance(cfg.get("hydra"), dict) else {}
    mode = runtime_mode or str(hydra_cfg.get("mode", "RUN")).upper()
    cfg_job_num: Any = None
    if isinstance(hydra_cfg.get("job"), dict):
        cfg_job_num = hydra_cfg.get("job", {}).get("num")
    cfg_sweep_dir = ""
    if isinstance(hydra_cfg.get("sweep"), dict):
        cfg_sweep_dir = str(hydra_cfg.get("sweep", {}).get("dir", ""))
    sweep_candidate = str(sweep_dir_override) if sweep_dir_override is not None else (runtime_sweep_dir or cfg_sweep_dir)
    has_concrete_sweep = bool(sweep_candidate and "${" not in sweep_candidate)
    has_concrete_job_num = isinstance(runtime_job_num, int) or isinstance(cfg_job_num, int)
    is_multirun_context = mode == "MULTIRUN" or has_concrete_job_num

    if is_multirun_context and has_concrete_sweep:
        sweep_dir = (
            str(sweep_dir_override)
            if sweep_dir_override is not None
            else runtime_sweep_dir
            or str(
                hydra_cfg.get("sweep", {}).get("dir", "")
                if isinstance(hydra_cfg.get("sweep"), dict)
                else ""
            )
        )
        date_match = _DATE_RE.search(sweep_dir)
        sweep_match = _SWEEP_RE.search(sweep_dir)
        if date_match and sweep_match:
            date_s = date_match.group(1)
            sweep_s = _normalize_hhmmss(sweep_match.group(1))
            return f"date={date_s}__sweep_id={sweep_s}"

        if sweep_dir_override is not None:
            return _sanitize_family(Path(sweep_dir_override).name)
        if sweep_dir:
            return _sanitize_family(Path(sweep_dir).name)
        return "tune_multirun"

    run_id = str(cfg.get("runner", {}).get("run_id", "")).strip()
    return run_id or "tune_run"


def apply_lineage(resolved_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = str(resolved_cfg.get("experiment", {}).get("kind", ""))
    if kind != "tune":
        return resolved_cfg

    tune_cfg = (
        resolved_cfg.get("tune", {}) if isinstance(resolved_cfg.get("tune"), dict) else {}
    )
    lineage_cfg = (
        tune_cfg.get("lineage", {})
        if isinstance(tune_cfg.get("lineage"), dict)
        else {}
    )
    enabled = bool(lineage_cfg.get("enabled", True))

    family_id = compute_family_id(resolved_cfg)

    lineage = resolved_cfg.get("lineage")
    if not isinstance(lineage, dict):
        lineage = {}
        resolved_cfg["lineage"] = lineage

    lineage["family_id"] = family_id
    lineage["kind"] = "tune"

    hydra_cfg = resolved_cfg.get("hydra", {})
    try:
        lineage["job_num"] = int(HydraConfig.get().job.num)
    except Exception:
        if isinstance(hydra_cfg, dict):
            job_cfg = hydra_cfg.get("job", {})
            if isinstance(job_cfg, dict) and "num" in job_cfg:
                lineage["job_num"] = job_cfg.get("num")

    if not enabled:
        return resolved_cfg

    wandb_cfg = resolved_cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}
        resolved_cfg["wandb"] = wandb_cfg

    wandb_cfg["group"] = f"tune__{family_id}"

    tags = wandb_cfg.get("tags")
    if isinstance(tags, list):
        tag_list = [str(t) for t in tags]
    else:
        tag_list = []

    required = [f"kind:tune", f"family:{family_id}"]
    for tag in required:
        if tag not in tag_list:
            tag_list.append(tag)
    wandb_cfg["tags"] = tag_list

    return resolved_cfg
