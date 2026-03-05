from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback  # pyright: ignore[reportMissingImports]
from hydra.core.hydra_config import HydraConfig  # pyright: ignore[reportMissingImports]
from omegaconf import DictConfig, OmegaConf  # pyright: ignore[reportMissingImports]

from experiment.tune.runner import generate_tune_artifacts


def _is_unresolved_interpolation(value: str) -> bool:
    return "${" in value


def _resolve_sweep_dir(config: DictConfig) -> tuple[dict[str, Any], Path]:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved Hydra config must be a dict")

    sweep_dir_raw: str | None = None
    try:
        sweep_dir_raw = str(HydraConfig.get().sweep.dir)
    except Exception:
        hydra_cfg = resolved.get("hydra", {})
        sweep_cfg = hydra_cfg.get("sweep", {}) if isinstance(hydra_cfg, dict) else {}
        raw = sweep_cfg.get("dir") if isinstance(sweep_cfg, dict) else None
        if isinstance(raw, str):
            sweep_dir_raw = raw

    if not isinstance(sweep_dir_raw, str) or not sweep_dir_raw.strip():
        raise ValueError("hydra.sweep.dir is empty; cannot collect tune results")
    if _is_unresolved_interpolation(sweep_dir_raw):
        raise ValueError(
            f"hydra.sweep.dir is unresolved interpolation: {sweep_dir_raw}"
        )

    return resolved, Path(sweep_dir_raw)


class TuneReportCallback(Callback):
    def __init__(
        self,
        results_csv: str = "tune_results.csv",
        summary_json: str = "tune_summary.json",
    ) -> None:
        self.results_csv = results_csv
        self.summary_json = summary_json

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        try:
            resolved, sweep_dir = _resolve_sweep_dir(config)
            hydra_cfg = resolved.get("hydra")
            if not isinstance(hydra_cfg, dict):
                hydra_cfg = {}
                resolved["hydra"] = hydra_cfg
            hydra_cfg["mode"] = "MULTIRUN"
            sweep_cfg = hydra_cfg.get("sweep")
            if not isinstance(sweep_cfg, dict):
                sweep_cfg = {}
                hydra_cfg["sweep"] = sweep_cfg
            sweep_cfg["dir"] = str(sweep_dir)

            generate_tune_artifacts(
                output_root=sweep_dir,
                cfg=resolved,
                results_csv_name=self.results_csv,
                summary_json_name=self.summary_json,
            )
        except Exception as exc:
            error_dir = Path(".")
            try:
                sweep_dir_raw = str(HydraConfig.get().sweep.dir)
                if sweep_dir_raw.strip() and not _is_unresolved_interpolation(
                    sweep_dir_raw
                ):
                    error_dir = Path(sweep_dir_raw)
            except Exception:
                try:
                    resolved = OmegaConf.to_container(config, resolve=True)
                    if isinstance(resolved, dict):
                        hydra_cfg = resolved.get("hydra", {})
                        sweep_cfg = (
                            hydra_cfg.get("sweep", {})
                            if isinstance(hydra_cfg, dict)
                            else {}
                        )
                        raw = sweep_cfg.get("dir") if isinstance(sweep_cfg, dict) else None
                        if (
                            isinstance(raw, str)
                            and raw.strip()
                            and not _is_unresolved_interpolation(raw)
                        ):
                            error_dir = Path(raw)
                except Exception:
                    pass

            error_dir.mkdir(parents=True, exist_ok=True)
            error_path = error_dir / "tune_report_error.json"
            payload = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            error_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
