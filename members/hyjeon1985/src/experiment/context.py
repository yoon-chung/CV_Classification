from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.contracts import Notifier, UploadBackend, WandbLogger
from experiment.spec import ExperimentSpec


@dataclass(frozen=True)
class RuntimeContext:
    run_dir: Path
    spec: ExperimentSpec
    upload_backend: UploadBackend
    wandb_logger: WandbLogger
    notifier: Notifier
    cfg: dict[str, Any]

    @classmethod
    def create(
        cls,
        run_dir: Path,
        spec: ExperimentSpec,
        integrations: dict[str, Any],
    ) -> "RuntimeContext":
        return cls(
            run_dir=run_dir,
            spec=spec,
            upload_backend=integrations["upload_backend"],
            wandb_logger=integrations["wandb_logger"],
            notifier=integrations["notifier"],
            cfg=integrations.get("cfg", {}),
        )
