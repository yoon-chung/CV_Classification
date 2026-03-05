"""W&B logger implementation"""

import os
from pathlib import Path
from typing import Any, Optional, cast

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = cast(Any, None)

from experiment.contracts import WandbLogger


class RealWandbLogger(WandbLogger):
    """Real W&B logger using wandb SDK"""

    def __init__(self, project: Optional[str] = None, entity: Optional[str] = None):
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed")
        self.project = project or os.environ.get("WANDB_PROJECT", "cv-doc-class")
        self.entity = entity or os.environ.get("WANDB_ENTITY")
        self._run = None

    @staticmethod
    def _normalize_init_dir(raw_dir: Optional[str]) -> Optional[str]:
        if not raw_dir:
            return None
        path = Path(raw_dir)
        normalized = path.parent if path.name == "wandb" else path
        return str(normalized)

    def init(self, project: str, name: str, config: dict) -> None:
        if self._run is not None:
            return

        wandb_cfg = config.get("wandb", {}) if isinstance(config, dict) else {}
        cfg_project = wandb_cfg.get("project") if isinstance(wandb_cfg, dict) else None
        cfg_entity = wandb_cfg.get("entity") if isinstance(wandb_cfg, dict) else None
        cfg_group = wandb_cfg.get("group") if isinstance(wandb_cfg, dict) else None
        cfg_tags = wandb_cfg.get("tags") if isinstance(wandb_cfg, dict) else None
        cfg_mode = wandb_cfg.get("mode") if isinstance(wandb_cfg, dict) else None
        cfg_dir = wandb_cfg.get("dir") if isinstance(wandb_cfg, dict) else None
        init_dir = self._normalize_init_dir(
            str(cfg_dir) if cfg_dir is not None else None
        )
        tags = [str(tag) for tag in cfg_tags] if isinstance(cfg_tags, list) else None

        self._run = wandb.init(  # pyright: ignore[reportAttributeAccessIssue]
            project=project or cfg_project or self.project,
            entity=cfg_entity or self.entity,
            name=name,
            group=cfg_group,
            tags=tags,
            mode=cfg_mode,
            dir=init_dir,
            config=config,
            reinit=True,
        )

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self._run:
            wandb.log(metrics, step=step)  # pyright: ignore[reportAttributeAccessIssue]

    def finish(self) -> None:
        if self._run:
            wandb.finish()  # pyright: ignore[reportAttributeAccessIssue]
            self._run = None
