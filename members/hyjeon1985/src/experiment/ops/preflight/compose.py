from __future__ import annotations

from uuid import uuid4
import importlib
from pathlib import Path
from typing import Any


def _register_uuid_resolver(OmegaConf):
    """Register uuid resolver to handle ${uuid:} in configs."""
    try:
        OmegaConf.register_new_resolver("uuid", lambda: uuid4().hex, replace=True)
    except TypeError:
        try:
            OmegaConf.register_new_resolver("uuid", lambda: uuid4().hex)
        except ValueError:
            pass  # Already registered


def compose_config(
    *, config_root: Path, config_name: str, overrides: list[str]
) -> dict[str, Any]:
    try:
        hydra = importlib.import_module("hydra")
        hydra_global = importlib.import_module("hydra.core.global_hydra")
        omegaconf = importlib.import_module("omegaconf")
    except ModuleNotFoundError as e:
        raise RuntimeError("Hydra is not available") from e

    initialize_config_dir = getattr(hydra, "initialize_config_dir")
    compose = getattr(hydra, "compose")
    GlobalHydra = getattr(hydra_global, "GlobalHydra")
    OmegaConf = getattr(omegaconf, "OmegaConf")

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        cfg = compose(
            config_name=config_name, overrides=overrides, return_hydra_config=False
        )
        _register_uuid_resolver(OmegaConf)
        return OmegaConf.to_container(cfg, resolve=True)
