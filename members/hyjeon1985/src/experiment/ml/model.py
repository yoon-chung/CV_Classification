from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import timm  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]


def create_timm_model(model_cfg: Mapping[str, Any], in_chans: int = 3) -> nn.Module:
    backbone = str(model_cfg.get("backbone", "resnet18"))
    pretrained = bool(model_cfg.get("pretrained", True))
    num_classes = int(model_cfg.get("num_classes", 17))
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
    )
