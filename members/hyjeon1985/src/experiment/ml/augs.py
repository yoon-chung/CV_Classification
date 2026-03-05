from __future__ import annotations

from typing import Literal

import albumentations as A  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from albumentations.pytorch import ToTensorV2  # pyright: ignore[reportMissingImports]


MixMode = Literal["none", "mixup", "cutmix"]
AugPreset = Literal[
    "baseline",
    "explore_v1",
    "explore_quality_shift_v1",
    "aug_v2",
    "aug_v3",
    "quality_shift",
    "aug_lite",
    "aug_balanced",
    "aug_strong",
]


def validate_mix_label_smoothing(mix: str, label_smoothing: float) -> None:
    if mix != "none" and label_smoothing > 0.0:
        raise ValueError("augmentation.mix!=none requires train.label_smoothing=0.0")


def build_train_transform(
    preset: str | int, target_size: int | None = None
) -> A.Compose:
    if target_size is None:
        if isinstance(preset, int):
            target_size = preset
            preset = "baseline"
        else:
            raise TypeError("build_train_transform() requires (preset, target_size)")

    preset_name = str(preset)
    if preset_name in ["baseline", "aug_lite"]:
        extra: list[A.BasicTransform] = []
    elif preset_name in ["explore_v1", "aug_v2", "aug_balanced"]:
        extra = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=7),
            A.RandomBrightnessContrast(p=0.3),
        ]
    elif preset_name in [
        "explore_quality_shift_v1",
        "aug_v3",
        "quality_shift",
        "aug_strong",
    ]:
        extra = [
            A.ImageCompression(p=0.3, quality_lower=30),
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2),
        ]
    else:
        raise ValueError(f"Unsupported augmentation.preset: {preset_name}")

    return A.Compose(
        [
            A.Resize(target_size, target_size),
            *extra,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_valid_transform(target_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(target_size, target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _sample_lambda(alpha: float, device: torch.device) -> torch.Tensor:
    if alpha <= 0.0:
        return torch.tensor(1.0, device=device)
    dist = torch.distributions.Beta(alpha, alpha)
    return dist.sample().to(device=device)


def _apply_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    lam = float(_sample_lambda(alpha, images.device).item())
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def _apply_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    batch_size, _, height, width = images.shape
    index = torch.randperm(batch_size, device=images.device)
    lam = float(_sample_lambda(alpha, images.device).item())

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    cut_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1.0 - (cut_area / float(width * height))
    return mixed_images, targets, targets[index], lam_adjusted


def apply_batch_mix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mix: MixMode,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mix == "none":
        return images, targets, targets, 1.0
    if mix == "mixup":
        return _apply_mixup(images, targets, alpha)
    if mix == "cutmix":
        return _apply_cutmix(images, targets, alpha)
    raise ValueError(f"Unsupported augmentation.mix: {mix}")
