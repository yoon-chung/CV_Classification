from __future__ import annotations

import os
from pathlib import Path
import random
import time
from typing import Any, Callable, cast

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from PIL import Image
from sklearn.metrics import f1_score  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader, Dataset  # pyright: ignore[reportMissingImports]

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]
from experiment.ml.augs import (  # pyright: ignore[reportMissingImports]
    MixMode,
    apply_batch_mix,
    build_train_transform,
    build_valid_transform,
    validate_mix_label_smoothing,
)
from experiment.ops.logger import get_logger
from experiment.pipeline import register_node  # pyright: ignore[reportMissingImports]

from .base import load_node_result, save_node_result


def _get_cfg(ctx: RuntimeContext) -> dict:
    if isinstance(ctx.cfg, dict):
        return ctx.cfg
    raise TypeError("RuntimeContext.cfg must be a dictionary")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_grad_scaler(
    enabled: bool,
) -> torch.cuda.amp.GradScaler | torch.amp.GradScaler:
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return amp_mod.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast("cuda", enabled=enabled)
        except TypeError:
            return amp_mod.autocast(enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, bool]:
    scheduler_cfg = cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler"), dict) else {}
    name = str(scheduler_cfg.get("name", "none")).strip().lower()
    if name in {"", "none"}:
        return None, False

    warmup_epochs = max(0, int(scheduler_cfg.get("warmup_epochs", 0) or 0))
    main_epochs = max(1, epochs - warmup_epochs)
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step_per_batch = False

    if name == "cosine":
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=min_lr
        )
    elif name == "step":
        step_size = max(1, int(scheduler_cfg.get("step_size", 3)))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif name == "onecycle":
        max_lr = float(scheduler_cfg.get("max_lr", cfg.get("optimizer", {}).get("lr", 1e-3)))
        pct_start = (
            min(0.99, max(0.01, warmup_epochs / max(1, epochs)))
            if epochs > 0
            else 0.3
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=max(1, epochs),
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=pct_start,
        )
        return scheduler, True
    else:
        raise ValueError(f"Unsupported scheduler.name: {name}")

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, main],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main
    return scheduler, step_per_batch


def _load_split_frames(ctx: RuntimeContext) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    prep = load_node_result(ctx, "prep")
    if prep is None:
        raise RuntimeError("prep.json is required before train node")

    train_split_path = Path(str(prep.get("train_split_csv", "")))
    val_split_path = Path(str(prep.get("val_split_csv", "")))
    if not train_split_path.exists() or not val_split_path.exists():
        raise FileNotFoundError("prep split files are missing")

    train_df = pd.read_csv(train_split_path)
    val_df = pd.read_csv(val_split_path)
    return train_df, val_df, bool(prep.get("dummy_data", False))


def _resolve_image_path(image_dir: Path, image_id: str, exts: list[str]) -> Path:
    direct = image_dir / image_id
    if direct.exists():
        return direct

    stem = Path(image_id).stem
    for ext in exts:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for ID: {image_id}")


def _dummy_image_tensor(image_id: str, size: int) -> torch.Tensor:
    seed = abs(hash(image_id)) % (2**32)
    rng = np.random.default_rng(seed)
    arr = rng.random((size, size, 3), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _dummy_image_array(image_id: str, size: int) -> np.ndarray:
    seed = abs(hash(image_id)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _load_image_tensor(
    image_dir: Path, image_id: str, exts: list[str], size: int
) -> torch.Tensor:
    image_path = _resolve_image_path(image_dir, image_id, exts)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_image_array(image_dir: Path, image_id: str, exts: list[str]) -> np.ndarray:
    image_path = _resolve_image_path(image_dir, image_id, exts)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    return arr


class _ImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        image_exts: list[str],
        target_size: int,
        dummy_data: bool,
        label_to_idx: dict[int, int],
        transform: Callable[..., dict[str, Any]] | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_exts = image_exts
        self.target_size = target_size
        self.dummy_data = dummy_data
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_id = str(row["ID"])
        label = int(row["target"])
        if self.transform is not None:
            if self.dummy_data:
                image_arr = _dummy_image_array(image_id, self.target_size)
            else:
                image_arr = _load_image_array(self.image_dir, image_id, self.image_exts)
            out = self.transform(image=image_arr)
            image = out["image"]
        else:
            if self.dummy_data:
                image = _dummy_image_tensor(image_id, self.target_size)
            else:
                image = _load_image_tensor(
                    self.image_dir, image_id, self.image_exts, self.target_size
                )
        target_idx = self.label_to_idx[label]
        return image, torch.tensor(target_idx, dtype=torch.long)


class _TinyConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _resolve_backbone_name(cfg: dict, backbone_override: str | None = None) -> str:
    if backbone_override is not None:
        backbone = str(backbone_override).strip()
    else:
        model_cfg = cfg.get("model", {})
        backbone = str(model_cfg.get("backbone", "resnet18")).strip()
    return backbone or "resnet18"


def _build_model(
    cfg: dict, num_classes: int, backbone_override: str | None = None
) -> nn.Module:
    model_cfg = cfg.get("model", {})
    backbone = _resolve_backbone_name(cfg, backbone_override=backbone_override)
    pretrained = bool(model_cfg.get("pretrained", False))

    if backbone.lower() == "tinyconvnet":
        return _TinyConvNet(num_classes)

    try:
        import timm  # pyright: ignore[reportMissingImports]
    except Exception as e:
        raise RuntimeError(
            "timm is required to build the requested backbone "
            f"'{backbone}'. Install timm or set model.backbone=tinyconvnet."
        ) from e

    try:
        return timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )
    except Exception as e:
        raise RuntimeError(
            f"timm.create_model failed for backbone='{backbone}' (pretrained={pretrained})."
        ) from e


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_label: list[int],
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))

            pred_idx = torch.argmax(logits, dim=1).cpu().tolist()
            true_idx = targets.cpu().tolist()
            y_pred.extend(
                [
                    int(idx_to_label[i]) if i < len(idx_to_label) else int(i)
                    for i in pred_idx
                ]
            )
            y_true.extend(
                [
                    int(idx_to_label[i]) if i < len(idx_to_label) else int(i)
                    for i in true_idx
                ]
            )

    mean_loss = float(sum(losses) / max(1, len(losses)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return mean_loss, macro_f1


def _safe_wandb_log(
    ctx: RuntimeContext, metrics: dict, step: int | None = None
) -> None:
    try:
        ctx.wandb_logger.log(metrics, step=step)
    except Exception:
        pass


@register_node("train")
def train_node(ctx: RuntimeContext) -> None:
    train_started = time.perf_counter()
    logger = get_logger(__name__)
    cfg = _get_cfg(ctx)
    train_df, val_df, dummy_data = _load_split_frames(ctx)

    labels = sorted(
        {int(v) for v in pd.concat([train_df["target"], val_df["target"]]).tolist()}
    )
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = [int(label) for label in labels]
    num_classes = len(idx_to_label)

    dataset_cfg = cfg.get("dataset", {})
    image_dir = Path(str(dataset_cfg.get("image_dir_train", "")))
    image_exts = list(dataset_cfg.get("image_extensions", [".jpg", ".png", ".jpeg"]))
    target_size = int(cfg.get("preprocess", {}).get("target_size", 224))

    augmentation_cfg = cfg.get("augmentation", {})
    preset = str(augmentation_cfg.get("preset", "baseline"))
    mix_mode_raw = str(augmentation_cfg.get("mix", "none"))
    if mix_mode_raw not in {"none", "mixup", "cutmix"}:
        raise ValueError(f"Unsupported augmentation.mix: {mix_mode_raw}")
    mix_mode = cast(MixMode, mix_mode_raw)

    train_transform = build_train_transform(preset, target_size)
    valid_transform = build_valid_transform(target_size)

    train_dataset = _ImageDataset(
        frame=train_df,
        image_dir=image_dir,
        image_exts=image_exts,
        target_size=target_size,
        dummy_data=dummy_data,
        label_to_idx=label_to_idx,
        transform=train_transform,
    )
    val_dataset = _ImageDataset(
        frame=val_df,
        image_dir=image_dir,
        image_exts=image_exts,
        target_size=target_size,
        dummy_data=dummy_data,
        label_to_idx=label_to_idx,
        transform=valid_transform,
    )

    batch_size = int(cfg.get("train", {}).get("batch_size", 16))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    seed = int(cfg.get("train", {}).get("seed", 42))
    _seed_everything(seed)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=val_generator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_cuda = str(os.environ.get("CVDC_REQUIRE_CUDA", "0")).strip() == "1"
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CVDC_REQUIRE_CUDA=1 but torch.cuda.is_available() is false. Aborting to avoid CPU run."
        )
    model_backbone_requested = _resolve_backbone_name(cfg)
    model_backbone_used = (
        "tinyconvnet"
        if model_backbone_requested.lower() == "tinyconvnet"
        else model_backbone_requested
    )
    fallback_used = bool(model_backbone_used == "tinyconvnet")
    model = _build_model(cfg, num_classes, backbone_override=model_backbone_used).to(
        device
    )

    train_cfg = cfg.get("train", {})
    optimizer_cfg = cfg.get("optimizer", {})
    class_weight_mode = str(train_cfg.get("class_weight_mode", "none"))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    validate_mix_label_smoothing(mix_mode, label_smoothing)

    weight_tensor = None
    if class_weight_mode in {"balanced", "sqrt"}:
        counts = train_df["target"].value_counts().to_dict()
        weights = []
        total = float(len(train_df))
        n_classes = len(idx_to_label)
        for label in idx_to_label:
            c = float(counts.get(label, 1.0))
            w = total / (n_classes * c)
            if class_weight_mode == "sqrt":
                w = np.sqrt(w)
            weights.append(w)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor, label_smoothing=label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-2)),
    )

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(enabled=amp_enabled)
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    epochs = max(0, int(train_cfg.get("epochs", 1)))
    steps_per_epoch = max(1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
    scheduler, scheduler_step_per_batch = _build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    early_stop_cfg = (
        train_cfg.get("early_stop", {}) if isinstance(train_cfg.get("early_stop"), dict) else {}
    )
    early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
    early_stop_monitor = str(early_stop_cfg.get("monitor", "val/macro_f1")).strip()
    early_stop_mode = str(early_stop_cfg.get("mode", "max")).strip().lower()
    if early_stop_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported train.early_stop.mode: {early_stop_mode}")
    early_stop_patience = max(0, int(early_stop_cfg.get("patience", 3)))
    early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.001))
    early_stop_warmup_epochs = max(0, int(early_stop_cfg.get("warmup_epochs", 2)))

    checkpoints_dir = ctx.run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_monitor = float("-inf") if early_stop_mode == "max" else float("inf")
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    train_loss_history: list[float] = []
    start_epoch = 0
    resumed = False
    early_stopped = False
    stop_epoch = 0
    stop_reason = ""
    no_improve_epochs = 0

    if bool(cfg.get("runner", {}).get("resume", False)):
        if last_path.exists():
            resume_state = torch.load(last_path, map_location="cpu")
            if isinstance(resume_state, dict) and "model_state_dict" in resume_state:
                model.load_state_dict(resume_state["model_state_dict"])
                if "optimizer_state_dict" in resume_state:
                    optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                scaler_state = resume_state.get("scaler_state_dict")
                if scaler_state and hasattr(scaler, "load_state_dict"):
                    scaler.load_state_dict(scaler_state)
                scheduler_state = resume_state.get("scheduler_state_dict")
                if scheduler is not None and scheduler_state:
                    scheduler.load_state_dict(scheduler_state)

                start_epoch = max(0, int(resume_state.get("epoch", 0)))
                best_epoch = int(resume_state.get("best_epoch", 0))
                best_macro_f1 = float(
                    resume_state.get("best_val_macro_f1", float("-inf"))
                )
                best_monitor = float(
                    resume_state.get(
                        "best_monitor",
                        float("-inf") if early_stop_mode == "max" else float("inf"),
                    )
                )
                resumed = True

                if best_path.exists():
                    best_ckpt = torch.load(best_path, map_location="cpu")
                    if isinstance(best_ckpt, dict) and isinstance(
                        best_ckpt.get("model_state_dict"), dict
                    ):
                        best_state = best_ckpt["model_state_dict"]
                else:
                    best_state = {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    }

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        num_batches = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)

            if mix_mode != "none":
                mixed_images, target_a, target_b, lam = apply_batch_mix(
                    images=images,
                    targets=targets,
                    mix=mix_mode,
                )
            else:
                mixed_images = images
                target_a = targets
                target_b = targets
                lam = 1.0

            with _autocast(enabled=amp_enabled):
                logits = model(mixed_images)
                if mix_mode == "none":
                    loss = criterion(logits, targets)
                else:
                    loss_a = criterion(logits, target_a)
                    loss_b = criterion(logits, target_b)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        if num_batches > 0 and num_batches % grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

        mean_train_loss = float(epoch_loss / max(1, num_batches))
        train_loss_history.append(mean_train_loss)
        val_loss, val_macro_f1 = _evaluate(
            model, val_loader, criterion, device, idx_to_label
        )
        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        monitor_value = val_macro_f1
        if early_stop_monitor in {"val/loss", "loss", "val_loss"}:
            monitor_value = val_loss

        improved = (
            monitor_value > (best_monitor + early_stop_min_delta)
            if early_stop_mode == "max"
            else monitor_value < (best_monitor - early_stop_min_delta)
        )
        if improved:
            best_monitor = monitor_value
            best_epoch = epoch + 1
            best_macro_f1 = float(val_macro_f1)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict()
                    if hasattr(scaler, "state_dict")
                    else None,
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "idx_to_label": idx_to_label,
                    "target_size": target_size,
                    "image_extensions": image_exts,
                    "dummy_data": dummy_data,
                    "model_backbone_requested": model_backbone_requested,
                    "model_backbone_used": model_backbone_used,
                    "fallback_used": fallback_used,
                    "best_val_macro_f1": float(best_macro_f1),
                    "best_epoch": int(best_epoch),
                    "best_monitor": float(best_monitor),
                    "epoch": int(epoch + 1),
                },
                best_path,
            )
        elif (epoch + 1) >= early_stop_warmup_epochs:
            no_improve_epochs += 1

        current_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state_dict": current_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict()
                if hasattr(scaler, "state_dict")
                else None,
                "scheduler_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "idx_to_label": idx_to_label,
                "target_size": target_size,
                "image_extensions": image_exts,
                "dummy_data": dummy_data,
                "model_backbone_requested": model_backbone_requested,
                "model_backbone_used": model_backbone_used,
                "fallback_used": fallback_used,
                "best_val_macro_f1": float(best_macro_f1),
                "best_epoch": int(best_epoch),
                "best_monitor": float(best_monitor),
                "epoch": int(epoch + 1),
            },
            last_path,
        )

        lr_value = float(optimizer.param_groups[0].get("lr", 0.0))
        logger.info(
            "Train epoch end | epoch=%s/%s train_loss=%.6f val_loss=%.6f val_macro_f1=%.6f lr=%.8f no_improve=%s",
            int(epoch + 1),
            int(epochs),
            float(mean_train_loss),
            float(val_loss),
            float(val_macro_f1),
            float(lr_value),
            int(no_improve_epochs),
        )
        _safe_wandb_log(
            ctx,
            {
                "train/loss": float(mean_train_loss),
                "val/loss": float(val_loss),
                "val/macro_f1": float(val_macro_f1),
                "train/lr": lr_value,
            },
            step=int(epoch + 1),
        )

        stop_epoch = epoch + 1
        if (
            early_stop_enabled
            and (epoch + 1) >= early_stop_warmup_epochs
            and no_improve_epochs >= early_stop_patience
        ):
            early_stopped = True
            stop_reason = "early_stop_patience"
            logger.info(
                "Train early stop | epoch=%s patience=%s monitor=%s mode=%s",
                int(epoch + 1),
                int(early_stop_patience),
                early_stop_monitor,
                early_stop_mode,
            )
            break

    if epochs == 0 or best_epoch == 0:
        val_loss, val_macro_f1 = _evaluate(
            model, val_loader, criterion, device, idx_to_label
        )
        best_macro_f1 = float(val_macro_f1)
        best_epoch = max(1, stop_epoch)
        best_monitor = (
            float(val_loss)
            if early_stop_monitor in {"val/loss", "loss", "val_loss"}
            else float(val_macro_f1)
        )
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict()
                if hasattr(scaler, "state_dict")
                else None,
                "scheduler_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "idx_to_label": idx_to_label,
                "target_size": target_size,
                "image_extensions": image_exts,
                "dummy_data": dummy_data,
                "model_backbone_requested": model_backbone_requested,
                "model_backbone_used": model_backbone_used,
                "fallback_used": fallback_used,
                "best_val_macro_f1": float(best_macro_f1),
                "best_epoch": int(best_epoch),
                "best_monitor": float(best_monitor),
                "epoch": int(stop_epoch),
            },
            best_path,
        )

    if not best_path.exists():
        torch.save(
            {
                "model_state_dict": best_state,
                "idx_to_label": idx_to_label,
                "target_size": target_size,
                "image_extensions": image_exts,
                "dummy_data": dummy_data,
                "model_backbone_requested": model_backbone_requested,
                "model_backbone_used": model_backbone_used,
                "fallback_used": fallback_used,
                "best_val_macro_f1": float(best_macro_f1),
                "best_epoch": int(best_epoch),
                "best_monitor": float(best_monitor),
                "epoch": int(stop_epoch),
            },
            best_path,
        )

    elapsed_sec = float(time.perf_counter() - train_started)
    _safe_wandb_log(
        ctx,
        {
            "train/best_epoch": float(best_epoch),
            "train/best_val_macro_f1": float(best_macro_f1),
            "train/stop_epoch": float(stop_epoch),
            "train/early_stopped": 1.0 if early_stopped else 0.0,
            "train/resumed": 1.0 if resumed else 0.0,
            "train/elapsed_sec": elapsed_sec,
        },
        step=int(stop_epoch) if stop_epoch > 0 else None,
    )

    result = {
        "node": "train",
        "status": "completed",
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "epochs": int(epochs),
        "start_epoch": int(start_epoch),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_macro_f1),
        "best_monitor": float(best_monitor),
        "train_loss_last": float(train_loss_history[-1]) if train_loss_history else 0.0,
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "resumed": bool(resumed),
        "early_stopped": bool(early_stopped),
        "stop_epoch": int(stop_epoch),
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed_sec,
        "model_backbone_requested": model_backbone_requested,
        "model_backbone_used": model_backbone_used,
        "fallback_used": fallback_used,
        "preset_applied": preset,
        "mix_applied": bool(mix_mode != "none"),
        "config_snapshot": {
            "augmentation.preset": preset,
            "augmentation.mix": mix_mode_raw,
            "model.backbone": model_backbone_requested,
            "train.seed": int(seed),
            "train.class_weight_mode": str(class_weight_mode),
            "train.label_smoothing": float(train_cfg.get("label_smoothing", 0.0)),
            "optimizer.lr": float(optimizer_cfg.get("lr", 1e-4)),
            "preprocess.target_size": int(target_size),
        },
    }
    save_node_result(ctx, "train", result)
