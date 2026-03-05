from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torch.optim import AdamW  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]

from experiment.ml.augs import (
    MixMode,
    apply_batch_mix,
    build_train_transform,
    build_valid_transform,
    validate_mix_label_smoothing,
)
from experiment.ml.data import (
    DocumentTrainDataset,
    SplitConfig,
    build_dummy_train_dataframe,
    load_train_dataframe,
    split_train_valid_indices,
)
from experiment.ml.metrics import macro_f1_score
from experiment.ml.model import create_timm_model


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _build_class_weights(
    targets: np.ndarray, num_classes: int, mode: str
) -> torch.Tensor | None:
    if mode not in {"balanced", "sqrt"}:
        return None
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    total = float(class_counts.sum())
    weights = total / (num_classes * class_counts)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _build_datasets(
    cfg: dict[str, Any],
) -> tuple[DocumentTrainDataset, DocumentTrainDataset, np.ndarray]:
    dataset_cfg = cfg["dataset"]
    split_cfg = cfg["split"]
    preprocess_cfg = cfg["preprocess"]
    train_cfg = cfg["train"]

    dummy_mode = bool(cfg["runner"].get("dummy_data", False))
    target_size = int(preprocess_cfg["target_size"])

    if dummy_mode:
        df = build_dummy_train_dataframe(
            num_samples=96, num_classes=int(cfg["model"]["num_classes"])
        )
    else:
        df = load_train_dataframe(dataset_cfg["train_csv"])

    split = SplitConfig(
        strategy=str(split_cfg["strategy"]),
        n_splits=int(split_cfg["n_splits"]),
        fold_index=int(split_cfg["fold_index"]),
        seed=int(split_cfg["seed"]),
    )
    train_idx, valid_idx = split_train_valid_indices(df, split)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    image_extensions = dataset_cfg.get("image_extensions", [".jpg", ".png", ".jpeg"])

    train_dataset = DocumentTrainDataset(
        dataframe=train_df,
        image_dir=dataset_cfg["image_dir_train"],
        image_extensions=image_extensions,
        transform=build_train_transform(target_size),
        dummy_mode=dummy_mode,
        image_size=target_size,
        seed=int(train_cfg["seed"]),
    )
    valid_dataset = DocumentTrainDataset(
        dataframe=valid_df,
        image_dir=dataset_cfg["image_dir_train"],
        image_extensions=image_extensions,
        transform=build_valid_transform(target_size),
        dummy_mode=dummy_mode,
        image_size=target_size,
        seed=int(train_cfg["seed"]) + 100_000,
    )
    return train_dataset, valid_dataset, train_df["target"].to_numpy(dtype=np.int64)


def run_training(cfg: dict[str, Any], run_dir: str | Path) -> dict[str, float | int]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    optimizer_cfg = cfg["optimizer"]
    augmentation_cfg = cfg["augmentation"]

    mix_mode_raw = str(augmentation_cfg.get("mix", "none"))
    if mix_mode_raw not in {"none", "mixup", "cutmix"}:
        raise ValueError(f"Unsupported augmentation.mix: {mix_mode_raw}")
    mix_mode = cast(MixMode, mix_mode_raw)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    validate_mix_label_smoothing(mix_mode, label_smoothing)

    _set_seed(int(train_cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, valid_dataset, train_targets = _build_datasets(cfg)

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_timm_model(model_cfg).to(device)

    class_weights = _build_class_weights(
        targets=train_targets,
        num_classes=int(model_cfg["num_classes"]),
        mode=str(train_cfg.get("class_weight_mode", "none")),
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(enabled=use_amp)
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))

    epochs = int(train_cfg["epochs"])
    start_time = time.perf_counter()
    train_loss_last = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)

            mixed_images, target_a, target_b, lam = apply_batch_mix(
                images=images,
                targets=targets,
                mix=mix_mode,
            )

            with _autocast(enabled=use_amp):
                logits = model(mixed_images)
                loss_a = criterion(logits, target_a)
                loss_b = criterion(logits, target_b)
                loss = lam * loss_a + (1.0 - lam) * loss_b
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch = int(images.size(0))
            running_loss += float(loss.item()) * grad_accum_steps * batch
            seen += batch

        train_loss_last = running_loss / max(seen, 1)

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    val_loss_sum = 0.0
    n_val = 0

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            with _autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch = int(images.size(0))
            val_loss_sum += float(loss.item()) * batch
            n_val += batch

    macro_f1 = macro_f1_score(np.asarray(y_true), np.asarray(y_pred))
    val_loss = val_loss_sum / max(n_val, 1)
    elapsed_sec = time.perf_counter() - start_time

    train_json = {
        "train/loss": float(train_loss_last),
        "train/epochs": epochs,
        "elapsed_sec": float(elapsed_sec),
    }
    eval_json = {
        "macro_f1": float(macro_f1),
        "val/macro_f1": float(macro_f1),
        "val/loss": float(val_loss),
        "n_val": int(n_val),
    }

    (run_path / "train.json").write_text(
        json.dumps(train_json, indent=2), encoding="utf-8"
    )
    (run_path / "eval.json").write_text(
        json.dumps(eval_json, indent=2), encoding="utf-8"
    )
    return eval_json


def _default_cfg() -> dict[str, Any]:
    return {
        "runner": {"dummy_data": True},
        "dataset": {
            "train_csv": "data/train.csv",
            "image_dir_train": "data/train",
            "image_extensions": [".jpg", ".jpeg", ".png"],
        },
        "split": {
            "strategy": "stratified_kfold",
            "n_splits": 5,
            "fold_index": 0,
            "seed": 42,
        },
        "preprocess": {"target_size": 96},
        "model": {"backbone": "resnet18", "pretrained": False, "num_classes": 17},
        "train": {
            "seed": 42,
            "epochs": 1,
            "batch_size": 32,
            "grad_accum_steps": 1,
            "amp": True,
            "label_smoothing": 0.0,
            "class_weight_mode": "none",
        },
        "optimizer": {"lr": 1e-3, "weight_decay": 1e-2},
        "augmentation": {"mix": "none"},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ML train/eval runner")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument(
        "--mix", type=str, choices=["none", "mixup", "cutmix"], default="none"
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        choices=["none", "balanced", "sqrt"],
        default="none",
    )
    parser.add_argument("--train-csv", type=str, default="data/train.csv")
    parser.add_argument("--image-dir-train", type=str, default="data/train")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _default_cfg()
    cfg["runner"]["dummy_data"] = bool(args.dummy)
    cfg["dataset"]["train_csv"] = args.train_csv
    cfg["dataset"]["image_dir_train"] = args.image_dir_train
    cfg["model"]["backbone"] = args.backbone
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["train"]["amp"] = bool(args.amp)
    cfg["train"]["class_weight_mode"] = args.class_weight_mode
    cfg["augmentation"]["mix"] = args.mix

    result = run_training(cfg=cfg, run_dir=args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
