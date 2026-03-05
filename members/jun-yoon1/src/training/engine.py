from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import DocumentDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.models.factory import build_model


def normalize_id(value: str) -> str:
    return str(value).replace(".jpg", "")


def make_strong_classes(train_targets: np.ndarray, bottom_k: int = 3) -> set[int]:
    counts = pd.Series(train_targets).value_counts().sort_values()
    return set(counts.head(bottom_k).index.tolist())


def resolve_strong_classes(train_df: pd.DataFrame, aug_cfg: dict) -> set[int]:
    if not bool(aug_cfg.get("class_aware_policy", False)):
        return set()

    explicit = aug_cfg.get("weak_classes", [])
    if explicit:
        present = set(train_df["target"].unique().tolist())
        return set(int(x) for x in explicit if int(x) in present)

    strong_bottom_k = int(aug_cfg.get("strong_bottom_k", 3))
    return make_strong_classes(train_df["target"].values, bottom_k=strong_bottom_k)


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    class_aware_policy: bool,
    strong_classes: set[int],
    strong_profile: str,
):
    train_tf = build_train_transform(image_size=image_size, strong=False, strong_profile=strong_profile)
    strong_tf = build_train_transform(image_size=image_size, strong=True, strong_profile=strong_profile)
    val_tf = build_val_transform(image_size=image_size)

    train_ds = DocumentDataset(
        ids=train_df["ID_norm"].tolist(),
        image_dir=image_dir,
        targets=train_df["target"].tolist(),
        transform=train_tf,
        strong_transform=strong_tf if class_aware_policy else None,
        strong_classes=strong_classes,
    )
    val_ds = DocumentDataset(
        ids=val_df["ID_norm"].tolist(),
        image_dir=image_dir,
        targets=val_df["target"].tolist(),
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, strong_classes


def train_one_fold(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_cfg: dict,
    data_cfg: dict,
    model_cfg: dict,
    out_dir: str | Path,
    fold: int,
    num_classes: int,
    device: torch.device,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(model_cfg.get("image_size", data_cfg["image_size"]))
    batch_size = int(model_cfg.get("batch_size", data_cfg["batch_size"]))
    num_workers = int(data_cfg["num_workers"])
    aug_cfg = train_cfg["augmentation"]
    class_aware_policy = bool(aug_cfg["class_aware_policy"])
    strong_profile = str(aug_cfg.get("strong_profile", "v2"))
    strong_classes = resolve_strong_classes(train_df=train_df, aug_cfg=aug_cfg)

    train_loader, val_loader, strong_classes = build_loaders(
        train_df=train_df,
        val_df=val_df,
        image_dir=data_cfg["train_image_dir"],
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        class_aware_policy=class_aware_policy,
        strong_classes=strong_classes,
        strong_profile=strong_profile,
    )

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    class_weights = None
    if train_cfg["loss"].get("class_weight", "") == "balanced":
        train_counts = train_df["target"].value_counts().sort_index()
        weights = np.ones(num_classes, dtype=np.float32)
        for c in range(num_classes):
            cnt = train_counts.get(c, 1)
            weights[c] = len(train_df) / (num_classes * cnt)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(train_cfg["loss"].get("label_smoothing", 0.0)),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["epochs"]), eta_min=float(train_cfg["lr"]) * 0.05
    )
    scaler = GradScaler(enabled=bool(train_cfg["amp"]))

    best_f1 = -1.0
    best_path = out_dir / f"{model_name}_fold{fold}_best.pt"
    logs = []
    best_oof = None
    best_oof_proba = None

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        tr_loss = 0.0
        n_train = 0
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(train_cfg["amp"])):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * images.size(0)
            n_train += images.size(0)

        model.eval()
        val_loss = 0.0
        n_val = 0
        y_true = []
        y_pred = []
        y_prob = []
        y_ids = []
        with torch.no_grad():
            for images, targets, image_ids in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=bool(train_cfg["amp"])):
                    logits = model(images)
                    loss = criterion(logits, targets)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                val_loss += loss.item() * images.size(0)
                n_val += images.size(0)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                y_prob.extend(probs.cpu().numpy().tolist())
                y_ids.extend(list(image_ids))

        scheduler.step()
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        epoch_log = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": tr_loss / max(n_train, 1),
            "val_loss": val_loss / max(n_val, 1),
            "val_macro_f1": macro_f1,
            "strong_classes": sorted(list(strong_classes)),
        }
        logs.append(epoch_log)
        print(
            f"[{model_name}] fold={fold} epoch={epoch} "
            f"train_loss={epoch_log['train_loss']:.4f} "
            f"val_loss={epoch_log['val_loss']:.4f} "
            f"val_macro_f1={macro_f1:.5f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "model_name": model_name,
                    "fold": fold,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "image_size": image_size,
                },
                best_path,
            )
            best_oof = pd.DataFrame(
                {
                    "ID": [f"{x}.jpg" for x in y_ids],
                    "target_true": y_true,
                    "target_pred": y_pred,
                    "fold": fold,
                    "model": model_name,
                }
            )
            proba_cols = {
                f"p{c}": [row[c] for row in y_prob] for c in range(num_classes)
            }
            best_oof_proba = pd.DataFrame(
                {
                    "ID": [f"{x}.jpg" for x in y_ids],
                    "target_true": y_true,
                    "target_pred": y_pred,
                    "fold": fold,
                    "model": model_name,
                    **proba_cols,
                }
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(out_dir / f"{model_name}_fold{fold}_history.csv", index=False)
    if best_oof is None:
        best_oof = pd.DataFrame(columns=["ID", "target_true", "target_pred", "fold", "model"])
    if best_oof_proba is None:
        cols = ["ID", "target_true", "target_pred", "fold", "model"] + [
            f"p{c}" for c in range(num_classes)
        ]
        best_oof_proba = pd.DataFrame(columns=cols)
    best_oof.to_csv(out_dir / f"{model_name}_fold{fold}_oof.csv", index=False)
    best_oof_proba.to_csv(out_dir / f"{model_name}_fold{fold}_oof_proba.csv", index=False)

    return {
        "model": model_name,
        "fold": fold,
        "best_f1": best_f1,
        "checkpoint": str(best_path),
    }


def run_kfold_training(
    model_name: str,
    df: pd.DataFrame,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    train_cfg: dict,
    data_cfg: dict,
    model_cfg: dict,
    out_dir: str | Path,
    device: torch.device,
):
    results = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(data_cfg["num_classes"])
    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        fold_result = train_one_fold(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            out_dir=out_dir,
            fold=fold,
            num_classes=num_classes,
            device=device,
        )
        results.append(fold_result)

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / f"{model_name}_kfold_summary.csv", index=False)
    return res_df


def summarize_all_models(records: list[dict], out_path: str | Path):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    score = (
        df.groupby("model", as_index=False)
        .agg(f1_mean=("best_f1", "mean"), f1_std=("best_f1", "std"), n_folds=("best_f1", "count"))
    )
    score = score.sort_values("f1_mean", ascending=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    score.to_csv(out_path, index=False)
    return score
