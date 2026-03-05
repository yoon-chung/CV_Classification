from __future__ import annotations

import os
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    f1_score,
    precision_recall_fscore_support,
)
import torch  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]
from experiment.pipeline import register_node  # pyright: ignore[reportMissingImports]

from .base import load_node_result, save_node_result
from .train import _ImageDataset, _build_model, _get_cfg


def _safe_wandb_log(
    ctx: RuntimeContext, metrics: dict, step: int | None = None
) -> None:
    try:
        ctx.wandb_logger.log(metrics, step=step)
    except Exception:
        pass


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).std())


def _evaluate_with_metadata(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    idx_to_label: list[int],
    val_df: pd.DataFrame,
    low_margin_threshold: float,
    high_conf_wrong_threshold: float,
) -> tuple[float, float, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    pred_rows: list[dict[str, float | int | str]] = []

    id_values = [str(v) for v in val_df["ID"].tolist()]
    row_offset = 0

    with torch.no_grad():
        for images, targets in loader:
            batch_size = int(targets.shape[0])
            batch_ids = id_values[row_offset : row_offset + batch_size]
            row_offset += batch_size

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))

            probs = torch.softmax(logits, dim=1)
            top_k = min(2, int(probs.shape[1]))
            top_probs, _ = torch.topk(probs, k=top_k, dim=1)
            pred_idx = torch.argmax(logits, dim=1)

            pred_idx_cpu = pred_idx.detach().cpu().tolist()
            true_idx_cpu = targets.detach().cpu().tolist()
            top_probs_cpu = top_probs.detach().cpu().tolist()

            for i in range(batch_size):
                pred_i = int(pred_idx_cpu[i])
                true_i = int(true_idx_cpu[i])
                pred_label = (
                    int(idx_to_label[pred_i]) if pred_i < len(idx_to_label) else pred_i
                )
                true_label = (
                    int(idx_to_label[true_i]) if true_i < len(idx_to_label) else true_i
                )
                prob_top1 = float(top_probs_cpu[i][0]) if top_probs_cpu[i] else 0.0
                prob_top2 = (
                    float(top_probs_cpu[i][1]) if len(top_probs_cpu[i]) > 1 else 0.0
                )
                prob_margin = float(prob_top1 - prob_top2)
                is_correct = 1 if pred_label == true_label else 0

                y_true.append(true_label)
                y_pred.append(pred_label)
                pred_rows.append(
                    {
                        "ID": batch_ids[i] if i < len(batch_ids) else "",
                        "target_true": int(true_label),
                        "target_pred": int(pred_label),
                        "is_correct": int(is_correct),
                        "prob_top1": prob_top1,
                        "prob_top2": prob_top2,
                        "prob_margin": prob_margin,
                    }
                )

    if row_offset != len(id_values):
        raise RuntimeError(
            f"Eval row alignment mismatch: consumed={row_offset} expected={len(id_values)}"
        )

    mean_loss = float(sum(losses) / max(1, len(losses)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    labels = [int(v) for v in idx_to_label]
    precisions, recalls, f1_values, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    class_rows: list[dict[str, float | int]] = []
    for label, precision, recall, f1_value, support in zip(
        labels, precisions, recalls, f1_values, supports
    ):
        class_rows.append(
            {
                "label": int(label),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1_value),
                "support": int(support),
            }
        )
    class_df = pd.DataFrame(class_rows)
    pred_df = pd.DataFrame(pred_rows)

    error_rate = float((pred_df["is_correct"] == 0).mean()) if not pred_df.empty else 0.0
    confidences = [float(v) for v in pred_df["prob_top1"].tolist()] if not pred_df.empty else []
    margins = [float(v) for v in pred_df["prob_margin"].tolist()] if not pred_df.empty else []
    correct_confidences = (
        [float(v) for v in pred_df.loc[pred_df["is_correct"] == 1, "prob_top1"].tolist()]
        if not pred_df.empty
        else []
    )
    wrong_confidences = (
        [float(v) for v in pred_df.loc[pred_df["is_correct"] == 0, "prob_top1"].tolist()]
        if not pred_df.empty
        else []
    )
    correct_margins = (
        [float(v) for v in pred_df.loc[pred_df["is_correct"] == 1, "prob_margin"].tolist()]
        if not pred_df.empty
        else []
    )
    wrong_margins = (
        [float(v) for v in pred_df.loc[pred_df["is_correct"] == 0, "prob_margin"].tolist()]
        if not pred_df.empty
        else []
    )

    low_margin_rate = (
        float((pred_df["prob_margin"] <= low_margin_threshold).mean())
        if not pred_df.empty
        else 0.0
    )
    high_conf_wrong_rate = (
        float(
            (
                (pred_df["is_correct"] == 0)
                & (pred_df["prob_top1"] >= high_conf_wrong_threshold)
            ).mean()
        )
        if not pred_df.empty
        else 0.0
    )

    worst_class_label = float(labels[0]) if labels else 0.0
    worst_class_f1 = 0.0
    class_f1_std = 0.0
    if len(f1_values) > 0:
        min_index = int(np.argmin(f1_values))
        worst_class_label = float(labels[min_index])
        worst_class_f1 = float(f1_values[min_index])
        class_f1_std = float(np.asarray(f1_values, dtype=np.float64).std())

    metadata_metrics: dict[str, float] = {
        "selection/error_rate": error_rate,
        "selection/n_errors": float(int((pred_df["is_correct"] == 0).sum()))
        if not pred_df.empty
        else 0.0,
        "selection/confidence_mean": float(_safe_mean(confidences) or 0.0),
        "selection/confidence_correct_mean": float(_safe_mean(correct_confidences) or 0.0),
        "selection/confidence_wrong_mean": float(_safe_mean(wrong_confidences) or 0.0),
        "selection/confidence_std": float(_safe_std(confidences) or 0.0),
        "selection/margin_mean": float(_safe_mean(margins) or 0.0),
        "selection/margin_correct_mean": float(_safe_mean(correct_margins) or 0.0),
        "selection/margin_wrong_mean": float(_safe_mean(wrong_margins) or 0.0),
        "selection/low_margin_rate": low_margin_rate,
        "selection/high_conf_wrong_rate": high_conf_wrong_rate,
        "selection/worst_class_label": worst_class_label,
        "selection/worst_class_f1": worst_class_f1,
        "selection/class_f1_std": class_f1_std,
    }

    return mean_loss, macro_f1, pred_df, class_df, metadata_metrics


@register_node("eval")
def eval_node(ctx: RuntimeContext) -> None:
    cfg = _get_cfg(ctx)

    prep = load_node_result(ctx, "prep")
    train_result = load_node_result(ctx, "train")
    if prep is None or train_result is None:
        raise RuntimeError("prep.json and train.json are required before eval node")

    val_split_path = Path(str(prep.get("val_split_csv", "")))
    checkpoint_path = Path(str(train_result.get("checkpoint", "")))
    if not val_split_path.exists():
        raise FileNotFoundError(f"val split csv not found: {val_split_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    val_df = pd.read_csv(val_split_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    idx_to_label = [int(v) for v in state.get("idx_to_label", [])]
    if not idx_to_label:
        idx_to_label = sorted({int(v) for v in val_df["target"].tolist()})
    label_to_idx = {label: i for i, label in enumerate(idx_to_label)}

    dataset_cfg = cfg.get("dataset", {})
    image_dir = Path(str(dataset_cfg.get("image_dir_train", "")))
    image_exts = [
        str(ext)
        for ext in state.get(
            "image_extensions",
            dataset_cfg.get("image_extensions", [".jpg", ".png", ".jpeg"]),
        )
    ]
    target_size = int(
        state.get("target_size", cfg.get("preprocess", {}).get("target_size", 224))
    )
    dummy_data = bool(state.get("dummy_data", prep.get("dummy_data", False)))

    eval_dataset = _ImageDataset(
        frame=val_df,
        image_dir=image_dir,
        image_exts=image_exts,
        target_size=target_size,
        dummy_data=dummy_data,
        label_to_idx=label_to_idx,
    )

    batch_size = int(cfg.get("train", {}).get("batch_size", 16))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone_override = state.get("model_backbone_used")
    model = _build_model(cfg, len(idx_to_label), backbone_override=backbone_override)
    model.load_state_dict(state["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_cuda = str(os.environ.get("CVDC_REQUIRE_CUDA", "0")).strip() == "1"
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CVDC_REQUIRE_CUDA=1 but torch.cuda.is_available() is false. Aborting to avoid CPU run."
        )
    model = model.to(device)

    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics"), dict) else {}
    metrics_enabled = bool(metrics_cfg.get("enabled", True))
    uncertainty_cfg = (
        metrics_cfg.get("uncertainty", {})
        if isinstance(metrics_cfg.get("uncertainty"), dict)
        else {}
    )
    export_cfg = (
        metrics_cfg.get("export", {}) if isinstance(metrics_cfg.get("export"), dict) else {}
    )
    low_margin_threshold = float(uncertainty_cfg.get("low_margin_threshold", 0.05))
    high_conf_threshold = float(uncertainty_cfg.get("high_conf_threshold", 0.90))
    export_eval_predictions = bool(export_cfg.get("eval_predictions", True)) and metrics_enabled
    export_class_metrics = bool(export_cfg.get("class_metrics", True)) and metrics_enabled

    criterion = nn.CrossEntropyLoss()
    val_loss, macro_f1, pred_df, class_df, metadata_metrics = _evaluate_with_metadata(
        model=model,
        loader=eval_loader,
        criterion=criterion,
        device=device,
        idx_to_label=idx_to_label,
        val_df=val_df,
        low_margin_threshold=low_margin_threshold,
        high_conf_wrong_threshold=high_conf_threshold,
    )

    best_val_macro_f1 = train_result.get("best_val_macro_f1")
    overfit_gap = None
    if isinstance(best_val_macro_f1, (int, float)):
        overfit_gap = float(best_val_macro_f1 - macro_f1)

    tune_selector = (
        cfg.get("tune", {}).get("selector", {})
        if isinstance(cfg.get("tune"), dict)
        and isinstance(cfg.get("tune", {}).get("selector"), dict)
        else {}
    )
    overfit_gap_threshold = float(tune_selector.get("overfit_gap_threshold", 0.03))
    overfit_weight = float(tune_selector.get("overfit_weight", 1.0))
    overfit_penalty = (
        max(0.0, float(overfit_gap) - overfit_gap_threshold) * overfit_weight
        if isinstance(overfit_gap, float)
        else 0.0
    )
    score_proxy = float(macro_f1 - overfit_penalty)

    selection_metrics: dict[str, float] = {
        "selection/val_macro_f1": float(macro_f1),
        "selection/val_loss": float(val_loss),
        "selection/fold_index": float(int(cfg.get("split", {}).get("fold_index", 0))),
        "selection/score_proxy": score_proxy,
        "tune/val_macro_f1": float(macro_f1),
        "tune/val_loss": float(val_loss),
    }
    if metrics_enabled:
        for metric_name, metric_value in metadata_metrics.items():
            selection_metrics[metric_name] = float(metric_value)
            if metric_name.startswith("selection/"):
                tune_metric = f"tune/{metric_name.split('/', 1)[1]}"
                selection_metrics[tune_metric] = float(metric_value)
    if isinstance(best_val_macro_f1, (int, float)):
        selection_metrics["selection/best_val_macro_f1"] = float(best_val_macro_f1)
        selection_metrics["tune/best_val_macro_f1"] = float(best_val_macro_f1)
    if isinstance(overfit_gap, float):
        selection_metrics["selection/overfit_gap"] = float(overfit_gap)
        selection_metrics["tune/overfit_gap"] = float(overfit_gap)
    if isinstance(best_val_macro_f1, (int, float)) and float(best_val_macro_f1) > 0.0:
        selection_metrics["selection/generalization_ratio"] = float(
            macro_f1 / float(best_val_macro_f1)
        )
    if isinstance(train_result.get("best_epoch"), (int, float)):
        selection_metrics["selection/best_epoch"] = float(train_result["best_epoch"])
    if isinstance(train_result.get("stop_epoch"), (int, float)):
        selection_metrics["selection/stop_epoch"] = float(train_result["stop_epoch"])
    if isinstance(train_result.get("early_stopped"), bool):
        selection_metrics["selection/early_stopped"] = (
            1.0 if train_result["early_stopped"] else 0.0
        )
    _safe_wandb_log(ctx, selection_metrics)

    eval_artifacts_dir = ctx.run_dir / "artifacts" / "eval"
    eval_artifacts_dir.mkdir(parents=True, exist_ok=True)
    val_predictions_path = eval_artifacts_dir / "val_predictions.csv"
    class_metrics_path = eval_artifacts_dir / "class_metrics.csv"
    if export_eval_predictions:
        pred_df.to_csv(val_predictions_path, index=False)
    if export_class_metrics:
        class_df.to_csv(class_metrics_path, index=False)

    result = {
        "node": "eval",
        "status": "completed",
        "macro_f1": float(macro_f1),
        "val/macro_f1": float(macro_f1),
        "val/loss": float(val_loss),
        "selection/overfit_gap": float(overfit_gap)
        if isinstance(overfit_gap, float)
        else None,
        "val_predictions_csv": str(val_predictions_path)
        if export_eval_predictions
        else None,
        "class_metrics_csv": str(class_metrics_path) if export_class_metrics else None,
        "n_val": int(len(val_df)),
    }
    if metrics_enabled:
        result.update({k: float(v) for k, v in metadata_metrics.items()})
    save_node_result(ctx, "eval", result)
