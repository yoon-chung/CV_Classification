from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader, Dataset  # pyright: ignore[reportMissingImports]

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]
from experiment.pipeline import register_node  # pyright: ignore[reportMissingImports]

from .base import load_node_result, save_node_result
from .train import _build_model, _dummy_image_tensor, _get_cfg, _load_image_tensor


def _safe_wandb_log(
    ctx: RuntimeContext, metrics: dict, step: int | None = None
) -> None:
    try:
        ctx.wandb_logger.log(metrics, step=step)
    except Exception:
        pass


class _InferenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        image_exts: list[str],
        target_size: int,
        dummy_data: bool,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_exts = image_exts
        self.target_size = target_size
        self.dummy_data = dummy_data

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_id = str(self.frame.iloc[idx]["ID"])
        if self.dummy_data:
            image = _dummy_image_tensor(image_id, self.target_size)
        else:
            image = _load_image_tensor(
                self.image_dir, image_id, self.image_exts, self.target_size
            )
        return image, image_id


def _collate_infer(
    batch: list[tuple[torch.Tensor, str]],
) -> tuple[torch.Tensor, list[str]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    ids = [item[1] for item in batch]
    return images, ids


_VALID_TTA_VIEWS = {
    "none",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "hflip_rot90",
    "vflip_rot90",
    "hflip_rot180",
    "vflip_rot180",
}


def _resolve_tta_views(tta_cfg: dict) -> list[str]:
    views_raw = tta_cfg.get("views")
    views: list[str] = []
    if isinstance(views_raw, list) and views_raw:
        for item in views_raw:
            view = str(item).strip().lower()
            if view:
                views.append(view)
    elif bool(tta_cfg.get("hflip", False)):
        # Backward compatibility for previous flag-based config.
        views = ["none", "hflip"]
    else:
        views = ["none"]

    out: list[str] = []
    seen: set[str] = set()
    for view in views:
        if view not in _VALID_TTA_VIEWS:
            raise ValueError(f"Unsupported infer.tta.view: {view}")
        if view not in seen:
            out.append(view)
            seen.add(view)
    if "none" not in seen:
        out.insert(0, "none")
    return out


def _apply_tta_view(images: torch.Tensor, view: str) -> torch.Tensor:
    if view == "none":
        return images
    if view == "rot90":
        return torch.rot90(images, k=1, dims=(2, 3))
    if view == "rot180":
        return torch.rot90(images, k=2, dims=(2, 3))
    if view == "rot270":
        return torch.rot90(images, k=3, dims=(2, 3))
    if view == "hflip":
        return torch.flip(images, dims=[3])
    if view == "vflip":
        return torch.flip(images, dims=[2])
    if view == "hflip_rot90":
        return torch.flip(torch.rot90(images, k=1, dims=(2, 3)), dims=[3])
    if view == "vflip_rot90":
        return torch.flip(torch.rot90(images, k=1, dims=(2, 3)), dims=[2])
    if view == "hflip_rot180":
        return torch.flip(torch.rot90(images, k=2, dims=(2, 3)), dims=[3])
    if view == "vflip_rot180":
        return torch.flip(torch.rot90(images, k=2, dims=(2, 3)), dims=[2])
    raise ValueError(f"Unsupported TTA view: {view}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_infer_cache_key(
    *,
    checkpoint_path: Path,
    test_ids_path: Path,
    idx_to_label: list[int],
    target_size: int,
    tta_views: list[str],
    backbone_used: str,
) -> str:
    ck_stat = checkpoint_path.stat()
    test_stat = test_ids_path.stat()
    payload = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size": int(ck_stat.st_size),
        "checkpoint_mtime_ns": int(ck_stat.st_mtime_ns),
        "checkpoint_digest": _sha256_file(checkpoint_path),
        "test_ids_path": str(test_ids_path.resolve()),
        "test_ids_size": int(test_stat.st_size),
        "test_ids_mtime_ns": int(test_stat.st_mtime_ns),
        "test_ids_digest": _sha256_file(test_ids_path),
        "idx_to_label": [int(v) for v in idx_to_label],
        "target_size": int(target_size),
        "tta_views": [str(v) for v in tta_views],
        "backbone_used": str(backbone_used),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _read_cache_probabilities(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            ids = np.asarray(payload["ids"]).astype(str)
            probs = np.asarray(payload["probs"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
    except Exception as exc:
        raise RuntimeError(f"Failed to load infer cache: {cache_path}") from exc

    if ids.ndim != 1 or probs.ndim != 2 or labels.ndim != 1:
        raise ValueError(f"Invalid infer cache payload shape: {cache_path}")
    if probs.shape[0] != ids.shape[0] or probs.shape[1] != labels.shape[0]:
        raise ValueError(f"Infer cache payload mismatch: {cache_path}")
    return ids, probs, labels


def _build_prediction_rows(
    *,
    ordered_ids: list[str],
    probs: np.ndarray,
    idx_to_label: list[int],
) -> tuple[list[dict], list[dict]]:
    pred_rows: list[dict] = []
    confidence_rows: list[dict] = []
    if probs.ndim != 2 or probs.shape[0] != len(ordered_ids):
        raise ValueError("Prediction probability shape mismatch")

    for i, image_id in enumerate(ordered_ids):
        row = probs[i]
        pred_idx = int(np.argmax(row))
        label = int(idx_to_label[pred_idx]) if pred_idx < len(idx_to_label) else pred_idx

        if row.shape[0] >= 2:
            top2_idx = np.argpartition(row, -2)[-2:]
            top2_idx = top2_idx[np.argsort(row[top2_idx])[::-1]]
            prob_top1 = float(row[top2_idx[0]])
            prob_top2 = float(row[top2_idx[1]])
        else:
            prob_top1 = float(row[0]) if row.shape[0] == 1 else 0.0
            prob_top2 = 0.0

        pred_rows.append({"ID": image_id, "target": label})
        confidence_rows.append(
            {
                "ID": image_id,
                "target": label,
                "prob_top1": prob_top1,
                "prob_top2": prob_top2,
                "prob_margin": float(prob_top1 - prob_top2),
            }
        )
    return pred_rows, confidence_rows


@register_node("infer")
def infer_node(ctx: RuntimeContext) -> None:
    cfg = _get_cfg(ctx)

    prep = load_node_result(ctx, "prep")
    train_result = load_node_result(ctx, "train")
    if prep is None or train_result is None:
        raise RuntimeError("prep.json and train.json are required before infer node")

    test_ids_path = Path(str(prep.get("test_ids_csv", "")))
    checkpoint_path = Path(str(train_result.get("checkpoint", "")))
    if not test_ids_path.exists():
        raise FileNotFoundError(f"test IDs csv not found: {test_ids_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    test_df = pd.read_csv(test_ids_path)
    if "ID" not in test_df.columns:
        raise ValueError("test IDs csv must contain ID column")

    state = torch.load(checkpoint_path, map_location="cpu")
    idx_to_label = [int(v) for v in state.get("idx_to_label", [])]
    if not idx_to_label:
        raise RuntimeError("Checkpoint does not contain idx_to_label")

    dataset_cfg = cfg.get("dataset", {})
    image_dir = Path(str(dataset_cfg.get("image_dir_test", "")))
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

    infer_dataset = _InferenceDataset(
        frame=test_df,
        image_dir=image_dir,
        image_exts=image_exts,
        target_size=target_size,
        dummy_data=dummy_data,
    )

    batch_size = int(cfg.get("train", {}).get("batch_size", 16))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_infer,
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
    model.eval()

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
    export_confidence_csv = (
        bool(export_cfg.get("infer_predictions_with_confidence", True)) and metrics_enabled
    )
    export_probabilities = (
        bool(export_cfg.get("infer_probabilities", True)) and metrics_enabled
    )
    infer_cfg = cfg.get("infer", {}) if isinstance(cfg.get("infer"), dict) else {}
    tta_cfg = (
        infer_cfg.get("tta", {}) if isinstance(infer_cfg.get("tta"), dict) else {}
    )
    tta_enabled = bool(tta_cfg.get("enabled", False))
    tta_views = _resolve_tta_views(tta_cfg) if tta_enabled else ["none"]
    tta_view_count = len(tta_views)

    pipeline_cache_cfg = (
        cfg.get("pipeline", {}).get("cache", {})
        if isinstance(cfg.get("pipeline"), dict)
        and isinstance(cfg.get("pipeline", {}).get("cache"), dict)
        else {}
    )
    tta_cache_enabled = tta_enabled and bool(tta_cfg.get("cache_enabled", False))
    cache_root_raw = str(tta_cfg.get("cache_root", "")).strip()
    if not cache_root_raw:
        cache_root_raw = str(pipeline_cache_cfg.get("root", "")).strip()
    cache_namespace = str(pipeline_cache_cfg.get("namespace", "cvdc")).strip() or "cvdc"
    cache_subdir = str(tta_cfg.get("cache_subdir", "infer_tta")).strip() or "infer_tta"
    cache_dir: Path | None = None
    if tta_cache_enabled and cache_root_raw:
        cache_dir = Path(cache_root_raw) / cache_namespace / cache_subdir
        cache_dir.mkdir(parents=True, exist_ok=True)

    ordered_test_ids = [str(v) for v in test_df["ID"].tolist()]
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key = _build_infer_cache_key(
            checkpoint_path=checkpoint_path,
            test_ids_path=test_ids_path,
            idx_to_label=idx_to_label,
            target_size=target_size,
            tta_views=tta_views,
            backbone_used=str(state.get("model_backbone_used", "")),
        )
        cache_path = cache_dir / f"{cache_key}.npz"

    pred_rows: list[dict] = []
    confidence_rows: list[dict] = []
    proba_ids: list[str] = []
    proba_rows: list[np.ndarray] = []
    cache_hit = False
    if cache_path is not None and cache_path.exists():
        try:
            cache_ids, cache_probs, cache_labels = _read_cache_probabilities(cache_path)
            id_to_idx = {str(v): i for i, v in enumerate(cache_ids.tolist())}
            label_to_idx = {int(v): i for i, v in enumerate(cache_labels.tolist())}
            row_take = np.asarray([id_to_idx[v] for v in ordered_test_ids], dtype=np.int64)
            col_take = np.asarray([label_to_idx[int(v)] for v in idx_to_label], dtype=np.int64)
            aligned_probs = cache_probs[row_take][:, col_take]
            proba_ids = ordered_test_ids.copy()
            proba_rows = [aligned_probs[i].astype(np.float32) for i in range(aligned_probs.shape[0])]
            pred_rows, confidence_rows = _build_prediction_rows(
                ordered_ids=ordered_test_ids,
                probs=aligned_probs,
                idx_to_label=idx_to_label,
            )
            cache_hit = True
        except Exception:
            cache_hit = False

    if not cache_hit:
        with torch.no_grad():
            for images, ids in infer_loader:
                images = images.to(device)
                probs_sum: torch.Tensor | None = None
                for view in tta_views:
                    view_images = _apply_tta_view(images, view)
                    view_probs = torch.softmax(model(view_images), dim=1)
                    probs_sum = view_probs if probs_sum is None else (probs_sum + view_probs)
                if probs_sum is None:
                    raise RuntimeError("No TTA views configured for infer")
                probs = probs_sum / float(max(1, tta_view_count))
                probs_cpu = probs.cpu().numpy().astype(np.float32)
                for i, image_id in enumerate(ids):
                    proba_ids.append(str(image_id))
                    proba_rows.append(probs_cpu[i])

        aligned_probs = np.asarray(proba_rows, dtype=np.float32)
        pred_rows, confidence_rows = _build_prediction_rows(
            ordered_ids=proba_ids,
            probs=aligned_probs,
            idx_to_label=idx_to_label,
        )
        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                ids=np.asarray(proba_ids, dtype=str),
                probs=aligned_probs,
                labels=np.asarray(idx_to_label, dtype=np.int64),
                views=np.asarray(tta_views, dtype=str),
            )

    infer_path = ctx.run_dir / "predictions.csv"
    pd.DataFrame(pred_rows).to_csv(infer_path, index=False)
    confidence_path = ctx.run_dir / "artifacts" / "infer" / "predictions_with_confidence.csv"
    if export_confidence_csv:
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(confidence_rows).to_csv(confidence_path, index=False)
    proba_path = ctx.run_dir / "artifacts" / "infer" / "predictions_proba.npz"
    if export_probabilities:
        proba_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            proba_path,
            ids=np.asarray(proba_ids, dtype=str),
            probs=np.asarray(proba_rows, dtype=np.float32),
            labels=np.asarray(idx_to_label, dtype=np.int64),
        )

    confidence_values = [float(row["prob_top1"]) for row in confidence_rows]
    margin_values = [float(row["prob_margin"]) for row in confidence_rows]
    low_margin_rate = (
        float(np.mean([v <= low_margin_threshold for v in margin_values]))
        if margin_values
        else 0.0
    )
    high_conf_rate = (
        float(np.mean([v >= high_conf_threshold for v in confidence_values]))
        if confidence_values
        else 0.0
    )
    infer_metrics = {
        "infer/n_test": float(len(pred_rows)),
        "infer/tta_enabled": 1.0 if tta_enabled else 0.0,
        "infer/tta_views": float(tta_view_count),
        "infer/tta_cache_enabled": 1.0 if tta_cache_enabled else 0.0,
        "infer/tta_cache_hit": 1.0 if cache_hit else 0.0,
    }
    if metrics_enabled:
        infer_metrics.update(
            {
                "infer/confidence_mean": float(np.mean(confidence_values))
                if confidence_values
                else 0.0,
                "infer/confidence_std": float(np.std(confidence_values))
                if confidence_values
                else 0.0,
                "infer/margin_mean": float(np.mean(margin_values))
                if margin_values
                else 0.0,
                "infer/margin_std": float(np.std(margin_values)) if margin_values else 0.0,
                "infer/low_margin_rate": low_margin_rate,
                "infer/high_conf_rate": high_conf_rate,
            }
        )
    _safe_wandb_log(ctx, infer_metrics)

    result = {
        "node": "infer",
        "status": "completed",
        "n_test": int(len(pred_rows)),
        "predictions_csv": str(infer_path),
        "predictions_with_confidence_csv": str(confidence_path)
        if export_confidence_csv
        else None,
        "predictions_proba_npz": str(proba_path) if export_probabilities else None,
        "tta_enabled": bool(tta_enabled),
        "tta_views": int(tta_view_count),
        "tta_cache_enabled": bool(tta_cache_enabled),
        "tta_cache_hit": bool(cache_hit),
        "tta_view_list": [str(v) for v in tta_views],
        "tta_cache_path": str(cache_path) if cache_path is not None else None,
        **infer_metrics,
    }
    save_node_result(ctx, "infer", result)
