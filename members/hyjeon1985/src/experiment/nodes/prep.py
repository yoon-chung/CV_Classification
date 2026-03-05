from __future__ import annotations

from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import StratifiedKFold  # pyright: ignore[reportMissingImports]

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]
from experiment.pipeline import register_node  # pyright: ignore[reportMissingImports]

from .base import save_node_result


def _get_cfg(ctx: RuntimeContext) -> dict:
    if isinstance(ctx.cfg, dict):
        return ctx.cfg
    raise TypeError("RuntimeContext.cfg must be a dictionary")


def _build_dummy_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 17))
    n_train = max(num_classes * 8, 64)
    n_test = max(num_classes * 2, 16)

    train_rows = []
    for i in range(n_train):
        train_rows.append({"ID": f"dummy_train_{i:05d}.jpg", "target": i % num_classes})

    test_rows = [{"ID": f"dummy_test_{i:05d}.jpg"} for i in range(n_test)]
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def _load_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    dummy_data = bool(cfg.get("runner", {}).get("dummy_data", False))
    if dummy_data:
        train_df, test_df = _build_dummy_frames(cfg)
        return train_df, test_df, True

    dataset_cfg = cfg.get("dataset", {})
    train_csv = Path(str(dataset_cfg.get("train_csv", "")))
    test_csv = Path(str(dataset_cfg.get("test_csv", "")))
    sample_submission_csv = train_csv.parent / "sample_submission.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not test_csv.exists() and not sample_submission_csv.exists():
        raise FileNotFoundError(
            f"test.csv not found: {test_csv} and sample_submission.csv not found: {sample_submission_csv}"
        )

    train_df = pd.read_csv(train_csv)
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
    else:
        test_df = pd.read_csv(sample_submission_csv)

    if "ID" not in train_df.columns or "target" not in train_df.columns:
        raise ValueError("train.csv must contain columns: ID, target")
    if "ID" not in test_df.columns:
        raise ValueError("test.csv (or sample_submission.csv) must contain column: ID")

    return train_df[["ID", "target"]].copy(), test_df[["ID"]].copy(), False


def _split_train_val(
    cfg: dict, train_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split_cfg = cfg.get("split", {})
    strategy = str(split_cfg.get("strategy", "stratified_kfold"))
    if strategy == "full_train":
        train_split = train_df.reset_index(drop=True)
        # Keep val split non-empty for downstream eval/infer pipeline compatibility.
        val_split = train_df.reset_index(drop=True)
        meta = {
            "strategy": strategy,
            "n_splits": 1,
            "fold_index": 0,
            "seed": int(split_cfg.get("seed", 42)),
        }
        return train_split, val_split, meta
    if strategy != "stratified_kfold":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    n_splits = int(split_cfg.get("n_splits", 5))
    fold_index = int(split_cfg.get("fold_index", 0))
    seed = int(split_cfg.get("seed", 42))
    if n_splits < 2:
        raise ValueError("split.n_splits must be >= 2")
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"split.fold_index must be in [0, {n_splits - 1}]")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(train_df["ID"], train_df["target"]))
    train_idx, val_idx = splits[fold_index]
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)

    meta = {
        "strategy": strategy,
        "n_splits": n_splits,
        "fold_index": fold_index,
        "seed": seed,
    }
    return train_split, val_split, meta


@register_node("prep")
def prep_node(ctx: RuntimeContext) -> None:
    cfg = _get_cfg(ctx)
    train_df, test_df, dummy_data = _load_frames(cfg)
    train_split, val_split, split_meta = _split_train_val(cfg, train_df)

    prep_dir = ctx.run_dir / "artifacts" / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    train_split_path = prep_dir / "train_split.csv"
    val_split_path = prep_dir / "val_split.csv"
    test_ids_path = prep_dir / "test_ids.csv"

    train_split.to_csv(train_split_path, index=False)
    val_split.to_csv(val_split_path, index=False)
    test_df[["ID"]].to_csv(test_ids_path, index=False)

    result = {
        "node": "prep",
        "status": "completed",
        "dummy_data": dummy_data,
        "split": split_meta,
        "n_train": int(len(train_split)),
        "n_val": int(len(val_split)),
        "n_test": int(len(test_df)),
        "train_split_csv": str(train_split_path),
        "val_split_csv": str(val_split_path),
        "test_ids_csv": str(test_ids_path),
    }
    save_node_result(ctx, "prep", result)
