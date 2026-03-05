from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from PIL import Image
from sklearn.model_selection import StratifiedKFold  # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset  # pyright: ignore[reportMissingImports]


ImageTransform = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int
    fold_index: int
    seed: int
    strategy: str = "stratified_kfold"


def load_train_dataframe(train_csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)
    required_columns = {"ID", "target"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"train.csv must include columns: {sorted(required_columns)}")
    return df[["ID", "target"]].copy()


def split_train_valid_indices(
    df: pd.DataFrame, split_cfg: SplitConfig
) -> tuple[np.ndarray, np.ndarray]:
    if split_cfg.strategy != "stratified_kfold":
        raise ValueError(f"Unsupported split.strategy: {split_cfg.strategy}")
    if split_cfg.n_splits < 2:
        raise ValueError("split.n_splits must be >= 2")
    if not 0 <= split_cfg.fold_index < split_cfg.n_splits:
        raise ValueError("split.fold_index must be in [0, n_splits)")

    splitter = StratifiedKFold(
        n_splits=split_cfg.n_splits,
        shuffle=True,
        random_state=split_cfg.seed,
    )

    splits = list(splitter.split(df["ID"].values, df["target"].values))
    train_idx, valid_idx = splits[split_cfg.fold_index]
    return train_idx, valid_idx


def build_dummy_train_dataframe(
    num_samples: int = 96,
    num_classes: int = 17,
) -> pd.DataFrame:
    if num_samples < num_classes:
        num_samples = num_classes

    targets = np.arange(num_samples, dtype=np.int64) % num_classes
    ids = [f"dummy_{i:05d}.jpg" for i in range(num_samples)]
    return pd.DataFrame({"ID": ids, "target": targets})


def _resolve_image_path(
    image_dir: Path,
    image_id: str,
    image_extensions: Sequence[str],
) -> Path:
    raw_path = image_dir / image_id
    if raw_path.exists():
        return raw_path

    stem = Path(image_id).stem
    for ext in image_extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for ID={image_id} under {image_dir}")


class DocumentTrainDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path,
        image_extensions: Sequence[str],
        transform: ImageTransform,
        *,
        dummy_mode: bool = False,
        image_size: int = 224,
        seed: int = 42,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_extensions = tuple(image_extensions)
        self.transform = transform
        self.dummy_mode = dummy_mode
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def _load_real_image(self, image_id: str) -> np.ndarray:
        image_path = _resolve_image_path(
            self.image_dir, image_id, self.image_extensions
        )
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def _load_dummy_image(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + index)
        image = rng.integers(
            low=0,
            high=256,
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        target = int(row["target"])

        if self.dummy_mode:
            image_np = self._load_dummy_image(index)
        else:
            image_np = self._load_real_image(str(row["ID"]))

        transformed = self.transform(image=image_np)
        image_tensor = transformed["image"]
        return image_tensor, target
