from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]
from sklearn.metrics import f1_score  # pyright: ignore[reportMissingImports]


def macro_f1_score(
    y_true: np.ndarray | list[int], y_pred: np.ndarray | list[int]
) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))
