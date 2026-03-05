from __future__ import annotations

from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions csv not found: {predictions_path}")
    return pd.read_csv(predictions_path)


def build_submission_frame(
    predictions: pd.DataFrame,
    *,
    id_column: str = "ID",
    target_column: str = "target",
) -> pd.DataFrame:
    if id_column not in predictions.columns or target_column not in predictions.columns:
        raise ValueError(
            f"predictions csv must contain {id_column} and {target_column} columns"
        )
    return predictions[[id_column, target_column]].copy()


def write_submission_csv(submission_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path
