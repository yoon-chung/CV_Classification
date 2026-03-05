import argparse
from pathlib import Path

import pandas as pd


LABEL_FIXES = {
    "aec62dced7af97cd": 14,
    "c5182ab809478f12": 14,
    "45f0d2dfc7e47c03": 7,
    "1ec14a14bbe633db": 7,
    "8646f2c3280a4f49": 3,
}

DROP_IDS = {
    "2b1076abe3e4338d",
    "024fe478044874ab",
}


def normalize_id(value: str) -> str:
    return str(value).replace(".jpg", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/train_v1.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    if "ID" not in df.columns or "target" not in df.columns:
        raise ValueError("Expected columns: ID, target")

    df["ID_norm"] = df["ID"].map(normalize_id)

    for image_id, target in LABEL_FIXES.items():
        idx = df.index[df["ID_norm"] == image_id]
        if len(idx) != 1:
            raise ValueError(f"Expected exactly one row for {image_id}, found {len(idx)}")
        df.loc[idx, "target"] = target

    before = len(df)
    df = df[~df["ID_norm"].isin(DROP_IDS)].copy()
    after = len(df)

    df = df.drop(columns=["ID_norm"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Input rows: {before}")
    print(f"Output rows: {after}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
