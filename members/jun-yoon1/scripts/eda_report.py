import argparse
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/train_v1.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--meta-csv", type=Path, default=Path("data/meta.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eda"))
    return parser.parse_args()


def normalize_id(value: str) -> str:
    return str(value).replace(".jpg", "")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    meta = pd.read_csv(args.meta_csv)
    df["ID_norm"] = df["ID"].map(normalize_id)

    class_count = (
        df["target"]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="count")
        .merge(meta, on="target", how="left")
        .sort_values("target")
    )
    class_count["ratio"] = class_count["count"] / class_count["count"].sum()
    class_count.to_csv(args.out_dir / "class_distribution.csv", index=False)

    h_list = []
    w_list = []
    missing = []
    for image_id in df["ID_norm"]:
        image_path = args.image_dir / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            missing.append(image_id)
            continue
        h, w = image.shape[:2]
        h_list.append(h)
        w_list.append(w)

    summary = {
        "num_rows": int(len(df)),
        "num_missing_images": int(len(missing)),
        "height_mean": float(pd.Series(h_list).mean()) if h_list else None,
        "height_std": float(pd.Series(h_list).std()) if h_list else None,
        "width_mean": float(pd.Series(w_list).mean()) if w_list else None,
        "width_std": float(pd.Series(w_list).std()) if w_list else None,
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)

    if missing:
        pd.DataFrame({"missing_id": missing}).to_csv(
            args.out_dir / "missing_images.csv", index=False
        )

    print(f"Saved: {args.out_dir}")


if __name__ == "__main__":
    main()
