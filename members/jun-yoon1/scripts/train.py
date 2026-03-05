import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.engine import normalize_id, run_kfold_training, summarize_all_models
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", type=str, default="ensemble3")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--direction", type=str, required=True)
    parser.add_argument("--base-exp-dir", type=str, default="experiments")
    parser.add_argument("--data-config", type=str, default="configs/base/data.yaml")
    parser.add_argument("--model-config", type=str, default="configs/base/model.yaml")
    parser.add_argument("--train-config", type=str, default="configs/base/train.yaml")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names. Empty uses configs/base/model.yaml list.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to run. Must be <= config n_splits.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs if > 0")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.base_exp_dir) / args.date / args.model / args.method / args.direction
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    if args.epochs > 0:
        train_cfg["epochs"] = int(args.epochs)

    set_seed(int(train_cfg["seed"]))

    df = pd.read_csv(data_cfg["train_csv"])
    if "ID" not in df.columns or "target" not in df.columns:
        raise ValueError("train csv must have columns: ID, target")
    df["ID_norm"] = df["ID"].map(normalize_id)

    max_folds = int(data_cfg["cv"]["n_splits"])
    use_folds = min(int(args.folds), max_folds)
    skf = StratifiedKFold(
        n_splits=max_folds,
        shuffle=True,
        random_state=int(data_cfg["cv"]["seed"]),
    )
    all_indices = list(skf.split(df["ID_norm"].values, df["target"].values))
    fold_indices = all_indices[:use_folds]

    if args.models.strip():
        cfg_map = {m["name"]: m for m in model_cfg["models"]}
        run_models = []
        for name in args.models.split(","):
            model_name = name.strip()
            if not model_name:
                continue
            if model_name in cfg_map:
                run_models.append(cfg_map[model_name])
            else:
                run_models.append({"name": model_name, "pretrained": True})
    else:
        run_models = model_cfg["models"]

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    (exp_dir / "notes.md").write_text(
        "# Experiment Notes\n\n"
        f"- date: {args.date}\n"
        f"- method: {args.method}\n"
        f"- direction: {args.direction}\n"
        f"- device: {device}\n"
        f"- folds: {use_folds}/{max_folds}\n"
        f"- models: {[m['name'] for m in run_models]}\n",
        encoding="utf-8",
    )

    all_records = []
    for m in run_models:
        name = m["name"]
        model_out_dir = exp_dir / name
        print(f"Start training model={name} | out={model_out_dir}")
        res_df = run_kfold_training(
            model_name=name,
            df=df,
            fold_indices=fold_indices,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            model_cfg=m,
            out_dir=model_out_dir,
            device=device,
        )
        all_records.extend(res_df.to_dict(orient="records"))

    summary = summarize_all_models(all_records, exp_dir / "models_summary.csv")
    print("Training finished.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
