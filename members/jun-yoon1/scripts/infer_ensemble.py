import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import DocumentDataset
from src.data.transforms import build_val_transform
from src.models.factory import build_model
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=str, default="configs/base/data.yaml")
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def get_test_ids(test_dir: Path) -> list[str]:
    ids = sorted([p.stem for p in test_dir.glob("*.jpg")])
    if not ids:
        raise FileNotFoundError(f"No test images found in {test_dir}")
    return ids


def predict_single_checkpoint(
    model_name: str,
    ckpt_path: Path,
    test_ids: list[str],
    test_dir: Path,
    num_classes: int,
    device: torch.device,
    num_workers: int,
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 384))

    transform = build_val_transform(image_size=image_size)
    ds = DocumentDataset(
        ids=test_ids,
        image_dir=test_dir,
        targets=None,
        transform=transform,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    exp_dir = args.exp_dir
    num_classes = int(data_cfg["num_classes"])
    test_dir = Path(data_cfg["test_image_dir"])
    sample_submission_path = Path(data_cfg["sample_submission_csv"])
    pred_dir = exp_dir / "predictions"
    sub_dir = exp_dir / "submissions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    if args.weights_file is None:
        weights_file = exp_dir / "ensemble" / "weights.csv"
    else:
        weights_file = args.weights_file
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    weights_df = pd.read_csv(weights_file)
    models = weights_df["model"].tolist()
    weights = weights_df["weight"].values.astype(np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)

    test_ids = get_test_ids(test_dir)
    test_ids_with_ext = [f"{x}.jpg" for x in test_ids]
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    model_probs = {}
    for model_name in models:
        model_dir = exp_dir / model_name
        ckpts = sorted(model_dir.glob(f"{model_name}_fold*_best.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints for model: {model_name}")

        fold_probs = []
        for ckpt in ckpts:
            print(f"Infer model={model_name} ckpt={ckpt.name}")
            p = predict_single_checkpoint(
                model_name=model_name,
                ckpt_path=ckpt,
                test_ids=test_ids,
                test_dir=test_dir,
                num_classes=num_classes,
                device=device,
                num_workers=int(args.num_workers),
                batch_size=int(args.batch_size),
            )
            fold_probs.append(p)

        avg_probs = np.mean(fold_probs, axis=0)
        model_probs[model_name] = avg_probs

        cols = [f"p{i}" for i in range(num_classes)]
        model_df = pd.DataFrame(avg_probs, columns=cols)
        model_df.insert(0, "ID", test_ids_with_ext)
        model_df.to_csv(pred_dir / f"{model_name}_test_proba.csv", index=False)

    ens_probs = np.zeros((len(test_ids), num_classes), dtype=np.float64)
    for w, m in zip(weights, models):
        ens_probs += w * model_probs[m]

    ens_pred = ens_probs.argmax(axis=1).astype(int)
    pred_df = pd.DataFrame({"ID": test_ids_with_ext, "target": ens_pred})
    pred_df.to_csv(pred_dir / "ensemble_pred.csv", index=False)

    sample = pd.read_csv(sample_submission_path)
    merged = sample[["ID"]].merge(pred_df, on="ID", how="left")
    merged["target"] = merged["target"].fillna(0).astype(int)
    submit_path = sub_dir / "submission_weighted.csv"
    merged.to_csv(submit_path, index=False)

    print(f"Saved: {submit_path}")
    print("Weights:")
    print(weights_df.to_string(index=False))


if __name__ == "__main__":
    main()
