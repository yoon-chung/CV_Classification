import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--random-trials", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def discover_models(exp_dir: Path) -> list[str]:
    models = []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir():
            continue
        if list(p.glob("*_oof_proba.csv")):
            models.append(p.name)
    return models


def load_model_oof(exp_dir: Path, model_name: str) -> pd.DataFrame:
    model_dir = exp_dir / model_name
    paths = sorted(model_dir.glob(f"{model_name}_fold*_oof_proba.csv"))
    if not paths:
        raise FileNotFoundError(f"No OOF proba files for model: {model_name}")
    dfs = [pd.read_csv(p) for p in paths]
    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.sort_values("ID").reset_index(drop=True)
    return out


def parse_proba(df: pd.DataFrame) -> np.ndarray:
    proba_cols = [c for c in df.columns if c.startswith("p")]
    proba_cols = sorted(proba_cols, key=lambda x: int(x[1:]))
    return df[proba_cols].values


def evaluate_weights(
    weights: np.ndarray, probs_list: list[np.ndarray], y_true: np.ndarray
) -> float:
    ens = np.zeros_like(probs_list[0], dtype=np.float64)
    for w, p in zip(weights, probs_list):
        ens += w * p
    pred = ens.argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro"))


def make_grid_weights_3(step: float) -> np.ndarray:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    grid = []
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if w3 < 0:
                w3 = 0.0
            grid.append([w1, w2, w3])
    grid = np.array(grid, dtype=np.float64)
    grid = grid / np.clip(grid.sum(axis=1, keepdims=True), 1e-12, None)
    return grid


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir
    out_dir = args.out_dir or (exp_dir / "ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models(exp_dir)
    if len(models) < 2:
        raise ValueError("Need at least 2 models for ensemble weight search.")

    oof_by_model = {m: load_model_oof(exp_dir, m) for m in models}
    base = oof_by_model[models[0]][["ID", "target_true"]].copy()
    for m in models[1:]:
        chk = oof_by_model[m][["ID", "target_true"]]
        if not base["ID"].equals(chk["ID"]):
            raise ValueError(f"ID mismatch between models: {models[0]} vs {m}")
        if not base["target_true"].equals(chk["target_true"]):
            raise ValueError(f"target_true mismatch between models: {models[0]} vs {m}")

    y_true = base["target_true"].values
    probs_list = [parse_proba(oof_by_model[m]) for m in models]

    model_scores = []
    for m, p in zip(models, probs_list):
        pred = p.argmax(axis=1)
        score = f1_score(y_true, pred, average="macro")
        model_scores.append({"model": m, "oof_macro_f1": float(score)})
    pd.DataFrame(model_scores).sort_values("oof_macro_f1", ascending=False).to_csv(
        out_dir / "model_oof_scores.csv", index=False
    )

    rng = np.random.default_rng(args.seed)
    candidates = []
    n = len(models)

    onehots = np.eye(n, dtype=np.float64)
    for row in onehots:
        candidates.append(row)
    candidates.append(np.ones(n, dtype=np.float64) / n)

    if n == 3:
        grid = make_grid_weights_3(step=float(args.grid_step))
        candidates.extend(grid.tolist())

    rand_w = rng.dirichlet(alpha=np.ones(n), size=int(args.random_trials))
    candidates.extend(rand_w.tolist())

    best_score = -1.0
    best_w = None
    logs = []
    for w in candidates:
        w = np.array(w, dtype=np.float64)
        w = w / np.clip(w.sum(), 1e-12, None)
        score = evaluate_weights(w, probs_list, y_true)
        logs.append({"score": score, **{f"w_{m}": w[i] for i, m in enumerate(models)}})
        if score > best_score:
            best_score = score
            best_w = w.copy()

    local = np.clip(
        best_w + rng.normal(loc=0.0, scale=0.03, size=(2000, len(models))), 0.0, None
    )
    local = local / np.clip(local.sum(axis=1, keepdims=True), 1e-12, None)
    for w in local:
        score = evaluate_weights(w, probs_list, y_true)
        logs.append({"score": score, **{f"w_{m}": w[i] for i, m in enumerate(models)}})
        if score > best_score:
            best_score = score
            best_w = w.copy()

    log_df = pd.DataFrame(logs).sort_values("score", ascending=False).reset_index(drop=True)
    log_df.head(200).to_csv(out_dir / "ensemble_search_top200.csv", index=False)

    weight_df = pd.DataFrame(
        {
            "model": models,
            "weight": best_w,
        }
    ).sort_values("weight", ascending=False)
    weight_df.to_csv(out_dir / "weights.csv", index=False)
    pd.DataFrame([{"best_oof_macro_f1": best_score}]).to_csv(
        out_dir / "best_score.csv", index=False
    )

    print(f"Saved weights to: {out_dir / 'weights.csv'}")
    print(f"Best OOF Macro F1: {best_score:.6f}")
    print(weight_df.to_string(index=False))


if __name__ == "__main__":
    main()
