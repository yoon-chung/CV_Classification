from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from experiment.solve.submission_writer import write_submission_csv


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {path}")
    df = pd.read_csv(path)
    required = {"ID", "target"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"predictions.csv must contain {required}, got {set(df.columns)}")
    out = df[["ID", "target"]].copy()
    out["ID"] = out["ID"].astype(str)
    out["target"] = out["target"].astype(int)
    dup_count = int(out["ID"].duplicated().sum())
    if dup_count > 0:
        raise ValueError(f"Duplicate IDs found in {path}: {dup_count}")
    return out


def _load_probabilities(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "artifacts" / "infer" / "predictions_proba.npz"
    if not path.exists():
        raise FileNotFoundError(f"predictions_proba.npz not found: {path}")

    try:
        with np.load(path, allow_pickle=False) as payload:
            ids = np.asarray(payload["ids"]).astype(str)
            probs = np.asarray(payload["probs"], dtype=np.float64)
            labels = np.asarray(payload["labels"], dtype=np.int64)
    except Exception as exc:
        raise RuntimeError(f"Failed to load probability payload: {path}") from exc

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D in {path}, got shape={probs.shape}")
    if ids.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"ids/labels must be 1D in {path}")
    if probs.shape[0] != ids.shape[0]:
        raise ValueError(
            f"Row mismatch in {path}: probs rows={probs.shape[0]}, ids={ids.shape[0]}"
        )
    if probs.shape[1] != labels.shape[0]:
        raise ValueError(
            f"Column mismatch in {path}: probs cols={probs.shape[1]}, labels={labels.shape[0]}"
        )
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise ValueError(f"Duplicate ids found in {path}")
    if np.unique(labels).shape[0] != labels.shape[0]:
        raise ValueError(f"Duplicate labels found in {path}")

    return ids, probs, labels


def _resolve_weights(
    *,
    models: dict[str, Path],
    explicit_weights: dict[str, float],
    weight_mode: str,
) -> dict[str, float]:
    if explicit_weights:
        out: dict[str, float] = {}
        for key in models:
            out[key] = float(explicit_weights.get(key, 1.0))
        return out

    if weight_mode == "uniform":
        return {key: 1.0 for key in models}

    if weight_mode != "val_macro_f1":
        raise ValueError(f"Unsupported weight mode: {weight_mode}")

    scores: dict[str, float] = {}
    for key, run_dir in models.items():
        eval_json = _read_json(run_dir / "eval.json")
        score = eval_json.get("macro_f1")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            score = eval_json.get("val/macro_f1")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            score = 1.0
        scores[key] = float(score)

    min_floor = 1e-8
    return {key: max(min_floor, value) for key, value in scores.items()}


def _soft_vote(
    *,
    probability_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    keys: list[str],
    weights: dict[str, float],
    ordered_ids: list[str],
) -> pd.DataFrame:
    if not keys:
        raise ValueError("keys must not be empty")

    for key in keys:
        if key not in probability_payloads:
            raise KeyError(f"Unknown model key: {key}")

    global_labels = sorted(
        {int(label) for key in keys for label in probability_payloads[key][2].tolist()}
    )
    if not global_labels:
        raise RuntimeError("No labels found in probability payloads")

    label_to_idx = {label: idx for idx, label in enumerate(global_labels)}
    total_probs = np.zeros((len(ordered_ids), len(global_labels)), dtype=np.float64)

    for key in keys:
        ids, probs, labels = probability_payloads[key]
        id_to_idx = {str(image_id): idx for idx, image_id in enumerate(ids.tolist())}
        missing_ids = [image_id for image_id in ordered_ids if image_id not in id_to_idx]
        if missing_ids:
            raise RuntimeError(
                f"Missing IDs in probability payload for key='{key}': {len(missing_ids)}"
            )

        take = np.asarray([id_to_idx[image_id] for image_id in ordered_ids], dtype=np.int64)
        aligned = probs[take]
        local_cols = np.asarray([label_to_idx[int(label)] for label in labels], dtype=np.int64)

        expanded = np.zeros_like(total_probs)
        expanded[:, local_cols] = aligned
        total_probs += float(weights.get(key, 1.0)) * expanded

    label_array = np.asarray(global_labels, dtype=np.int64)
    pred_idx = np.argmax(total_probs, axis=1)
    pred_labels = label_array[pred_idx]
    return pd.DataFrame({"ID": ordered_ids, "target": pred_labels.astype(int)})


def build_final_submissions(
    *,
    models: dict[str, Path],
    out_dir: Path,
    single_key: str,
    pair_keys: list[str],
    triple_keys: list[str],
    output_names: dict[str, str],
    weight_mode: str,
    explicit_weights: dict[str, float],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames = {key: _load_predictions(path) for key, path in models.items()}
    if single_key not in prediction_frames:
        raise KeyError(f"single_key not found: {single_key}")

    ordered_ids = prediction_frames[single_key]["ID"].astype(str).tolist()
    ref_ids = set(ordered_ids)
    for key in prediction_frames:
        cur_ids = set(prediction_frames[key]["ID"].astype(str).tolist())
        if cur_ids != ref_ids:
            missing = len(ref_ids - cur_ids)
            extra = len(cur_ids - ref_ids)
            raise RuntimeError(
                f"ID set mismatch between '{single_key}' and '{key}' (missing={missing}, extra={extra})"
            )

    probability_payloads = {key: _load_probabilities(path) for key, path in models.items()}
    for key, (ids, _, _) in probability_payloads.items():
        cur_ids = set(ids.tolist())
        if cur_ids != ref_ids:
            missing = len(ref_ids - cur_ids)
            extra = len(cur_ids - ref_ids)
            raise RuntimeError(
                f"Probability ID set mismatch between '{single_key}' and '{key}' (missing={missing}, extra={extra})"
            )

    weights = _resolve_weights(
        models=models, explicit_weights=explicit_weights, weight_mode=weight_mode
    )

    single_df = prediction_frames[single_key][["ID", "target"]].copy()
    pair_soft_df = _soft_vote(
        probability_payloads=probability_payloads,
        keys=pair_keys,
        weights=weights,
        ordered_ids=ordered_ids,
    )
    triple_soft_df = _soft_vote(
        probability_payloads=probability_payloads,
        keys=triple_keys,
        weights=weights,
        ordered_ids=ordered_ids,
    )

    single_name = output_names.get("single", "submission_1_single.csv")
    pair_soft_name = output_names.get(
        "pair_soft", "submission_2_pair_soft_ensemble.csv"
    )
    triple_soft_name = output_names.get(
        "triple_soft", "submission_3_triple_soft_ensemble.csv"
    )

    single_path = write_submission_csv(single_df, out_dir / single_name)
    pair_soft_path = write_submission_csv(pair_soft_df, out_dir / pair_soft_name)
    triple_soft_path = write_submission_csv(
        triple_soft_df, out_dir / triple_soft_name
    )

    summary = {
        "ensemble_method": "soft_vote_weighted",
        "weight_mode": weight_mode,
        "weights": weights,
        "models": {k: str(v) for k, v in models.items()},
        "single_key": single_key,
        "pair_keys": pair_keys,
        "triple_keys": triple_keys,
        "outputs": {
            "single": str(single_path),
            "pair_soft": str(pair_soft_path),
            "triple_soft": str(triple_soft_path),
        },
    }
    summary_path = out_dir / "final_submission_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _parse_key_value(arg: str, *, kind: str) -> tuple[str, str]:
    if "=" not in arg:
        raise ValueError(f"{kind} must be KEY=VALUE, got: {arg}")
    key, value = arg.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"{kind} must be KEY=VALUE, got: {arg}")
    return key, value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final submission files (single/pair_soft/triple_soft) from solve run outputs."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model mapping in KEY=RUN_DIR form. Repeat for each model.",
    )
    parser.add_argument(
        "--single-key",
        required=True,
        help="Model key to use for single submission.",
    )
    parser.add_argument(
        "--pair-keys",
        nargs=2,
        required=True,
        help="Two model keys for pair ensemble.",
    )
    parser.add_argument(
        "--triple-keys",
        nargs=3,
        required=True,
        help="Three model keys for triple ensemble.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for final submission CSV files.",
    )
    parser.add_argument(
        "--output-single",
        default="submission_1_single.csv",
        help="Filename for single submission output.",
    )
    parser.add_argument(
        "--output-pair-soft",
        default="submission_2_pair_soft_ensemble.csv",
        help="Filename for pair soft-vote submission output.",
    )
    parser.add_argument(
        "--output-triple-soft",
        default="submission_3_triple_soft_ensemble.csv",
        help="Filename for triple soft-vote submission output.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["uniform", "val_macro_f1"],
        default="val_macro_f1",
        help="How to derive voting weights when --weight is not provided.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Explicit weight in KEY=FLOAT form. Optional.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    models: dict[str, Path] = {}
    for item in args.model:
        key, value = _parse_key_value(item, kind="--model")
        models[key] = Path(value).resolve()

    explicit_weights: dict[str, float] = {}
    for item in args.weight:
        key, value = _parse_key_value(item, kind="--weight")
        explicit_weights[key] = float(value)

    summary = build_final_submissions(
        models=models,
        out_dir=Path(args.out_dir).resolve(),
        single_key=str(args.single_key),
        pair_keys=[str(v) for v in args.pair_keys],
        triple_keys=[str(v) for v in args.triple_keys],
        output_names={
            "single": str(args.output_single),
            "pair_soft": str(args.output_pair_soft),
            "triple_soft": str(args.output_triple_soft),
        },
        weight_mode=str(args.weight_mode),
        explicit_weights=explicit_weights,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
