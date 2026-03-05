from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class RunMeta:
    run_dir: Path
    run_id: str
    candidate_name: str
    candidate_hash: str
    backbone_profile: str
    fold_index: str


def _parse_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _load_run_meta(run_dir: Path) -> RunMeta:
    overrides = _parse_overrides(run_dir / ".hydra" / "overrides.yaml")
    run_id = run_dir.name.replace("run_id=", "", 1)
    return RunMeta(
        run_dir=run_dir,
        run_id=run_id,
        candidate_name=str(overrides.get("solve.candidate_name", "")),
        candidate_hash=str(overrides.get("solve.candidate_hash", "")),
        backbone_profile=str(overrides.get("solve.backbone_profile", "")),
        fold_index=str(overrides.get("split.fold_index", "")),
    )


def _id_signature(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _load_val_predictions(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "artifacts" / "eval" / "val_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = ["ID", "target_true", "target_pred"]
    if not set(required).issubset(set(df.columns)):
        return None
    out = df[required].copy()
    out["ID"] = out["ID"].astype(str)
    out = out.rename(columns={"target_true": "target_true", "target_pred": "pred"})
    out["pred"] = out["pred"].astype(int)
    out["target_true"] = out["target_true"].astype(int)
    return out


def _load_test_predictions(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"ID", "target"}
    if not required.issubset(set(df.columns)):
        return None
    out = df[["ID", "target"]].copy()
    out["ID"] = out["ID"].astype(str)
    out["target"] = out["target"].astype(int)
    out = out.rename(columns={"target": "pred"})
    return out


def _pairwise_rows(
    *,
    data_map: dict[str, tuple[RunMeta, pd.DataFrame]],
    with_truth: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[RunMeta, pd.DataFrame]]] = {}
    for run_key, (meta, df) in data_map.items():
        sig = _id_signature(df["ID"].astype(str).tolist())
        grouped.setdefault(sig, []).append((meta, df))

    rows: list[dict[str, Any]] = []
    for id_sig, items in grouped.items():
        if len(items) < 2:
            continue
        for (meta_a, df_a), (meta_b, df_b) in combinations(items, 2):
            merged = df_a.merge(
                df_b,
                on="ID",
                how="inner",
                suffixes=("_a", "_b"),
            )
            n_common = int(len(merged))
            if n_common <= 0:
                continue

            pred_a = merged["pred_a"]
            pred_b = merged["pred_b"]
            disagreement_rate = float((pred_a != pred_b).mean())
            row: dict[str, Any] = {
                "id_signature": id_sig,
                "n_common": n_common,
                "run_a": meta_a.run_id,
                "run_b": meta_b.run_id,
                "candidate_name_a": meta_a.candidate_name,
                "candidate_name_b": meta_b.candidate_name,
                "candidate_hash_a": meta_a.candidate_hash,
                "candidate_hash_b": meta_b.candidate_hash,
                "backbone_a": meta_a.backbone_profile,
                "backbone_b": meta_b.backbone_profile,
                "fold_index_a": meta_a.fold_index,
                "fold_index_b": meta_b.fold_index,
                "disagreement_rate": disagreement_rate,
            }

            if with_truth and "target_true_a" in merged.columns and "target_true_b" in merged.columns:
                true_a = merged["target_true_a"].astype(int)
                true_b = merged["target_true_b"].astype(int)
                if not true_a.equals(true_b):
                    continue
                target_true = true_a
                error_a = pred_a != target_true
                error_b = pred_b != target_true
                acc_a = float((pred_a == target_true).mean())
                acc_b = float((pred_b == target_true).mean())
                both_wrong = int((error_a & error_b).sum())
                union_wrong = int((error_a | error_b).sum())
                error_overlap_jaccard = (
                    float(both_wrong / union_wrong) if union_wrong > 0 else 0.0
                )
                oracle_acc = float(((pred_a == target_true) | (pred_b == target_true)).mean())
                best_single_acc = max(acc_a, acc_b)
                oracle_gain = float(oracle_acc - best_single_acc)
                complementarity_score = float(disagreement_rate * (1.0 - error_overlap_jaccard))

                row.update(
                    {
                        "acc_a": acc_a,
                        "acc_b": acc_b,
                        "both_wrong_count": both_wrong,
                        "union_wrong_count": union_wrong,
                        "error_overlap_jaccard": error_overlap_jaccard,
                        "oracle_acc": oracle_acc,
                        "oracle_gain_vs_best": oracle_gain,
                        "complementarity_score": complementarity_score,
                    }
                )
            rows.append(row)

    rows.sort(
        key=lambda r: (
            -float(r.get("complementarity_score", 0.0)),
            -float(r.get("oracle_gain_vs_best", 0.0)),
            -float(r.get("disagreement_rate", 0.0)),
            str(r.get("run_a", "")),
            str(r.get("run_b", "")),
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _aggregate_run_scores(
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[float]]] = {}

    def _push(run_id: str, key: str, value: float) -> None:
        buckets.setdefault(run_id, {})
        buckets[run_id].setdefault(key, [])
        buckets[run_id][key].append(float(value))

    for row in val_rows:
        for side in ("a", "b"):
            run_id = str(row.get(f"run_{side}", ""))
            if not run_id:
                continue
            if isinstance(row.get("disagreement_rate"), (int, float)):
                _push(run_id, "val_disagreement_mean", float(row["disagreement_rate"]))
            if isinstance(row.get("error_overlap_jaccard"), (int, float)):
                _push(run_id, "val_error_overlap_jaccard_mean", float(row["error_overlap_jaccard"]))
            if isinstance(row.get("oracle_gain_vs_best"), (int, float)):
                _push(run_id, "val_oracle_gain_mean", float(row["oracle_gain_vs_best"]))
            if isinstance(row.get("complementarity_score"), (int, float)):
                _push(run_id, "val_complementarity_mean", float(row["complementarity_score"]))

    for row in test_rows:
        for side in ("a", "b"):
            run_id = str(row.get(f"run_{side}", ""))
            if not run_id:
                continue
            if isinstance(row.get("disagreement_rate"), (int, float)):
                _push(run_id, "test_disagreement_mean", float(row["disagreement_rate"]))

    out: list[dict[str, Any]] = []
    for run_id, metrics in buckets.items():
        row: dict[str, Any] = {"run_id": run_id}
        for key, values in metrics.items():
            if values:
                row[key] = float(sum(values) / len(values))
                row[f"{key}_n"] = int(len(values))
        out.append(row)
    out.sort(key=lambda r: str(r.get("run_id", "")))
    return out


def analyze_solve_runs(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    val_map: dict[str, tuple[RunMeta, pd.DataFrame]] = {}
    test_map: dict[str, tuple[RunMeta, pd.DataFrame]] = {}
    included_runs: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        meta = _load_run_meta(run_dir)
        val_df = _load_val_predictions(run_dir)
        test_df = _load_test_predictions(run_dir)
        included_runs.append(
            {
                "run_id": meta.run_id,
                "run_dir": str(meta.run_dir),
                "candidate_name": meta.candidate_name,
                "candidate_hash": meta.candidate_hash,
                "backbone_profile": meta.backbone_profile,
                "fold_index": meta.fold_index,
                "has_val_predictions": bool(val_df is not None),
                "has_test_predictions": bool(test_df is not None),
            }
        )
        if val_df is not None:
            val_map[meta.run_id] = (meta, val_df)
        if test_df is not None:
            test_map[meta.run_id] = (meta, test_df)

    val_rows = _pairwise_rows(data_map=val_map, with_truth=True)
    test_rows = _pairwise_rows(data_map=test_map, with_truth=False)
    run_scores = _aggregate_run_scores(val_rows, test_rows)

    val_pairs_path = out_dir / "ensemble_val_pairs.csv"
    test_pairs_path = out_dir / "ensemble_test_pairs.csv"
    run_scores_path = out_dir / "ensemble_run_scores.csv"
    summary_path = out_dir / "ensemble_summary.json"
    manifest_path = out_dir / "ensemble_runs.json"

    _write_csv(val_pairs_path, val_rows)
    _write_csv(test_pairs_path, test_rows)
    _write_csv(run_scores_path, run_scores)
    manifest_path.write_text(
        json.dumps(included_runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "n_input_runs": len(run_dirs),
        "n_val_runs": len(val_map),
        "n_test_runs": len(test_map),
        "n_val_pairs": len(val_rows),
        "n_test_pairs": len(test_rows),
        "top_val_pair_by_complementarity": val_rows[0] if val_rows else None,
        "top_val_pair_by_oracle_gain": (
            max(
                (
                    row
                    for row in val_rows
                    if isinstance(row.get("oracle_gain_vs_best"), (int, float))
                ),
                key=lambda r: float(r.get("oracle_gain_vs_best", 0.0)),
                default=None,
            )
        ),
        "paths": {
            "val_pairs_csv": str(val_pairs_path),
            "test_pairs_csv": str(test_pairs_path),
            "run_scores_csv": str(run_scores_path),
            "runs_json": str(manifest_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise ensemble-complementarity metrics for solve runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Solve run directory path (repeat for multiple runs).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write pairwise metrics CSV/JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dir]
    out_dir = Path(args.out_dir).resolve()
    summary = analyze_solve_runs(run_dirs=run_dirs, out_dir=out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
