#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_overrides(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        lines.append(text)
    return lines


def _collect_run_record(run_dir: Path) -> dict[str, Any]:
    run_id = run_dir.name
    prep = _load_json(run_dir / "prep.json")
    train = _load_json(run_dir / "train.json")
    eval_payload = _load_json(run_dir / "eval.json")
    infer = _load_json(run_dir / "infer.json")
    overrides = _load_overrides(run_dir / "hydra" / "overrides.yaml")

    split = prep.get("split", {}) if isinstance(prep.get("split"), dict) else {}
    return {
        "run_id": run_id,
        "split_strategy": split.get("strategy", ""),
        "fold_index": split.get("fold_index", ""),
        "split_seed": split.get("seed", ""),
        "n_train": prep.get("n_train", ""),
        "n_val": prep.get("n_val", ""),
        "n_test": prep.get("n_test", ""),
        "train_seed": train.get("train.seed", ""),
        "epochs_completed": train.get("epochs_completed", ""),
        "best_epoch": train.get("best_epoch", ""),
        "best_val_macro_f1": train.get("best/val_macro_f1", ""),
        "early_stopped": train.get("early_stopped", ""),
        "macro_f1": eval_payload.get("macro_f1", ""),
        "val_loss": eval_payload.get("val/loss", ""),
        "overfit_gap": eval_payload.get("selection/overfit_gap", ""),
        "error_rate": eval_payload.get("selection/error_rate", ""),
        "class_f1_std": eval_payload.get("selection/class_f1_std", ""),
        "high_conf_wrong_rate": eval_payload.get("selection/high_conf_wrong_rate", ""),
        "confidence_mean": eval_payload.get("selection/confidence_mean", ""),
        "low_margin_rate": eval_payload.get("selection/low_margin_rate", ""),
        "tta_enabled": infer.get("infer/tta_enabled", ""),
        "tta_views": infer.get("infer/tta_views", ""),
        "tta_cache_enabled": infer.get("infer/tta_cache_enabled", ""),
        "tta_cache_hit": infer.get("infer/tta_cache_hit", ""),
        "hydra_overrides": " | ".join(overrides),
        "payloads": {
            "prep": prep,
            "train": train,
            "eval": eval_payload,
            "infer": infer,
            "overrides": overrides,
        },
    }


def _write_run_catalog(run_root: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = [
        "run_id",
        "split_strategy",
        "fold_index",
        "split_seed",
        "n_train",
        "n_val",
        "n_test",
        "train_seed",
        "epochs_completed",
        "best_epoch",
        "best_val_macro_f1",
        "early_stopped",
        "macro_f1",
        "val_loss",
        "overfit_gap",
        "error_rate",
        "class_f1_std",
        "high_conf_wrong_rate",
        "confidence_mean",
        "low_margin_rate",
        "tta_enabled",
        "tta_views",
        "tta_cache_enabled",
        "tta_cache_hit",
        "hydra_overrides",
    ]
    out_csv = run_root / "run_catalog.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    out_jsonl = run_root / "run_payloads.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for row in records:
            payload = {
                "run_id": row.get("run_id", ""),
                "payloads": row.get("payloads", {}),
            }
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _move_children_flat(run_root: Path, child_name: str) -> None:
    child = run_root / child_name
    if not child.exists() or not child.is_dir():
        return
    for src in sorted(child.iterdir()):
        if not src.is_file():
            continue
        dst = run_root / src.name
        if dst.exists():
            dst = run_root / f"{child_name}__{src.name}"
        shutil.move(str(src), str(dst))
    try:
        child.rmdir()
    except OSError:
        # Residual files/dirs are intentionally left as-is.
        pass


def _flatten_run_archive(run_root: Path, dry_run: bool) -> None:
    runs_dir = run_root / "runs"
    records: list[dict[str, Any]] = []
    if runs_dir.exists() and runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                records.append(_collect_run_record(run_dir))

    if dry_run:
        print(f"[dry-run] {run_root} records={len(records)}")
        return

    _write_run_catalog(run_root, records)
    if runs_dir.exists() and runs_dir.is_dir():
        shutil.rmtree(runs_dir)

    _move_children_flat(run_root, "ensemble")
    _move_children_flat(run_root, "final_submissions")


def _iter_run_roots(solve_root: Path) -> list[Path]:
    if not solve_root.exists():
        return []
    out: list[Path] = []
    for kind_dir in sorted(solve_root.iterdir()):
        if not kind_dir.is_dir():
            continue
        if kind_dir.name not in {"solve-team", "solve-team-final"}:
            continue
        for run_root in sorted(kind_dir.iterdir()):
            if run_root.is_dir() and run_root.name.startswith("run_id="):
                out.append(run_root)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten solve archive run folders to minimal reproducibility payloads."
    )
    parser.add_argument(
        "--root",
        default="members/hyjeon1985/archive",
        help="Archive root path.",
    )
    parser.add_argument(
        "--date",
        default="2026-03-03",
        help="Archive date tag (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print targets without modifying files.",
    )
    args = parser.parse_args()

    solve_root = Path(args.root) / f"date={args.date}" / "solve"
    run_roots = _iter_run_roots(solve_root)
    if not run_roots:
        print(f"No run roots found under {solve_root}")
        return

    for run_root in run_roots:
        _flatten_run_archive(run_root=run_root, dry_run=bool(args.dry_run))
        print(f"[ok] {run_root}")


if __name__ == "__main__":
    main()
