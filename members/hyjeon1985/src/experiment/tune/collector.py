from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

_VOLATILE_OVERRIDE_PREFIXES = (
    "split.fold_index=",
    "train.seed=",
    "runner.run_id=",
    "hydra.run.dir=",
    "hydra.sweep.dir=",
    "hydra.sweep.subdir=",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_overrides(trial_dir: Path) -> list[str]:
    overrides_path = trial_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return []

    rows: list[str] = []
    for line in overrides_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        rows.append(stripped)
    return rows


def _sha1_10(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _extract_numeric(payload: dict[str, Any] | None, keys: list[str]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _extract_from_overrides(overrides: list[str], prefix: str) -> str | None:
    for item in overrides:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return None


def _discover_trial_dirs(root: Path) -> list[Path]:
    out: set[Path] = set()
    if (root / "train.json").exists():
        out.add(root)

    for path in root.rglob("train.json"):
        out.add(path.parent)

    return sorted(out)


def collect_trial_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for trial_dir in _discover_trial_dirs(root):
        train_payload = _read_json(trial_dir / "train.json")
        eval_payload = _read_json(trial_dir / "eval.json")
        prep_payload = _read_json(trial_dir / "prep.json")
        overrides = _read_overrides(trial_dir)

        if train_payload is None:
            status = "failed"
        elif eval_payload is None:
            status = "partial"
        else:
            train_status = str(train_payload.get("status", "")).lower()
            eval_status = str(eval_payload.get("status", "")).lower()
            status = (
                "success"
                if train_status == "completed" and eval_status == "completed"
                else "partial"
            )

        macro_f1 = _extract_numeric(eval_payload, ["macro_f1", "val/macro_f1"])
        val_loss = _extract_numeric(eval_payload, ["val/loss"])
        best_val_macro_f1 = _extract_numeric(train_payload, ["best_val_macro_f1"])

        overfit_gap = _extract_numeric(eval_payload, ["selection/overfit_gap"])
        if overfit_gap is None and macro_f1 is not None and best_val_macro_f1 is not None:
            overfit_gap = float(best_val_macro_f1 - macro_f1)
        elapsed_sec = _extract_numeric(train_payload, ["elapsed_sec"])

        sorted_overrides = sorted(overrides)
        overrides_hash = _sha1_10(sorted_overrides)
        candidate_overrides = [
            item
            for item in sorted_overrides
            if not item.startswith(_VOLATILE_OVERRIDE_PREFIXES)
        ]
        candidate_hash = _sha1_10(candidate_overrides)

        fold_raw = _extract_from_overrides(overrides, "split.fold_index=")
        seed_raw = _extract_from_overrides(overrides, "train.seed=")
        if fold_raw is None and isinstance(prep_payload, dict):
            split = prep_payload.get("split")
            if isinstance(split, dict) and isinstance(split.get("fold_index"), int):
                fold_raw = str(split["fold_index"])

        checkpoint = train_payload.get("checkpoint") if isinstance(train_payload, dict) else None
        last_checkpoint = (
            train_payload.get("last_checkpoint") if isinstance(train_payload, dict) else None
        )

        rows.append(
            {
                "trial_dir": str(trial_dir),
                "candidate_hash": candidate_hash,
                "overrides_hash": overrides_hash,
                "fold_index": int(fold_raw) if fold_raw is not None and fold_raw.isdigit() else "",
                "seed": int(seed_raw) if seed_raw is not None and seed_raw.isdigit() else "",
                "macro_f1": macro_f1 if macro_f1 is not None else "",
                "val_loss": val_loss if val_loss is not None else "",
                "best_val_macro_f1": best_val_macro_f1 if best_val_macro_f1 is not None else "",
                "overfit_gap": overfit_gap if overfit_gap is not None else "",
                "elapsed_sec": elapsed_sec if elapsed_sec is not None else "",
                "status": status,
                "checkpoint": str(checkpoint) if checkpoint is not None else "",
                "last_checkpoint": str(last_checkpoint) if last_checkpoint is not None else "",
            }
        )

    return rows


def write_tune_results_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "trial_dir",
        "candidate_hash",
        "overrides_hash",
        "fold_index",
        "seed",
        "macro_f1",
        "val_loss",
        "best_val_macro_f1",
        "overfit_gap",
        "elapsed_sec",
        "status",
        "checkpoint",
        "last_checkpoint",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
