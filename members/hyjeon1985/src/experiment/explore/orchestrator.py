from __future__ import annotations

import csv
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from experiment.context import RuntimeContext
from experiment.explore.executor import ExploreExecutor
from experiment.explore.metrics import (
    read_best_val_macro_f1,
    read_json,
    read_macro_f1,
    read_val_loss,
)
from experiment.explore.planner import ExploreItem, create_planner
from experiment.explore.selection import StageScore, apply_pruning, select_topk
from experiment.ops.logger import get_logger


class ExploreOrchestrator:
    def __init__(self, ctx: RuntimeContext, python_bin: str | None = None):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.run_dir = ctx.run_dir
        self.logger = get_logger(__name__)
        self.python_bin = python_bin or os.environ.get("PYTHON_BIN") or sys.executable
        env_root = os.environ.get("ROOT_DIR")
        if env_root and Path(env_root).is_absolute():
            self.root_dir = Path(env_root)
        else:
            self.root_dir = Path(__file__).resolve().parents[3]
        self.queue_id = datetime.now().strftime("date=%Y-%m-%d__ts=%H%M%S")
        self._parallelism_info: dict[str, Any] = {}
        self.executor = ExploreExecutor(
            root_dir=self.root_dir,
            python_bin=self.python_bin,
            queue_id=self.queue_id,
            job_timeout_sec=int(self.cfg.get("explore", {}).get("job_timeout_sec", 3600)),
        )

    def run(self) -> dict[str, Any]:
        self._parallelism_info = self._compute_parallelism_info()

        items = self._plan_items()
        stages = self._get_stages()
        topk = int(self.cfg.get("explore", {}).get("selection", {}).get("topk", 1))

        self._write_plan(items=items, stages=stages)
        self.ctx.notifier.send(
            f"Explore queue start | queue_id={self.queue_id} | items={len(items)}",
            level="info",
        )
        self.logger.info(
            "Explore queue start | queue_id=%s items=%s stages=%s",
            self.queue_id,
            len(items),
            len(stages),
        )

        all_records: list[dict[str, Any]] = []
        current_items = items
        stage_summaries: list[dict[str, Any]] = []

        for stage_idx, stage_cfg in enumerate(stages):
            stage_name = str(stage_cfg.get("name", f"stage{stage_idx + 1}"))
            self.ctx.notifier.send(
                f"Explore stage start | queue_id={self.queue_id} | stage={stage_name} | candidates={len(current_items)}",
                level="info",
            )
            self.logger.info(
                "Explore stage start | queue_id=%s stage=%s candidates=%s",
                self.queue_id,
                stage_name,
                len(current_items),
            )

            stage_records, stage_scores = self._run_stage(
                stage_idx=stage_idx,
                stage_name=stage_name,
                stage_cfg=stage_cfg,
                items=current_items,
            )
            all_records.extend(stage_records)

            pruned_scores = apply_pruning(
                stage_cfg=stage_cfg,
                scores=stage_scores,
                stage_records=stage_records,
            )
            promoted = select_topk(scores=pruned_scores, topk=topk)

            stage_summary = self._build_stage_summary(
                stage_name=stage_name,
                candidates=len(current_items),
                executed=len(stage_records),
                scored=len(stage_scores),
                kept_after_pruning=len(pruned_scores),
                promoted=len(promoted),
                best_score=promoted[0].score if promoted else None,
                best_item=promoted[0].item.name if promoted else None,
            )
            stage_summaries.append(stage_summary)

            topk_text = (
                ", ".join(f"{s.item.name}:{s.score:.4f}" for s in promoted) or "none"
            )
            self.ctx.notifier.send(
                f"Explore stage end | queue_id={self.queue_id} | stage={stage_name} | best={stage_summary['best_item']}:{stage_summary['best_score']} | topk={topk_text}",
                level="info",
            )
            self.logger.info(
                "Explore stage end | queue_id=%s stage=%s promoted=%s best_item=%s best_score=%s",
                self.queue_id,
                stage_name,
                len(promoted),
                stage_summary["best_item"],
                stage_summary["best_score"],
            )

            current_items = [s.item for s in promoted]
            if not current_items:
                break

        self._write_results_csv(all_records)
        summary = self._write_summary(
            items=items, records=all_records, stage_summaries=stage_summaries
        )
        return summary

    def _compute_parallelism_info(self) -> dict[str, Any]:
        runner_cfg = (
            self.cfg.get("runner", {})
            if isinstance(self.cfg.get("runner"), dict)
            else {}
        )
        runner_max_concurrency = int(runner_cfg.get("max_concurrency", 1) or 1)
        if runner_max_concurrency < 1:
            runner_max_concurrency = 1

        raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_gpu_ids: list[str] = []
        if isinstance(raw_cvd, str) and raw_cvd.strip():
            visible_gpu_ids = [p.strip() for p in raw_cvd.split(",") if p.strip()]

        gpu_count = len(visible_gpu_ids)
        soft_cap_ratio = 0.75
        soft_cap = (
            max(1, int(math.floor(gpu_count * soft_cap_ratio))) if gpu_count > 0 else 1
        )

        effective_concurrency = min(runner_max_concurrency, gpu_count or 1, soft_cap)
        effective_concurrency = max(1, int(effective_concurrency))

        clamped: dict[str, Any] = {
            "requested": runner_max_concurrency,
            "gpu_count": gpu_count,
            "soft_cap": soft_cap,
            "effective": effective_concurrency,
            "soft_cap_ratio": soft_cap_ratio,
        }
        clamp_reasons: list[str] = []
        if effective_concurrency < runner_max_concurrency:
            if gpu_count <= 1 and runner_max_concurrency > 1:
                clamp_reasons.append("no_multi_gpu_visible")
            if (
                gpu_count > 0
                and effective_concurrency < gpu_count
                and effective_concurrency == soft_cap
            ):
                clamp_reasons.append("soft_cap_75pct")
            if (
                gpu_count > 0
                and effective_concurrency == gpu_count
                and runner_max_concurrency > gpu_count
            ):
                clamp_reasons.append("gpu_count")
            if effective_concurrency == 1 and runner_max_concurrency > 1:
                clamp_reasons.append("effective_single")

        allowed_gpu_ids = (
            visible_gpu_ids[:effective_concurrency] if visible_gpu_ids else []
        )

        return {
            "runner_max_concurrency": runner_max_concurrency,
            "cuda_visible_devices": raw_cvd,
            "visible_gpu_ids": visible_gpu_ids,
            "gpu_count": gpu_count,
            "soft_cap_ratio": soft_cap_ratio,
            "soft_cap": soft_cap,
            "effective_concurrency": effective_concurrency,
            "allowed_gpu_ids": allowed_gpu_ids,
            "clamp": clamped,
            "clamp_reasons": clamp_reasons,
        }

    def _plan_items(self) -> list[ExploreItem]:
        planner = create_planner(self.cfg)
        if hasattr(planner, "plan_matrix"):
            planned = list(getattr(planner, "plan_matrix")())
            if planned:
                return planned
        if hasattr(planner, "plan_overnight"):
            planned = list(getattr(planner, "plan_overnight")())
            if planned and self.cfg.get("explore", {}).get("matrix") is None:
                return planned
        return self._plan_matrix_items()

    def _plan_matrix_items(self) -> list[ExploreItem]:
        explore_cfg = self.cfg.get("explore", {})
        matrix = explore_cfg.get("matrix", {})
        if not isinstance(matrix, dict) or not matrix:
            return []

        keys = list(matrix.keys())
        values_list = [
            matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys
        ]

        allow_mix_with_label_smoothing = bool(
            explore_cfg.get("constraints", {}).get(
                "allow_mix_with_label_smoothing", True
            )
        )
        max_jobs = int(explore_cfg.get("max_jobs", 0) or 0)

        items: list[ExploreItem] = []
        for idx, combo in enumerate(product(*values_list)):
            combo_map = dict(zip(keys, combo))
            if (
                not allow_mix_with_label_smoothing
                and str(combo_map.get("augmentation.mix", "none")) != "none"
                and float(combo_map.get("train.label_smoothing", 0.0)) > 0.0
            ):
                continue

            item_name = f"item_{idx:04d}"
            overrides = [
                f"{k}={self._serialize_override_value(v)}" for k, v in combo_map.items()
            ]
            summary = ", ".join(f"{k}={v}" for k, v in combo_map.items())
            items.append(
                ExploreItem(name=item_name, overrides=overrides, summary=summary)
            )

            if max_jobs > 0 and len(items) >= max_jobs:
                break

        return items

    def _get_stages(self) -> list[dict[str, Any]]:
        stages = self.cfg.get("explore", {}).get("stages", [])
        if not isinstance(stages, list):
            return []
        return [s for s in stages if isinstance(s, dict)]

    def _run_stage(
        self,
        *,
        stage_idx: int,
        stage_name: str,
        stage_cfg: dict[str, Any],
        items: list[ExploreItem],
    ) -> tuple[list[dict[str, Any]], list[StageScore]]:
        fold_indices = self._get_stage_folds(stage_cfg)
        stage_records: list[dict[str, Any]] = []
        score_map: dict[str, list[float]] = {}

        jobs: list[dict[str, Any]] = []
        job_idx = 0
        for item in items:
            for fold in fold_indices:
                child_name = self._child_name(
                    stage_name=stage_name, item_name=item.name, fold=fold
                )
                child_run_dir = (
                    self.run_dir / "children" / stage_name / item.name / f"fold_{fold}"
                )

                overrides = list(item.overrides)
                overrides.extend(self._stage_overrides(stage_cfg=stage_cfg, fold=fold))
                overrides.extend(
                    [
                        f"scenario={self.cfg.get('experiment', {}).get('scenario', 'local')}",
                        f"runner_profile={self.cfg.get('runner', {}).get('profile', 'local_proxy')}",
                        "kind=explore",
                        "++explore.orchestrator.enabled=false",
                        f"runner.run_id={child_name}",
                        f'hydra.run.dir="{str(child_run_dir)}"',
                    ]
                )

                parent_wandb = (
                    self.cfg.get("wandb", {})
                    if isinstance(self.cfg.get("wandb"), dict)
                    else {}
                )
                parent_mode = parent_wandb.get("mode")
                if isinstance(parent_mode, str) and parent_mode.strip():
                    overrides.append(f"wandb.mode={parent_mode}")

                parent_project = parent_wandb.get("project")
                if isinstance(parent_project, str) and parent_project.strip():
                    overrides.append(f"wandb.project={parent_project}")

                parent_entity = parent_wandb.get("entity")
                if isinstance(parent_entity, str) and parent_entity.strip():
                    overrides.append(f"wandb.entity={parent_entity}")

                parent_group = parent_wandb.get("group")
                if isinstance(parent_group, str) and parent_group.strip():
                    overrides.append(f"wandb.group={parent_group}")

                jobs.append(
                    {
                        "job_idx": job_idx,
                        "item_name": item.name,
                        "item_summary": getattr(item, "summary", ""),
                        "fold": fold,
                        "child_run_dir": child_run_dir,
                        "overrides": overrides,
                    }
                )
                job_idx += 1

        parallel = self._parallelism_info or {}
        effective_concurrency = int(parallel.get("effective_concurrency", 1) or 1)
        allowed_gpu_ids = parallel.get("allowed_gpu_ids")
        allowed_gpu_ids = allowed_gpu_ids if isinstance(allowed_gpu_ids, list) else []

        job_results: list[tuple[int, dict[str, Any], float | None]] = []

        if effective_concurrency <= 1 or not allowed_gpu_ids:
            gpu_id = allowed_gpu_ids[0] if allowed_gpu_ids else None
            for job in jobs:
                result, macro_f1 = self._run_one_job(
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    job=job,
                    gpu_id=gpu_id,
                )
                job_results.append((int(job["job_idx"]), result, macro_f1))
        else:
            buckets: list[list[dict[str, Any]]] = [
                [] for _ in range(effective_concurrency)
            ]
            for job in jobs:
                idx = int(job["job_idx"]) % effective_concurrency
                buckets[idx].append(job)

            with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
                futures = []
                for worker_idx, bucket in enumerate(buckets):
                    gpu_id = str(allowed_gpu_ids[worker_idx])
                    futures.append(
                        executor.submit(
                            self._run_job_bucket,
                            stage_idx=stage_idx,
                            stage_name=stage_name,
                            bucket=bucket,
                            gpu_id=gpu_id,
                        )
                    )

                for fut in as_completed(futures):
                    job_results.extend(fut.result())

        job_results.sort(key=lambda x: x[0])
        for _, record, macro_f1 in job_results:
            stage_records.append(record)
            if macro_f1 is not None:
                score_map.setdefault(str(record["item_name"]), []).append(macro_f1)

        item_lookup = {i.name: i for i in items}
        scores: list[StageScore] = []
        for item_name, values in score_map.items():
            if not values:
                continue
            score = sum(values) / len(values)
            scores.append(
                StageScore(
                    item=item_lookup[item_name], score=score, n_folds=len(values)
                )
            )

        scores.sort(key=lambda x: x.score, reverse=True)
        return stage_records, scores

    def _stage_overrides(self, *, stage_cfg: dict[str, Any], fold: int) -> list[str]:
        out: list[str] = []

        if stage_cfg.get("stop_after"):
            out.append(f"pipeline.stop_after={stage_cfg['stop_after']}")
        if stage_cfg.get("train_batch_size") is not None:
            out.append(f"train.batch_size={stage_cfg['train_batch_size']}")
        if stage_cfg.get("train_epochs") is not None:
            out.append(f"train.epochs={stage_cfg['train_epochs']}")

        out.append(f"split.fold_index={fold}")
        return out

    def _get_stage_folds(self, stage_cfg: dict[str, Any]) -> list[int]:
        folds = stage_cfg.get("fold_indices")
        if isinstance(folds, list) and folds:
            return [int(f) for f in folds]

        parent_fold = int(self.cfg.get("split", {}).get("fold_index", 0))
        return [parent_fold]

    def _run_one_job(
        self,
        *,
        stage_idx: int,
        stage_name: str,
        job: dict[str, Any],
        gpu_id: str | None,
    ) -> tuple[dict[str, Any], float | None]:
        child_run_dir = job["child_run_dir"]
        if not isinstance(child_run_dir, Path):
            child_run_dir = Path(str(child_run_dir))

        child_result = self.executor.run_child(
            overrides=list(job["overrides"]),
            child_run_dir=child_run_dir,
            assigned_gpu=gpu_id,
        )
        success = child_result["returncode"] == 0 and not child_result["timed_out"]
        macro_f1 = read_macro_f1(child_run_dir / "eval.json")

        record = {
            "item_name": job["item_name"],
            "item_summary": job["item_summary"],
            "stage": stage_name,
            "stage_index": stage_idx,
            "fold": job["fold"],
            "status": "success" if success else "failed",
            "macro_f1": macro_f1,
            "child_run_dir": str(child_run_dir),
            "assigned_gpu": child_result["assigned_gpu"],
            "returncode": child_result["returncode"],
            "timed_out": child_result["timed_out"],
            "elapsed_sec": child_result["elapsed_sec"],
        }

        eval_payload = read_json(child_run_dir / "eval.json")
        train_payload = read_json(child_run_dir / "train.json")

        val_loss = read_val_loss(eval_payload)
        best_val_macro_f1 = read_best_val_macro_f1(train_payload)
        overfit_gap = None
        if isinstance(best_val_macro_f1, float) and isinstance(macro_f1, float):
            overfit_gap = float(best_val_macro_f1 - macro_f1)

        record["val_loss"] = val_loss
        record["best_val_macro_f1"] = best_val_macro_f1
        record["overfit_gap"] = overfit_gap
        return record, macro_f1

    def _run_job_bucket(
        self,
        *,
        stage_idx: int,
        stage_name: str,
        bucket: list[dict[str, Any]],
        gpu_id: str,
    ) -> list[tuple[int, dict[str, Any], float | None]]:
        out: list[tuple[int, dict[str, Any], float | None]] = []
        for job in bucket:
            record, macro_f1 = self._run_one_job(
                stage_idx=stage_idx,
                stage_name=stage_name,
                job=job,
                gpu_id=gpu_id,
            )
            out.append((int(job["job_idx"]), record, macro_f1))
        return out


    def _write_plan(
        self, *, items: list[ExploreItem], stages: list[dict[str, Any]]
    ) -> None:
        payload = {
            "queue_id": self.queue_id,
            "planned_jobs": len(items),
            "items": [
                {
                    "name": i.name,
                    "summary": getattr(i, "summary", ""),
                    "overrides": i.overrides,
                }
                for i in items
            ],
            "stages": stages,
        }
        (self.run_dir / "explore_plan.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_results_csv(self, records: list[dict[str, Any]]) -> None:
        path = self.run_dir / "explore_results.csv"
        fields = [
            "item_name",
            "item_summary",
            "stage",
            "stage_index",
            "fold",
            "status",
            "macro_f1",
            "val_loss",
            "best_val_macro_f1",
            "overfit_gap",
            "child_run_dir",
            "assigned_gpu",
            "returncode",
            "timed_out",
            "elapsed_sec",
        ]
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    def _write_summary(
        self,
        *,
        items: list[ExploreItem],
        records: list[dict[str, Any]],
        stage_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        best = self._best_from_records(records)
        summary = {
            "queue_id": self.queue_id,
            "planned_items": len(items),
            "executed_children": len(records),
            "best": best,
            "stage_summaries": stage_summaries,
            "parallelism": self._parallelism_info,
        }
        (self.run_dir / "explore_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def _best_from_records(
        self, records: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        with_scores = [
            r for r in records if isinstance(r.get("macro_f1"), (int, float))
        ]
        if not with_scores:
            return None

        best = max(with_scores, key=lambda r: float(r["macro_f1"]))
        return {
            "item_name": best.get("item_name"),
            "item_summary": best.get("item_summary"),
            "macro_f1": float(best["macro_f1"]),
            "stage": best.get("stage"),
            "child_run_dir": best.get("child_run_dir"),
        }

    @staticmethod
    def _serialize_override_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _child_name(*, stage_name: str, item_name: str, fold: int) -> str:
        safe_stage = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_" for c in stage_name
        )
        safe_item = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_" for c in item_name
        )
        return f"{safe_stage}__{safe_item}__f{fold}"

    @staticmethod
    def _build_stage_summary(
        *,
        stage_name: str,
        candidates: int,
        executed: int,
        scored: int,
        kept_after_pruning: int,
        promoted: int,
        best_score: float | None,
        best_item: str | None,
    ) -> dict[str, Any]:
        return {
            "stage": stage_name,
            "candidates": candidates,
            "executed": executed,
            "scored": scored,
            "kept_after_pruning": kept_after_pruning,
            "promoted": promoted,
            "best_score": round(float(best_score), 6)
            if best_score is not None
            else None,
            "best_item": best_item,
        }


def run_orchestrator(ctx: RuntimeContext) -> dict[str, Any]:
    orchestrator = ExploreOrchestrator(ctx=ctx)
    return orchestrator.run()
