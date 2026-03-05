from __future__ import annotations

import os
from pathlib import Path

import hydra  # pyright: ignore[reportMissingImports]
from hydra.core.hydra_config import HydraConfig  # pyright: ignore[reportMissingImports]
from omegaconf import DictConfig, OmegaConf  # pyright: ignore[reportMissingImports]

import experiment.nodes as _nodes  # noqa: F401  # pyright: ignore[reportMissingImports]
from experiment import integrations as integ
from experiment import spec
from experiment.context import RuntimeContext
from experiment.explore.orchestrator import run_orchestrator
from experiment.messages import (
    format_slack_explore_summary,
    format_slack_summary,
    safe_slack_text,
)
from experiment.metrics import (
    extract_explore_summary,
    extract_selection_summary,
    extract_summary,
    extract_tune_summary,
)
from experiment.ops.logger import get_logger, setup_logging
from experiment.pipeline import run_pipeline
from experiment.sanitize import sanitize_for_wandb
from experiment.solve.runner import run_solve_pipeline
from experiment.tune.lineage import apply_lineage
from experiment.tune.runner import run_tune_pipeline

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="experiment")
def main(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved config must be a dictionary")

    if resolved.get("experiment", {}).get("kind") == "tune":
        resolved = apply_lineage(resolved)

    scenario = resolved.get("experiment", {}).get("scenario", "local")
    wandb_mode = resolved.get("wandb", {}).get("mode", "disabled")

    profile = resolved.get("runner", {}).get("profile", "")
    kind = resolved.get("experiment", {}).get("kind", "")
    run_id = resolved.get("runner", {}).get("run_id", "")
    run_name = f"{profile}__{kind}__{run_id}"

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        stage=kind or "experiment",
        run_id=run_id,
        log_file=run_dir / "app.log",
    )
    logger = get_logger(__name__)
    logger.info(
        "Experiment bootstrap | scenario=%s profile=%s kind=%s run_id=%s run_dir=%s",
        scenario,
        profile,
        kind,
        run_id,
        run_dir,
    )

    notifier = integ.create_notifier(resolved)
    wandb_logger = integ.create_wandb_logger(resolved)

    wandb_cfg = (
        resolved.get("wandb", {}) if isinstance(resolved.get("wandb"), dict) else {}
    )
    logger.info(
        "W&B resolved config | mode=%s project=%s entity=%s group=%s dir=%s",
        wandb_cfg.get("mode"),
        wandb_cfg.get("project"),
        wandb_cfg.get("entity"),
        wandb_cfg.get("group"),
        wandb_cfg.get("dir"),
    )

    notifier.send(
        safe_slack_text(
            f"실험 시작 | run_name={run_name} | profile={profile} | kind={kind} | run_id={run_id}"
        ),
        level="info",
    )

    wandb_started = False
    if scenario == "cloud" and wandb_mode != "disabled":
        wandb_project = resolved.get("wandb", {}).get("project", "")
        wandb_logger.init(
            project=wandb_project,
            name=run_name,
            config=sanitize_for_wandb(resolved),
        )
        wandb_started = True

    try:
        experiment_spec = spec.from_dict(resolved)

        from experiment.ops.preflight.adapter_single_run import run_preflight_for_hydra

        overrides = HydraConfig.get().overrides.task
        run_preflight_for_hydra(resolved, overrides)

        integrations = {
            "upload_backend": integ.create_upload_backend(resolved),
            "wandb_logger": wandb_logger,
            "notifier": notifier,
            "cfg": resolved,
        }

        ctx = RuntimeContext.create(
            run_dir=run_dir, spec=experiment_spec, integrations=integrations
        )

        explore_enabled = bool(
            resolved.get("explore", {}).get("orchestrator", {}).get("enabled", True)
        )
        is_explore = experiment_spec.kind == "explore"
        is_child_process = os.environ.get("EXPERIMENT_CHILD") == "1"

        if is_explore and explore_enabled and is_child_process:
            notifier.send(
                safe_slack_text(
                    "Explore orchestrator disabled in child process | reason=EXPERIMENT_CHILD"
                ),
                level="warning",
            )

        if is_explore and explore_enabled and not is_child_process:
            logger.info("Starting explore orchestrator | run_dir=%s", run_dir)
            explore_summary = run_orchestrator(ctx)
            best = (
                explore_summary.get("best")
                if isinstance(explore_summary.get("best"), dict)
                else None
            )
            if best is None:
                queue_id = explore_summary.get("queue_id")
                notifier.send(
                    safe_slack_text(
                        " | ".join(
                            [
                                f"Explore 실패 | run_name={run_name}",
                                f"profile={profile}",
                                f"kind={kind}",
                                f"run_id={run_id}",
                                f"queue_id={queue_id}",
                                "reason=best_missing",
                            ]
                        )
                    ),
                    level="error",
                )
                raise RuntimeError(
                    f"Explore finished without best result (queue_id={queue_id})"
                )

            summary_metrics, summary_raw = extract_explore_summary(explore_summary)
            if wandb_started:
                wandb_logger.log(summary_metrics)

            notifier.send(
                safe_slack_text(
                    format_slack_explore_summary(
                        run_name=run_name,
                        run_id=run_id,
                        profile=profile,
                        kind=kind,
                        explore_summary=explore_summary,
                    )
                ),
                level="info",
            )
            logger.info(
                "Explore completed | queue_id=%s planned=%s executed=%s",
                explore_summary.get("queue_id"),
                explore_summary.get("planned_items"),
                explore_summary.get("executed_children"),
            )
        else:
            logger.info("Starting %s run | step=%s", kind, experiment_spec.pipeline.step)
            if kind == "tune":
                run_tune_pipeline(
                    ctx=ctx,
                    step=experiment_spec.pipeline.step,
                    stop_after=experiment_spec.pipeline.stop_after,
                )
            elif kind == "solve":
                run_solve_pipeline(
                    ctx=ctx,
                    step=experiment_spec.pipeline.step,
                    stop_after=experiment_spec.pipeline.stop_after,
                )
            else:
                run_pipeline(
                    ctx=ctx,
                    step=experiment_spec.pipeline.step,
                    stop_after=experiment_spec.pipeline.stop_after,
                )

            if kind == "explore":
                summary_metrics, summary_raw = extract_selection_summary(run_dir)
            elif kind == "tune":
                summary_metrics, summary_raw = extract_tune_summary(run_dir)
            else:
                summary_metrics, summary_raw = extract_summary(run_dir)
            if wandb_started:
                wandb_logger.log(summary_metrics)

            notifier.send(
                safe_slack_text(
                    format_slack_summary(
                        run_name=run_name,
                        run_id=run_id,
                        profile=profile,
                        kind=kind,
                        summary_raw=summary_raw,
                    )
                ),
                level="info",
            )
            logger.info("Pipeline completed | run_dir=%s", run_dir)

        print(f"Pipeline completed. Results in: {run_dir}")
    except Exception as e:
        err = safe_slack_text(f"{type(e).__name__}: {e}")
        logger.exception("Experiment failed | error=%s", err)
        notifier.send(
            f"실험 실패 | run_name={run_name} | profile={profile} | kind={kind} | run_id={run_id} | error={err}",
            level="error",
        )
        raise
    finally:
        if wandb_started:
            wandb_logger.finish()
            logger.info("W&B run finished")
