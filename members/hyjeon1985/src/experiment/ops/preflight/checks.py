from __future__ import annotations

from typing import Any, Callable, Mapping

from pathlib import Path

from .models import PreflightIssue, PreflightRuleset

# Forbidden legacy tokens (exact match on KEY part of KEY=VALUE)
FORBIDDEN_TOKENS = [
    "stage",
    "upload.dry_run",
    "upload.enabled",
    "upload.backend",
    "overnight",
    "model_backbone",
    "preprocess_target_size",
    "optimizer_lr",
    "augmentation_preset",
    "augmentation_mix",
    "train_label_smoothing",
    "train_seed",
    "train_class_weight_mode",
]


def check_no_legacy_override_tokens(
    *, overrides: list[str], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []
    for override in overrides:
        if "=" not in override:
            continue

        key = override.split("=")[0]
        if key in FORBIDDEN_TOKENS:
            issues.append(
                PreflightIssue(
                    severity="error",
                    check_id="spec.no_legacy_override_tokens",
                    message=f"Forbidden legacy token in override: '{key}'",
                    run_id=run_id,
                    context={"override": override, "key": key},
                )
            )

        # Check wandb.artifacts.enabled=true
        if key == "wandb.artifacts.enabled":
            value = override.split("=", 1)[1]
            if value.lower() in ("true", "1"):
                issues.append(
                    PreflightIssue(
                        severity="error",
                        check_id="spec.no_legacy_override_tokens",
                        message="wandb.artifacts.enabled must not be true",
                        run_id=run_id,
                        context={"override": override},
                    )
                )

    return issues


def check_no_env_injection(
    *, overrides: list[str], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []
    for override in overrides:
        if "WANDB_MODE=" in override:
            issues.append(
                PreflightIssue(
                    severity="error",
                    check_id="spec.no_env_injection",
                    message="WANDB_MODE should not be set via override (use env var instead)",
                    run_id=run_id,
                    context={"override": override},
                )
            )
    return issues


def check_compose_success(
    *, cfg: dict[str, Any], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    # This check is implicit - if compose fails, it will raise an exception
    return []


def check_local_has_s3_disabled(
    *, cfg: dict[str, Any], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []
    scenario = cfg.get("experiment", {}).get("scenario", "")
    s3_enabled = cfg.get("upload", {}).get("s3", {}).get("enabled", False)

    if scenario == "local" and s3_enabled:
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.local_has_s3_disabled",
                message="scenario=local requires upload.s3.enabled=false",
                run_id=run_id,
                context={"scenario": scenario, "s3_enabled": s3_enabled},
            )
        )
    return issues


def check_wandb_artifacts_disabled(
    *, cfg: dict[str, Any], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []
    artifacts_enabled = cfg.get("wandb", {}).get("artifacts", {}).get("enabled", False)

    if artifacts_enabled:
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.wandb_artifacts_disabled",
                message="wandb.artifacts.enabled must be false (metadata-only policy)",
                run_id=run_id,
                context={"artifacts_enabled": artifacts_enabled},
            )
        )
    return issues


def check_wandb_mode_valid(
    *, cfg: dict[str, Any], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []
    mode = cfg.get("wandb", {}).get("mode", "disabled")
    valid_modes = ["disabled", "offline", "online"]

    if mode not in valid_modes:
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.wandb_mode_valid",
                message=f"wandb.mode must be one of {valid_modes}, got '{mode}'",
                run_id=run_id,
                context={"mode": mode, "valid_modes": valid_modes},
            )
        )
    return issues


def check_parallelism_multi_gpu_env(
    *, cfg: dict[str, Any], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []

    runner_cfg = cfg.get("runner", {}) if isinstance(cfg.get("runner"), dict) else {}
    max_concurrency = int(runner_cfg.get("max_concurrency", 1) or 1)
    if max_concurrency <= 1:
        return issues

    raw_cvd = env.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(raw_cvd, str) or not raw_cvd.strip():
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.parallelism_multi_gpu_env",
                message=(
                    "runner.max_concurrency>1 requires CUDA_VISIBLE_DEVICES to be set to a multi-GPU list "
                    "(e.g. CUDA_VISIBLE_DEVICES=0,1)"
                ),
                run_id=run_id,
                context={
                    "max_concurrency": max_concurrency,
                    "cuda_visible_devices": raw_cvd,
                },
            )
        )
        return issues

    visible_gpu_ids = [p.strip() for p in raw_cvd.split(",") if p.strip()]
    if len(visible_gpu_ids) < 2:
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.parallelism_multi_gpu_env",
                message=(
                    "runner.max_concurrency>1 requires at least 2 visible GPUs via CUDA_VISIBLE_DEVICES"
                ),
                run_id=run_id,
                context={
                    "max_concurrency": max_concurrency,
                    "cuda_visible_devices": raw_cvd,
                    "visible_gpu_ids": visible_gpu_ids,
                },
            )
        )

    return issues


def check_data_paths_exist(
    *, cfg: dict[str, Any], run_id: str, env: Mapping[str, str]
) -> list[Any]:
    issues: list[Any] = []

    dummy_data = bool(cfg.get("runner", {}).get("dummy_data", False))
    if dummy_data:
        return issues

    dataset = cfg.get("dataset", {})
    train_csv = dataset.get("train_csv")
    image_dir_train = dataset.get("image_dir_train")

    missing_keys: list[str] = []
    if not train_csv:
        missing_keys.append("dataset.train_csv")
    if not image_dir_train:
        missing_keys.append("dataset.image_dir_train")

    if missing_keys:
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.data_paths_exist",
                message=f"Missing required dataset path(s): {missing_keys}",
                run_id=run_id,
                context={"missing_keys": missing_keys},
            )
        )
        return issues

    train_csv_path = Path(str(train_csv)).expanduser()
    image_dir_train_path = Path(str(image_dir_train)).expanduser()

    if not train_csv_path.exists():
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.data_paths_exist",
                message=f"dataset.train_csv does not exist: '{train_csv_path}'",
                run_id=run_id,
                context={"key": "dataset.train_csv", "path": str(train_csv_path)},
            )
        )
    elif not train_csv_path.is_file():
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.data_paths_exist",
                message=f"dataset.train_csv is not a file: '{train_csv_path}'",
                run_id=run_id,
                context={"key": "dataset.train_csv", "path": str(train_csv_path)},
            )
        )

    if not image_dir_train_path.exists():
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.data_paths_exist",
                message=f"dataset.image_dir_train does not exist: '{image_dir_train_path}'",
                run_id=run_id,
                context={
                    "key": "dataset.image_dir_train",
                    "path": str(image_dir_train_path),
                },
            )
        )
    elif not image_dir_train_path.is_dir():
        issues.append(
            PreflightIssue(
                severity="error",
                check_id="cfg.data_paths_exist",
                message=f"dataset.image_dir_train is not a directory: '{image_dir_train_path}'",
                run_id=run_id,
                context={
                    "key": "dataset.image_dir_train",
                    "path": str(image_dir_train_path),
                },
            )
        )

    return issues


ChecksFn = Callable[..., list[Any]]

CHECKS: dict[str, ChecksFn] = {
    "spec.no_legacy_override_tokens": check_no_legacy_override_tokens,
    "spec.no_env_injection": check_no_env_injection,
    "cfg.compose_success": check_compose_success,
    "cfg.local_has_s3_disabled": check_local_has_s3_disabled,
    "cfg.wandb_artifacts_disabled": check_wandb_artifacts_disabled,
    "cfg.wandb_mode_valid": check_wandb_mode_valid,
    "cfg.parallelism_multi_gpu_env": check_parallelism_multi_gpu_env,
    "cfg.data_paths_exist": check_data_paths_exist,
}


def run_preflight_checks(
    cfg: dict[str, Any],
    schedule: dict[str, Any],
    spec: dict[str, Any],
    ruleset: PreflightRuleset,
) -> list[Any]:
    """Run preflight checks without re-composing config."""
    from .rulesets import RULESETS
    import os

    issues = []
    check_ids = RULESETS.get(ruleset, [])
    env = os.environ

    items = schedule.get("items", [])
    for item in items:
        run_id = item.get("run_id", "unknown")
        overrides = item.get("overrides", [])

        for check_id in check_ids:
            check_fn = CHECKS.get(check_id)
            if not check_fn:
                continue

            if check_id.startswith("spec."):
                issues.extend(check_fn(overrides=overrides, run_id=run_id, env=env))
            elif check_id.startswith("cfg."):
                issues.extend(check_fn(cfg=cfg, run_id=run_id, env=env))

    return issues
