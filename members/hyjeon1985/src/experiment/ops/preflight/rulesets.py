from .models import PreflightRuleset

# MVP rulesets
RULESETS: dict[PreflightRuleset, list[str]] = {
    "local_fast": [
        "spec.no_legacy_override_tokens",
        "spec.no_env_injection",
        "cfg.compose_success",
        "cfg.data_paths_exist",
        "cfg.parallelism_multi_gpu_env",
        "cfg.local_has_s3_disabled",
        "cfg.wandb_artifacts_disabled",
        "cfg.wandb_mode_valid",
    ],
    "confirm": [
        "spec.no_legacy_override_tokens",
        "spec.no_env_injection",
        "cfg.compose_success",
        "cfg.data_paths_exist",
        "cfg.parallelism_multi_gpu_env",
        "cfg.local_has_s3_disabled",
        "cfg.wandb_artifacts_disabled",
        "cfg.wandb_mode_valid",
    ],
    "solve": [
        "spec.no_legacy_override_tokens",
        "spec.no_env_injection",
        "cfg.compose_success",
        "cfg.data_paths_exist",
        "cfg.parallelism_multi_gpu_env",
        "cfg.local_has_s3_disabled",
        "cfg.wandb_artifacts_disabled",
        "cfg.wandb_mode_valid",
    ],
}
