from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .checks import CHECKS
from .compose import compose_config
from .models import (
    PreflightError,
    PreflightIssue,
    PreflightMode,
    PreflightReport,
    PreflightRuleset,
)
from .rulesets import RULESETS


def run_preflight(
    *,
    spec: Any,
    schedule: Any,
    config_root: Path,
    env: Mapping[str, str],
) -> Any:
    issues: list[Any] = []

    # Get mode and ruleset from spec with fallback
    mode: PreflightMode = getattr(spec, "preflight_mode", "warn")
    ruleset: PreflightRuleset = getattr(spec, "preflight_ruleset", "confirm")

    # Fallback based on experiment_kind if not set
    if not mode or not ruleset:
        kind = getattr(spec, "experiment_kind", None)
        kind_str = str(kind).lower() if kind else "explore"
        if "explore" in kind_str:
            mode = mode or "warn"
            ruleset = ruleset or "local_fast"
        elif "tune" in kind_str:
            mode = mode or "warn"
            ruleset = ruleset or "confirm"
        elif "solve" in kind_str:
            mode = mode or "strict"
            ruleset = ruleset or "solve"
        else:
            mode = mode or "warn"
            ruleset = ruleset or "confirm"

    # Mode=off: skip all checks
    if mode == "off":
        return PreflightReport(passed=True, mode=mode, ruleset=ruleset, issues=[])

    check_ids = RULESETS.get(ruleset, [])

    items = getattr(schedule, "items", []) or []
    for item in items:
        run_id = getattr(item, "run_id", "unknown")
        overrides = getattr(item, "overrides", []) or []

        # Spec-level checks
        for check_id in check_ids:
            if not check_id.startswith("spec."):
                continue
            check_fn = CHECKS.get(check_id)
            if not check_fn:
                continue
            issues.extend(check_fn(overrides=overrides, run_id=run_id, env=env))

        # Cfg-level checks (need to compose config)
        cfg_level_checks = [c for c in check_ids if c.startswith("cfg.")]
        if not cfg_level_checks:
            continue

        try:
            cfg = compose_config(
                config_root=config_root,
                config_name=getattr(spec, "config_name", "experiment"),
                overrides=overrides,
            )

            # Update mode/ruleset from composed config if present
            cfg_preflight = cfg.get("preflight", {})
            mode = cfg_preflight.get("mode", mode)
            ruleset = cfg_preflight.get("ruleset", ruleset)

            for check_id in cfg_level_checks:
                check_fn = CHECKS.get(check_id)
                if not check_fn:
                    continue
                issues.extend(check_fn(cfg=cfg, run_id=run_id, env=env))
        except Exception as e:
            issues.append(
                PreflightIssue(
                    severity="error",
                    check_id="cfg.compose_success",
                    message=f"Failed to compose config: {e}",
                    run_id=run_id,
                    context={"error": str(e)},
                )
            )

    has_errors = any(i.severity == "error" for i in issues)
    passed = not has_errors

    report = PreflightReport(
        passed=passed,
        mode=mode,
        ruleset=ruleset,
        issues=issues,
    )

    if mode == "strict" and has_errors:
        details = "; ".join(f"[{i.check_id}] {i.message}" for i in issues[:3])
        raise PreflightError(f"Preflight failed with {len(issues)} issue(s): {details}")

    return report
