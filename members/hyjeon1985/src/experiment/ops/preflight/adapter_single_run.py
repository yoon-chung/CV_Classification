"""Single-run preflight adapter for Hydra-based execution."""

from __future__ import annotations

import os
import sys
from typing import Any

from experiment.ops.preflight.checks import CHECKS
from experiment.ops.preflight.models import (
    PreflightIssue,
    PreflightMode,
    PreflightRuleset,
)
from experiment.ops.preflight.rulesets import RULESETS


def run_preflight_for_hydra(cfg: dict[str, Any], overrides: list[str]) -> None:
    """Run preflight checks for single Hydra run.

    Args:
        cfg: Resolved Hydra config dict
        overrides: List of override strings from CLI

    Raises:
        RuntimeError: If strict mode and checks fail
    """
    # Get preflight settings from config
    preflight = cfg.get("preflight", {})
    mode: PreflightMode = preflight.get("mode", "warn")
    ruleset: PreflightRuleset = preflight.get("ruleset", "confirm")

    # Mode=off: skip all checks
    if mode == "off":
        return

    # Get checks for this ruleset
    check_ids = RULESETS.get(ruleset, [])
    if not check_ids:
        return

    run_id = cfg.get("runner", {}).get("run_id", "single")
    issues: list[Any] = []  # PreflightIssue is dynamically created

    env = dict(os.environ)

    # Run spec-level checks (check overrides)
    for check_id in check_ids:
        if not check_id.startswith("spec."):
            continue
        check_fn = CHECKS.get(check_id)
        if not check_fn:
            continue
        # Spec checks need overrides, run_id, env
        check_issues = check_fn(
            overrides=overrides,
            run_id=run_id,
            env=env,
        )
        issues.extend(check_issues)

    # Run cfg-level checks (check composed config)
    for check_id in check_ids:
        if not check_id.startswith("cfg."):
            continue
        check_fn = CHECKS.get(check_id)
        if not check_fn:
            continue
        # Cfg checks need cfg, run_id, env
        check_issues = check_fn(
            cfg=cfg,
            run_id=run_id,
            env=env,
        )
        issues.extend(check_issues)

    # Report issues
    if issues:
        print("Preflight issues found:", file=sys.stderr)
        for issue in issues:
            print(
                f"  [{issue.severity}] {issue.check_id}: {issue.message}",
                file=sys.stderr,
            )

        # Strict mode: raise error
        if mode == "strict":
            error_count = sum(1 for i in issues if i.severity == "error")
            raise RuntimeError(
                f"Preflight strict mode failed with {error_count} error(s). "
                "Fix issues or set preflight.mode=warn to continue."
            )
