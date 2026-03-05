from __future__ import annotations

import importlib
from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, Optional

PreflightMode = Literal["off", "warn", "strict"]
PreflightRuleset = Literal["local_fast", "confirm", "solve"]


class PreflightError(RuntimeError):
    """Raised when preflight check fails in strict mode."""


@dataclass(slots=True)
class _FallbackPreflightIssue:
    severity: Literal["error", "warn", "info"]
    check_id: str
    message: str
    run_id: Optional[str] = None
    context: dict = dc_field(default_factory=dict)


@dataclass(slots=True)
class _FallbackPreflightReport:
    passed: bool
    mode: PreflightMode
    ruleset: PreflightRuleset
    issues: list[Any]


def _build_models() -> tuple[type, type]:
    try:
        pydantic = importlib.import_module("pydantic")
    except ModuleNotFoundError:
        return _FallbackPreflightIssue, _FallbackPreflightReport

    create_model = getattr(pydantic, "create_model", None)
    BaseModel = getattr(pydantic, "BaseModel", None)
    Field = getattr(pydantic, "Field", None)
    if not (create_model and BaseModel and Field):
        return _FallbackPreflightIssue, _FallbackPreflightReport

    IssueModel = create_model(
        "PreflightIssue",
        __base__=BaseModel,
        severity=(Literal["error", "warn", "info"], ...),
        check_id=(str, ...),
        message=(str, ...),
        run_id=(Optional[str], None),
        context=(dict, Field(default_factory=dict)),
    )

    ReportModel = create_model(
        "PreflightReport",
        __base__=BaseModel,
        passed=(bool, ...),
        mode=(PreflightMode, ...),
        ruleset=(PreflightRuleset, ...),
        issues=(list[IssueModel], ...),
    )

    return IssueModel, ReportModel


PreflightIssue, PreflightReport = _build_models()
