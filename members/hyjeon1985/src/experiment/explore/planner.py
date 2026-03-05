from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class ExploreItem:
    name: str
    overrides: list[str]
    summary: str


class ExplorePlanner:
    def __init__(self, base_config: dict):
        self.base = base_config

    def plan_grid_search(
        self,
        param_name: str,
        values: list[Any],
    ) -> Iterator[ExploreItem]:
        for i, value in enumerate(values):
            yield ExploreItem(
                name=f"{param_name}_{i}",
                overrides=[f"{param_name}={value}"],
                summary=f"{param_name}={value}",
            )

    def plan_overnight(self) -> list[ExploreItem]:
        return self.plan_matrix()

    def plan_matrix(self) -> list[ExploreItem]:
        explore_cfg = self.base.get("explore", {})
        matrix = explore_cfg.get("matrix", {})
        if not isinstance(matrix, dict) or not matrix:
            return []

        matrix_keys = sorted(str(k) for k in matrix.keys())
        matrix_values: list[list[Any]] = []
        for k in matrix_keys:
            raw = matrix.get(k)
            if not isinstance(raw, list) or len(raw) == 0:
                raise ValueError(f"explore.matrix['{k}'] must be a non-empty list")
            matrix_values.append(sorted(raw, key=_stable_value_sort_key))

        allow_mix_with_ls = bool(
            explore_cfg.get("constraints", {}).get(
                "allow_mix_with_label_smoothing", True
            )
        )
        max_jobs = int(explore_cfg.get("max_jobs", 0) or 0)
        if max_jobs < 0:
            raise ValueError("explore.max_jobs must be >= 0")

        items: list[ExploreItem] = []
        for values in itertools.product(*matrix_values):
            combo = dict(zip(matrix_keys, values, strict=True))
            if not _passes_constraints(
                combo=combo,
                base_cfg=self.base,
                allow_mix_with_label_smoothing=allow_mix_with_ls,
            ):
                continue

            overrides = [f"{k}={_format_override_value(combo[k])}" for k in matrix_keys]
            items.append(
                ExploreItem(
                    name=_default_item_name(overrides),
                    overrides=overrides,
                    summary=_format_summary(combo, key_order=matrix_keys),
                )
            )

            if max_jobs > 0 and len(items) >= max_jobs:
                break

        return items


def create_planner(spec: dict) -> ExplorePlanner:
    return ExplorePlanner(base_config=spec)


def _stable_value_sort_key(v: Any) -> tuple[str, str]:
    return (type(v).__name__, repr(v))


def _get_dotpath(cfg: dict, key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _passes_constraints(
    *,
    combo: dict[str, Any],
    base_cfg: dict,
    allow_mix_with_label_smoothing: bool,
) -> bool:
    if allow_mix_with_label_smoothing:
        return True

    mix = combo.get("augmentation.mix")
    if mix is None:
        mix = _get_dotpath(base_cfg, "augmentation.mix")

    ls = combo.get("train.label_smoothing")
    if ls is None:
        ls = _get_dotpath(base_cfg, "train.label_smoothing")

    mix_str = str(mix) if mix is not None else "none"
    try:
        ls_float = float(ls) if ls is not None else 0.0
    except Exception:
        ls_float = 0.0

    if mix_str != "none" and ls_float > 0.0:
        return False

    return True


def _format_override_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return repr(v)
    s = str(v)
    if any(c in s for c in [" ", ",", "[", "]", "{", "}"]):
        return repr(s)
    return s


def _default_item_name(overrides: list[str]) -> str:
    payload = "\n".join(overrides).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:10]
    return f"m_{h}"


def _format_summary(combo: dict[str, Any], *, key_order: list[str]) -> str:
    parts: list[str] = []
    for k in key_order:
        parts.append(f"{k}={_format_override_value(combo.get(k))}")
    return ", ".join(parts)
