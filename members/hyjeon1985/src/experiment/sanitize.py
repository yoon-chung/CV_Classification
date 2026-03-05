from __future__ import annotations

import re
from typing import Any


_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|webhook|access[_-]?key|secret[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)


def sanitize_for_wandb(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _SENSITIVE_KEY_RE.search(key):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = sanitize_for_wandb(v)
        return sanitized

    if isinstance(value, list):
        return [sanitize_for_wandb(v) for v in value]

    if isinstance(value, tuple):
        return tuple(sanitize_for_wandb(v) for v in value)

    return value
