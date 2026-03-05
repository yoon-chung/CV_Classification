from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    level: str | int = "INFO",
    *,
    stage: str | None = None,
    snapshot_id: str | None = None,
    run_id: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    stage = stage or os.getenv("STAGE", "-")
    snapshot_id = snapshot_id or os.getenv("SNAPSHOT_ID", "-")
    run_id = run_id or os.getenv("RUN_ID", "-")

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        record.stage = stage
        record.snapshot_id = snapshot_id
        record.run_id = run_id
        return record

    logging.setLogRecordFactory(record_factory)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(stage)s | %(run_id)s | %(message)s"
    )

    handlers: list[logging.Handler] = []
    stream_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    handlers.append(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def log_json(
    logger: logging.Logger,
    message: str,
    payload: dict[str, Any],
    level: str = "info",
) -> None:
    text = f"{message} | payload={json.dumps(payload, ensure_ascii=False)}"
    log_func = getattr(logger, level, logger.info)
    log_func(text)
