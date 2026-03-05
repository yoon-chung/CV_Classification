import json
from pathlib import Path

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]


def save_node_result(ctx: RuntimeContext, node_name: str, result: dict) -> Path:
    result_path = ctx.run_dir / f"{node_name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


def load_node_result(ctx: RuntimeContext, node_name: str) -> dict | None:
    result_path = ctx.run_dir / f"{node_name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    return None
