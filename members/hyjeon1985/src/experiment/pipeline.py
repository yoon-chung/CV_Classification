from typing import Callable

from experiment.context import RuntimeContext  # pyright: ignore[reportMissingImports]

NODES: dict[str, Callable[[RuntimeContext], None]] = {}


def register_node(name: str):
    def decorator(fn: Callable[[RuntimeContext], None]):
        NODES[name] = fn
        return fn

    return decorator


def run_pipeline(
    ctx: RuntimeContext,
    step: str = "full",
    stop_after: str | None = None,
) -> None:
    step_order = ["prep", "train", "eval", "infer", "submission", "upload"]

    if step == "full":
        steps = step_order.copy()
    else:
        steps = [step]

    for node_name in steps:
        if node_name not in NODES:
            raise ValueError(f"Unknown node: {node_name}")

        NODES[node_name](ctx)

        if stop_after == node_name:
            break
