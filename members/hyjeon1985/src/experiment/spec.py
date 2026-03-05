from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator  # pyright: ignore[reportMissingImports]


class ExperimentKind(BaseModel):
    value: Literal["explore", "tune", "solve"] = Field(default="explore")


class RunnerProfile(BaseModel):
    value: Literal["local_proxy", "local_confirm", "hpc_proxy", "hpc_confirm"]


class ExperimentScenario(BaseModel):
    value: Literal["local", "cloud"]


class PipelineConfig(BaseModel):
    step: Literal["prep", "train", "eval", "infer", "submission", "upload", "full"] = (
        "full"
    )
    stop_after: Optional[
        Literal["prep", "train", "eval", "infer", "submission", "upload"]
    ] = None
    cache_enabled: bool = False


class ExperimentSpec(BaseModel):
    kind: str = Field(..., pattern=r"^(explore|tune|solve)$")
    runner_profile: str = Field(
        ..., pattern=r"^(local_proxy|local_confirm|hpc_proxy|hpc_confirm)$"
    )
    scenario: str = Field(..., pattern=r"^(local|cloud)$")
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @field_validator("runner_profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        if value not in ["local_proxy", "local_confirm", "hpc_proxy", "hpc_confirm"]:
            raise ValueError(f"Invalid runner.profile: {value}")
        return value


def from_dict(cfg: dict) -> ExperimentSpec:
    return ExperimentSpec(
        kind=cfg.get("experiment", {}).get("kind", "explore"),
        runner_profile=cfg.get("runner", {}).get("profile", "local_proxy"),
        scenario=cfg.get("experiment", {}).get("scenario", "local"),
        pipeline=PipelineConfig(
            step=cfg.get("pipeline", {}).get("step", "full"),
            stop_after=cfg.get("pipeline", {}).get("stop_after"),
            cache_enabled=cfg.get("pipeline", {})
            .get("cache", {})
            .get("enabled", False),
        ),
    )
