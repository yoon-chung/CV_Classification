import os
from experiment.contracts import Notifier, UploadBackend, WandbLogger
from experiment.integrations.noop import (
    NoopNotifier,
    NoopUploadBackend,
    NoopWandbLogger,
)
from experiment.integrations.slack import SlackNotifier


def create_upload_backend(spec: dict) -> UploadBackend:
    scenario = spec.get("experiment", {}).get("scenario", "local")
    if scenario == "local":
        return NoopUploadBackend()
    s3_enabled = spec.get("upload", {}).get("s3", {}).get("enabled", False)
    if s3_enabled:
        try:
            from experiment.integrations.s3 import S3UploadBackend

            return S3UploadBackend()
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for S3 upload. Install with: pip install boto3"
            ) from e
    return NoopUploadBackend()


def create_wandb_logger(spec: dict) -> WandbLogger:
    scenario = spec.get("experiment", {}).get("scenario", "local")
    if scenario == "local":
        return NoopWandbLogger()
    wandb_mode = spec.get("wandb", {}).get("mode", "disabled")
    if wandb_mode in ["online", "offline"]:
        try:
            from experiment.integrations.wandb_logger import RealWandbLogger

            return RealWandbLogger()
        except ImportError as e:
            raise RuntimeError(
                "wandb is required for W&B logging. Install with: pip install wandb"
            ) from e
    return NoopWandbLogger()


def create_notifier(spec: dict) -> Notifier:
    """Factory for notifier"""
    slack_notify = os.environ.get("SLACK_NOTIFY", "0")
    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_notify == "1" and slack_webhook:
        return SlackNotifier()
    return NoopNotifier()


__all__ = [
    "NoopNotifier",
    "NoopUploadBackend",
    "NoopWandbLogger",
    "SlackNotifier",
    "create_notifier",
    "create_upload_backend",
    "create_wandb_logger",
]
