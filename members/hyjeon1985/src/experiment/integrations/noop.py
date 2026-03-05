from pathlib import Path
from typing import Optional


class NoopUploadBackend:
    def upload(self, local_path: Path, remote_key: str) -> bool:
        return True

    def is_available(self) -> bool:
        return False


class NoopWandbLogger:
    def init(self, project: str, name: str, config: dict) -> None:
        return None

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        return None

    def finish(self) -> None:
        return None


class NoopNotifier:
    def send(self, message: str, level: str = "info") -> bool:
        return True

    def is_enabled(self) -> bool:
        return False
