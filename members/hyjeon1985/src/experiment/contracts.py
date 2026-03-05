from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class UploadBackend(Protocol):
    def upload(self, local_path: Path, remote_key: str) -> bool: ...

    def is_available(self) -> bool: ...


@runtime_checkable
class WandbLogger(Protocol):
    def init(self, project: str, name: str, config: dict) -> None: ...

    def log(self, metrics: dict, step: Optional[int] = None) -> None: ...

    def finish(self) -> None: ...


@runtime_checkable
class Notifier(Protocol):
    def send(self, message: str, level: str = "info") -> bool: ...

    def is_enabled(self) -> bool: ...
