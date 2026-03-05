"""S3 upload backend implementation"""

import os
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception

from experiment.contracts import UploadBackend


class S3UploadBackend(UploadBackend):
    """Real S3 upload backend using boto3"""

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: str = "cvdc",
        region: str = "ap-northeast-2",
    ):
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is not installed")
        self.bucket = bucket or os.environ.get("S3_BUCKET")
        self.prefix = prefix
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def upload(self, local_path: Path, remote_key: str) -> bool:
        if not self.bucket:
            return False
        full_key = f"{self.prefix}/{remote_key}" if self.prefix else remote_key
        try:
            self.client.upload_file(str(local_path), self.bucket, full_key)
            return True
        except ClientError:
            return False

    def is_available(self) -> bool:
        return self.bucket is not None
