"""Slack notifier implementation"""

import os
from typing import Optional
import json
import urllib.request
import urllib.error

from experiment.contracts import Notifier


class SlackNotifier(Notifier):
    """Real Slack notifier using webhook"""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def is_enabled(self) -> bool:
        """Check if Slack is configured"""
        return self.webhook_url is not None and os.environ.get("SLACK_NOTIFY") == "1"

    def send(self, message: str, level: str = "info") -> bool:
        """Send message to Slack"""
        if not self.is_enabled() or self.webhook_url is None:
            return False

        webhook_url: str = self.webhook_url

        # Color based on level
        colors = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
        }

        payload = {
            "attachments": [
                {
                    "color": colors.get(level, "#36a64f"),
                    "text": message,
                    "footer": "CV Document Classification",
                }
            ]
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False
        except Exception:
            return False
