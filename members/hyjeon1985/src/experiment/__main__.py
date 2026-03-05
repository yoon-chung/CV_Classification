"""Entrypoint: python -m experiment"""

from experiment.runtime.env_bootstrap import load_env_bootstrap
from experiment.app import main


if __name__ == "__main__":
    # 1. Bootstrap first (load .env, set paths)
    load_env_bootstrap()
    # 2. Then run Hydra app
    main()  # pyright: ignore[reportCallIssue]
