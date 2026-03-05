"""Bootstrap environment before Hydra"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
except ImportError:
    load_dotenv = None


def load_env_bootstrap() -> None:
    """Load .env and set absolute paths"""

    # Find member root (3 levels up from this file)
    member_root = Path(__file__).resolve().parents[3]

    # Load .env if exists
    env_file = member_root / ".env"
    if env_file.exists() and load_dotenv is not None:
        load_dotenv(env_file)

    # Set ROOT_DIR if not set
    if not os.environ.get("ROOT_DIR"):
        os.environ["ROOT_DIR"] = str(member_root)

    root_dir = Path(os.environ["ROOT_DIR"])

    # Set RUNS_DIR default
    if not os.environ.get("RUNS_DIR"):
        runs_dir = root_dir / "outputs"
        os.environ["RUNS_DIR"] = str(runs_dir)
    runs_dir = Path(os.environ["RUNS_DIR"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Set LOG_DIR (RUNS_DIR 하위)
    if not os.environ.get("LOG_DIR"):
        log_dir = runs_dir / "logs"
        os.environ["LOG_DIR"] = str(log_dir)
    log_dir = Path(os.environ["LOG_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set CONFIG_DIR
    if not os.environ.get("CONFIG_DIR"):
        config_dir = root_dir / "configs"
        os.environ["CONFIG_DIR"] = str(config_dir)

    # Set DOCS_DIR
    if not os.environ.get("DOCS_DIR"):
        docs_dir = root_dir / "docs"
        os.environ["DOCS_DIR"] = str(docs_dir)

    # Set DATA_DIR
    if not os.environ.get("DATA_DIR"):
        data_dir = root_dir / "data"
        os.environ["DATA_DIR"] = str(data_dir)

    # Set CACHE_DIR
    if not os.environ.get("CACHE_DIR"):
        cache_dir = root_dir / "cache"
        os.environ["CACHE_DIR"] = str(cache_dir)
    cache_dir = Path(os.environ["CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME
    if not os.environ.get("HF_HOME"):
        hf_home = cache_dir / "huggingface"
        os.environ["HF_HOME"] = str(hf_home)
    hf_home = Path(os.environ["HF_HOME"])
    hf_home.mkdir(parents=True, exist_ok=True)

    # Set HF_HUB_CACHE
    if not os.environ.get("HF_HUB_CACHE"):
        hf_hub_cache = hf_home / "hub"
        os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)

    # Set TRANSFORMERS_CACHE (alias for HF_HUB_CACHE)
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
