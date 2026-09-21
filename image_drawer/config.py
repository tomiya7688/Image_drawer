"""Project configuration loading helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

CONFIG_ENV_VAR = "IMAGE_DRAWER_CONFIG"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load TOML configuration.

    When path is omitted, IMAGE_DRAWER_CONFIG is used. If neither is set,
    an empty configuration is returned so the package remains zero-config.
    """
    if path is None:
        configured_path = os.environ.get(CONFIG_ENV_VAR)
        if configured_path is None:
            return {}
        path = configured_path

    config_path = Path(path)
    with config_path.open("rb") as handle:
        return tomllib.load(handle)
