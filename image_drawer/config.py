"""project configurationをloadするhelper。"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

CONFIG_ENV_VAR = "IMAGE_DRAWER_CONFIG"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """TOML configurationをloadする。

    path省略時はIMAGE_DRAWER_CONFIGを使用する。
    どちらも未指定なら空configurationを返し、packageをzero-configで利用可能に保つ。
    """
    if path is None:
        configured_path = os.environ.get(CONFIG_ENV_VAR)
        if configured_path is None:
            return {}
        path = configured_path

    config_path = Path(path)
    with config_path.open("rb") as handle:
        return tomllib.load(handle)
