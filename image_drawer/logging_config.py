"""logging configuration helper。"""

from __future__ import annotations

import logging


def configure_logging(level: str | int = "INFO") -> None:
    """小さく安定したdefault formatでprocess-wide loggingを設定する。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
