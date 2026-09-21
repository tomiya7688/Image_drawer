"""Logging configuration helpers."""

from __future__ import annotations

import logging


def configure_logging(level: str | int = "INFO") -> None:
    """Configure process-wide logging with a small stable default format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
