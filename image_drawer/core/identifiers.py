"""Stable record identifier helpers."""

from __future__ import annotations

from uuid import uuid4


def new_id(kind: str) -> str:
    """Create a persistable string identifier with a human-readable prefix."""
    normalized = kind.strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("kind must not be empty")
    return f"{normalized}_{uuid4().hex}"
