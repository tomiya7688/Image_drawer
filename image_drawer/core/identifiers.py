"""安定したrecord IDを生成するhelper。"""

from __future__ import annotations

from uuid import uuid4


def new_id(kind: str) -> str:
    """人間が識別しやすいprefix付きの永続化可能なstring IDを生成する。"""
    normalized = kind.strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("kind must not be empty")
    return f"{normalized}_{uuid4().hex}"
