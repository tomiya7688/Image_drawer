"""prompt group単位のdeterministic split。"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from image_drawer.training.models import SplitConfig


_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalize_prompt_group(prompt: str) -> str:
    """表記揺れで明らかな近似promptを同じgroupへ寄せる。"""
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    normalized = _NON_WORD.sub(" ", normalized)
    return " ".join(normalized.split())


def prompt_group_id(prompt: str) -> str:
    normalized = normalize_prompt_group(prompt)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"prompt-group-{digest[:20]}"


def assign_split(group_id: str, config: SplitConfig) -> str:
    payload = f"{config.seed}:{group_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    fraction = value / float(1 << 64)
    train_boundary = float(config.train)
    validation_boundary = train_boundary + float(config.validation)
    if fraction < train_boundary:
        return "train"
    if fraction < validation_boundary:
        return "validation"
    return "test"
