"""Part embedding abstractions and a deterministic local baseline."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from image_drawer.core import Part

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    family: str
    model: str
    version: str
    dimensions: int

    @property
    def key(self) -> str:
        return f"{self.family}:{self.model}:{self.version}:{self.dimensions}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model": self.model,
            "version": self.version,
            "dimensions": self.dimensions,
            "key": self.key,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    part_id: str
    identity: EmbeddingIdentity
    vector: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class PartEmbedder(Protocol):
    """Cross-query/Part embedding contract used by retrieval."""

    identity: EmbeddingIdentity

    def embed_text(self, text: str) -> Sequence[float]:
        ...

    def embed_part(self, part: Part) -> Sequence[float]:
        ...


def _normalize(vector: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        output: list[str] = []
        for key in sorted(value, key=str):
            output.append(str(key))
            output.extend(_flatten_text(value[key]))
        return output
    if isinstance(value, (list, tuple, set)):
        output = []
        for item in value:
            output.extend(_flatten_text(item))
        return output
    return [str(value)]


class MetadataHashEmbedder:
    """Dependency-free feature-hashing baseline.

    It embeds query text and textual Part metadata into the same vector space.
    This is intentionally a plumbing/search baseline, not a semantic vision
    encoder. Richer embedders can replace it without changing repository,
    index, DSL, or runtime contracts.
    """

    def __init__(self, dimensions: int = 128) -> None:
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.identity = EmbeddingIdentity(
            family="metadata-text",
            model="feature-hash",
            version="v1",
            dimensions=dimensions,
        )

    def _embed_tokens(self, values: Sequence[str]) -> tuple[float, ...]:
        vector = [0.0] * self.identity.dimensions
        token_count = 0
        for value in values:
            for token in _TOKEN_RE.findall(value.lower()):
                token_count += 1
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.identity.dimensions
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[index] += sign
        if token_count == 0:
            return tuple(vector)
        return _normalize(vector)

    def embed_text(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str):
            raise TypeError("text query must be a string")
        return self._embed_tokens([text])

    def embed_part(self, part: Part) -> tuple[float, ...]:
        values = [part.category]
        values.extend(part.tags)
        values.extend(_flatten_text(part.attributes))
        values.extend(_flatten_text(part.metadata.get("search_text")))
        return self._embed_tokens(values)
