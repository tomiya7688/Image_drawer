"""vector index abstractionとdependency-freeなlocal cosine backend。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from image_drawer.part_bank.embedding import EmbeddingIdentity


@dataclass(frozen=True, slots=True)
class IndexIdentity:
    backend: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"backend": self.backend, "version": self.version}


@dataclass(frozen=True, slots=True)
class SearchHit:
    part_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PartIndex(Protocol):
    identity: IndexIdentity
    embedding_identity: EmbeddingIdentity

    def add(
        self,
        part_id: str,
        vector: Sequence[float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        ...


def _validate_vector(vector: Sequence[float], dimensions: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if len(values) != dimensions:
        raise ValueError(
            f"vector dimension mismatch: expected {dimensions}, got {len(values)}"
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("vectors must contain only finite values")
    return values


class BruteForceCosineIndex:
    """vector database backendが必要になるまで使用する小規模local index。"""

    identity = IndexIdentity(backend="bruteforce-cosine", version="v1")

    def __init__(self, embedding_identity: EmbeddingIdentity) -> None:
        self.embedding_identity = embedding_identity
        self._entries: dict[str, tuple[tuple[float, ...], dict[str, Any]]] = {}

    def add(
        self,
        part_id: str,
        vector: Sequence[float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(part_id, str) or not part_id:
            raise ValueError("part_id must be a non-empty string")
        values = _validate_vector(vector, self.embedding_identity.dimensions)
        item = (values, dict(metadata or {}))
        existing = self._entries.get(part_id)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting vector for part: {part_id}")
        self._entries[part_id] = item

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query = _validate_vector(vector, self.embedding_identity.dimensions)
        filters = dict(filters or {})
        query_norm = math.sqrt(sum(value * value for value in query))

        hits: list[SearchHit] = []
        for part_id, (candidate, metadata) in self._entries.items():
            if any(metadata.get(key) != value for key, value in filters.items()):
                continue
            candidate_norm = math.sqrt(sum(value * value for value in candidate))
            if query_norm == 0.0 or candidate_norm == 0.0:
                score = 0.0
            else:
                score = sum(a * b for a, b in zip(query, candidate))
                score /= query_norm * candidate_norm
                score = max(-1.0, min(1.0, score))
            hits.append(SearchHit(part_id=part_id, score=score, metadata=metadata))

        hits.sort(key=lambda hit: (-hit.score, hit.part_id))
        return hits[:top_k]

    def __len__(self) -> int:
        return len(self._entries)
