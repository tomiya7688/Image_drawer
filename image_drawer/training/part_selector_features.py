"""PartSelector baseline用の固定特徴量スキーマ。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from image_drawer.core import SerializableModel


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(slots=True)
class PartSelectorFeatureSchema(SerializableModel):
    """checkpointとdatasetの互換性を決める特徴量スキーマ。"""

    version: str = "part-selector-feature-v1"
    text_dimensions: int = 32
    include_retrieval_score: bool = True
    include_bbox_features: bool = True

    def __post_init__(self) -> None:
        if type(self.text_dimensions) is not int or self.text_dimensions <= 0:
            raise ValueError("text_dimensions must be a positive integer")

    @property
    def total_dimensions(self) -> int:
        dimensions = self.text_dimensions * 3
        if self.include_retrieval_score:
            dimensions += 1
        if self.include_bbox_features:
            dimensions += 2
        return dimensions

    @property
    def identity_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, Mapping):
        output: list[str] = []
        for key in sorted(value, key=str):
            output.append(str(key))
            output.extend(_flatten_text(value[key]))
        return output
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        output: list[str] = []
        for item in items:
            output.extend(_flatten_text(item))
        return output
    return [str(value)]


def _hash_text(values: list[str], dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for value in values:
        for token in _TOKEN_RE.findall(value.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
    norm = math.sqrt(sum(item * item for item in vector))
    if norm:
        vector = [item / norm for item in vector]
    return vector


def _context_text(example: Mapping[str, Any]) -> list[str]:
    context = example.get("context", {})
    if not isinstance(context, Mapping):
        context = {}
    values = [str(example.get("prompt", ""))]
    for key in ("category", "retrieval_filters"):
        if key in context:
            values.extend(_flatten_text(context[key]))
    return values


def _candidate_text(candidate: Mapping[str, Any]) -> list[str]:
    metadata = candidate.get("part_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    values: list[str] = []
    for key in (
        "category",
        "tags",
        "attributes",
        "quality",
        "extraction_method",
        "extraction_version",
    ):
        if key in metadata:
            values.extend(_flatten_text(metadata[key]))

    nested = metadata.get("metadata", {})
    if isinstance(nested, Mapping):
        for key in ("search_text", "style", "pose", "attributes"):
            if key in nested:
                values.extend(_flatten_text(nested[key]))

    identity = candidate.get("embedding_identity", {})
    if isinstance(identity, Mapping):
        for key in ("family", "model", "version", "dimensions"):
            if key in identity:
                values.extend(_flatten_text(identity[key]))
    return values


def _bbox_features(candidate: Mapping[str, Any]) -> tuple[float, float]:
    metadata = candidate.get("part_metadata", {})
    if not isinstance(metadata, Mapping):
        return 0.0, 0.0
    bbox = metadata.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(item) not in (int, float) for item in bbox)
    ):
        return 0.0, 0.0
    width = max(float(bbox[2]), 0.0)
    height = max(float(bbox[3]), 0.0)
    if width <= 0.0 or height <= 0.0:
        return 0.0, 0.0
    aspect = math.log(width / height)
    log_area = math.log1p(width * height) / 20.0
    return aspect, log_area


def encode_part_selector_candidate(
    example: Mapping[str, Any],
    candidate: Mapping[str, Any],
    schema: PartSelectorFeatureSchema,
) -> list[float]:
    """prompt/contextとPart metadataから固定長vectorを生成する。"""
    prompt_vector = _hash_text(
        _context_text(example),
        schema.text_dimensions,
    )
    part_vector = _hash_text(
        _candidate_text(candidate),
        schema.text_dimensions,
    )
    interaction = [
        left * right
        for left, right in zip(prompt_vector, part_vector)
    ]
    vector = [*prompt_vector, *part_vector, *interaction]

    if schema.include_retrieval_score:
        score = candidate.get("retrieval_score")
        if type(score) in (int, float) and math.isfinite(float(score)):
            vector.append(float(score))
        else:
            vector.append(0.0)

    if schema.include_bbox_features:
        vector.extend(_bbox_features(candidate))

    if len(vector) != schema.total_dimensions:
        raise AssertionError("feature vector dimension mismatch")
    return vector
