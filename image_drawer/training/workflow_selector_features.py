"""WorkflowSelector baseline用の固定特徴量スキーマ。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from image_drawer.core import SerializableModel


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(slots=True)
class WorkflowSelectorFeatureSchema(SerializableModel):
    """prompt/contextと既知workflowの静的情報だけを特徴量化する。"""

    version: str = "workflow-selector-feature-v1"
    text_dimensions: int = 32

    def __post_init__(self) -> None:
        if type(self.text_dimensions) is not int or self.text_dimensions <= 0:
            raise ValueError("text_dimensions must be a positive integer")

    @property
    def total_dimensions(self) -> int:
        return self.text_dimensions * 3

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
    if isinstance(value, set):
        output: list[str] = []
        for item in sorted(value, key=str):
            output.extend(_flatten_text(item))
        return output
    if isinstance(value, (list, tuple)):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_text(item))
        return output
    return [str(value)]


def _hash_text(values: Sequence[str], dimensions: int) -> list[float]:
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


def _prompt_text(
    prompt: str,
    context: Mapping[str, Any] | None,
) -> list[str]:
    values = [prompt]
    if isinstance(context, Mapping):
        prompt_metadata = context.get("prompt_metadata")
        if isinstance(prompt_metadata, Mapping):
            values.extend(_flatten_text(prompt_metadata))
        for key in ("task", "style", "resource_budget", "dataset_context"):
            if key in context:
                values.extend(_flatten_text(context[key]))
    return values


def workflow_static_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """run後のobjective/statusを除外した既知workflow定義を返す。"""
    workflow_metadata = candidate.get("workflow_metadata", {})
    if not isinstance(workflow_metadata, Mapping):
        workflow_metadata = {}
    return {
        "workflow_key": str(candidate.get("workflow_key", "")),
        "workflow_id": str(candidate.get("workflow_id", "")),
        "workflow_version": str(candidate.get("workflow_version", "")),
        "workflow_metadata": dict(workflow_metadata),
    }


def _workflow_text(candidate: Mapping[str, Any]) -> list[str]:
    static = workflow_static_record(candidate)
    values = [
        static["workflow_key"],
        static["workflow_id"],
        static["workflow_version"],
    ]
    metadata = static["workflow_metadata"]
    for key in ("content_hash", "backends", "metadata"):
        if key in metadata:
            values.extend(_flatten_text(metadata[key]))
    return values


def encode_workflow_selector_candidate(
    prompt: str,
    candidate: Mapping[str, Any],
    schema: WorkflowSelectorFeatureSchema,
    *,
    context: Mapping[str, Any] | None = None,
) -> list[float]:
    prompt_vector = _hash_text(
        _prompt_text(prompt, context),
        schema.text_dimensions,
    )
    workflow_vector = _hash_text(
        _workflow_text(candidate),
        schema.text_dimensions,
    )
    interaction = [
        left * right
        for left, right in zip(prompt_vector, workflow_vector)
    ]
    vector = [*prompt_vector, *workflow_vector, *interaction]
    if len(vector) != schema.total_dimensions:
        raise AssertionError("feature vector dimension mismatch")
    return vector
