"""Evaluator contracts, registry, and Image/ImageSet candidate handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from image_drawer.core import Artifact, ScoreSet


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    name: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


class Evaluator(Protocol):
    """Batch evaluator interface independent of selection policy."""

    identity: EvaluationIdentity

    def evaluate(
        self,
        artifacts: Sequence[Artifact],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ScoreSet:
        ...


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, evaluator: Evaluator) -> None:
        name = evaluator.identity.name
        if not isinstance(name, str) or not name:
            raise ValueError("evaluator name must be a non-empty string")
        if name in self._evaluators:
            raise ValueError(f"evaluator already registered: {name}")
        self._evaluators[name] = evaluator

    def resolve(self, name: str) -> Evaluator:
        try:
            return self._evaluators[name]
        except KeyError as exc:
            raise KeyError(f"unknown evaluator: {name}") from exc

    def identities(self) -> dict[str, EvaluationIdentity]:
        return {
            name: evaluator.identity
            for name, evaluator in sorted(self._evaluators.items())
        }


def artifact_candidates(artifact: Artifact) -> list[Artifact]:
    """Normalize Image or serialized ImageSet into candidate Artifacts."""
    if artifact.artifact_type == "Image":
        return [artifact]
    if artifact.artifact_type != "ImageSet":
        raise TypeError(
            f"evaluation/selection expects Image or ImageSet, "
            f"got {artifact.artifact_type}"
        )

    payload = artifact.metadata.get("candidates")
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "ImageSet metadata.candidates must contain serialized Artifacts"
        )

    candidates: list[Artifact] = []
    for index, item in enumerate(payload):
        if isinstance(item, Artifact):
            candidate = item
        elif isinstance(item, dict):
            candidate = Artifact.from_dict(item)
        else:
            raise TypeError(
                f"ImageSet candidate {index} must be an Artifact object"
            )
        if candidate.artifact_type != "Image":
            raise ValueError(
                f"ImageSet candidate {candidate.id} must have type Image"
            )
        candidates.append(candidate)

    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("ImageSet candidate IDs must be unique")
    return candidates
