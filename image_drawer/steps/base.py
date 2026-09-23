"""Common Step interface and declarative Step schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from image_drawer.core import Artifact, Score

ArtifactTypeSpec = str | tuple[str, ...]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Validation rules for one serializable Step parameter."""

    value_type: type | tuple[type, ...]
    required: bool = False
    default: Any = _MISSING
    choices: tuple[Any, ...] | None = None

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING


@dataclass(frozen=True, slots=True)
class StepSchema:
    """Static contract used by the validator and GUI/DSL layers."""

    step_type: str
    input_types: tuple[ArtifactTypeSpec, ...]
    output_type: str
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class StepContext:
    """Execution context supplied by the runtime to a Step implementation."""

    run_id: str
    step_id: str
    backend: str
    external_inputs: dict[str, Any]
    seed: int | None = None

    def artifact_id(self) -> str:
        return f"artifact:{self.run_id}:{self.step_id}"

    def score_id(self, suffix: str = "0") -> str:
        return f"score:{self.run_id}:{self.step_id}:{suffix}"

    def make_artifact(
        self,
        artifact_type: str,
        *,
        parent_artifact_ids: list[str] | None = None,
        uri: str | None = None,
        model: str | None = None,
        version: str | None = None,
        seed: int | None = None,
        scores: list[Score] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        return Artifact(
            id=self.artifact_id(),
            artifact_type=artifact_type,
            uri=uri,
            parent_artifact_ids=list(parent_artifact_ids or []),
            producing_step_id=self.step_id,
            backend=self.backend,
            model=model,
            version=version,
            seed=seed,
            scores=list(scores or []),
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class StepResult:
    """Primary dataflow Artifact plus optional inspectable side Artifacts."""

    primary: Artifact
    additional_artifacts: list[Artifact] = field(default_factory=list)

    def all_artifacts(self) -> list[Artifact]:
        artifacts = [self.primary, *self.additional_artifacts]
        ids = [artifact.id for artifact in artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("StepResult contains duplicate Artifact IDs")
        return artifacts


class Step(ABC):
    """Model/backend-independent runtime operation."""

    schema: StepSchema
    backend: str = "default"

    @abstractmethod
    def run(
        self,
        inputs: list[Artifact],
        params: dict[str, Any],
        context: StepContext,
    ) -> Artifact | StepResult:
        """Execute the Step.

        Plain Artifact returns remain supported. StepResult allows operations
        such as COMPOSE to expose structured side artifacts while preserving
        one primary artifact for normal workflow dataflow.
        """
