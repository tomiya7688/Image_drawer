"""共通Step interfaceと宣言的Step schema。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from image_drawer.core import Artifact, Score

ArtifactTypeSpec = str | tuple[str, ...]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """1個のserialize可能Step parameterに対するvalidation rule。"""

    value_type: type | tuple[type, ...]
    required: bool = False
    default: Any = _MISSING
    choices: tuple[Any, ...] | None = None

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING


@dataclass(frozen=True, slots=True)
class StepSchema:
    """validatorとGUI/DSL layerが使用するstatic contract。"""

    step_type: str
    input_types: tuple[ArtifactTypeSpec, ...]
    output_type: str
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class StepContext:
    """RuntimeからStep実装へ渡すexecution context。"""

    run_id: str
    step_id: str
    backend: str
    external_inputs: dict[str, Any]

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


class Step(ABC):
    """model/backendから独立したRuntime operation。"""

    schema: StepSchema
    backend: str = "default"

    @abstractmethod
    def run(
        self,
        inputs: list[Artifact],
        params: dict[str, Any],
        context: StepContext,
    ) -> Artifact:
        """Stepを実行し、M2で定義する単一output Artifactを返す。"""
