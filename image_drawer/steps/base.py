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

    def artifact_id(self, suffix: str | None = None) -> str:
        """追加Artifactには同一Stepに属する別IDを付ける。"""
        result = f"artifact:{self.run_id}:{self.step_id}"
        return f"{result}:{suffix}" if suffix else result

    def score_id(self, suffix: str = "0") -> str:
        return f"score:{self.run_id}:{self.step_id}:{suffix}"

    def make_artifact(
        self,
        artifact_type: str,
        *,
        suffix: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        uri: str | None = None,
        model: str | None = None,
        version: str | None = None,
        seed: int | None = None,
        scores: list[Score] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        return Artifact(
            id=self.artifact_id(suffix),
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
    """主出力と、同じStepが生成した独立した補助Artifactを保持する。"""

    output: Artifact
    additional_artifacts: list[Artifact] = field(default_factory=list)


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
    ) -> Artifact | StepResult:
        """主出力1個、必要なら追加の独立Artifactを返す。"""
