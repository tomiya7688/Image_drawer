"""Runtime、Part Bank、experimentで共有するserialize可能なcore record。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from image_drawer.core.serialization import SerializableModel


Metadata = dict[str, Any]


@dataclass(slots=True)
class Score(SerializableModel):
    id: str
    evaluator: str
    components: dict[str, float] = field(default_factory=dict)
    overall: float | None = None
    evaluator_version: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class Artifact(SerializableModel):
    id: str
    artifact_type: str
    uri: str | None = None
    parent_artifact_ids: list[str] = field(default_factory=list)
    producing_step_id: str | None = None
    backend: str | None = None
    model: str | None = None
    version: str | None = None
    seed: int | None = None
    scores: list[Score] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class SourceImage(SerializableModel):
    id: str
    uri: str
    width: int
    height: int
    checksum: str
    dataset: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class Part(SerializableModel):
    id: str
    source_image_id: str
    category: str
    crop_uri: str
    bbox: tuple[int, int, int, int]
    mask_uri: str | None = None
    embedding_refs: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    attributes: Metadata = field(default_factory=dict)
    quality: Metadata = field(default_factory=dict)
    extraction_method: str | None = None
    extraction_version: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class PartSet(SerializableModel):
    id: str
    query_id: str
    part_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    category: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.part_ids) != len(self.retrieval_scores):
            raise ValueError("part_ids and retrieval_scores must have equal length")


@dataclass(slots=True)
class Layout(SerializableModel):
    id: str
    canvas_width: int
    canvas_height: int
    slots: list[Metadata] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class PartPlacement(SerializableModel):
    id: str
    part_id: str
    x: float
    y: float
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    z_index: int = 0
    opacity: float = 1.0
    transform_metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class Composition(SerializableModel):
    id: str
    canvas_size: tuple[int, int]
    placements: list[PartPlacement] = field(default_factory=list)
    background: Any = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class StepSpec(SerializableModel):
    id: str
    type: str
    backend: str | None = None
    parameters: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowSpec(SerializableModel):
    id: str
    version: str
    nodes: list[StepSpec] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class StepExecution(SerializableModel):
    id: str
    step_id: str
    step_type: str
    backend: str | None = None
    parameters: Metadata = field(default_factory=dict)
    input_artifact_ids: list[str] = field(default_factory=list)
    output_artifact_ids: list[str] = field(default_factory=list)
    seed: int | None = None
    start_time: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory(SerializableModel):
    id: str
    workflow_id: str
    workflow_version: str
    prompt: str
    input_metadata: Metadata = field(default_factory=dict)
    executions: list[StepExecution] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    evaluations: list[Score] = field(default_factory=list)
    selections: list[Metadata] = field(default_factory=list)
    final_output_ids: list[str] = field(default_factory=list)
    human_feedback: list[Metadata] = field(default_factory=list)
    timing: Metadata = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)
