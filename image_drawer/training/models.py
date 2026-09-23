"""学習データセットの再現可能な正本モデル。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from image_drawer.core import SerializableModel


Metadata = dict[str, Any]


@dataclass(slots=True)
class SplitConfig(SerializableModel):
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(
            type(value) not in (int, float) or float(value) < 0.0
            for value in values
        ):
            raise ValueError("split ratios must be non-negative numbers")
        total = sum(float(value) for value in values)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")
        if type(self.seed) is not int:
            raise ValueError("split seed must be an integer")


@dataclass(slots=True)
class DatasetBuildConfig(SerializableModel):
    id: str
    split: SplitConfig = field(default_factory=SplitConfig)
    part_example_mode: str = "both"
    max_pairs_per_group: int = 64
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("dataset build id must not be empty")
        if self.part_example_mode not in {"pointwise", "pairwise", "both"}:
            raise ValueError(
                "part_example_mode must be pointwise, pairwise, or both"
            )
        if (
            type(self.max_pairs_per_group) is not int
            or self.max_pairs_per_group <= 0
        ):
            raise ValueError("max_pairs_per_group must be a positive integer")


@dataclass(slots=True)
class PartCandidateFeatures(SerializableModel):
    part_id: str
    retrieval_score: float | None = None
    part_metadata: Metadata = field(default_factory=dict)
    embedding_refs: dict[str, str] = field(default_factory=dict)
    embedding_identity: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class PartSelectorExample(SerializableModel):
    id: str
    split: str
    group_id: str
    example_kind: str
    prompt: str
    context: Metadata
    candidates: list[PartCandidateFeatures]
    label: Metadata
    source_trajectory_ids: list[str]
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowCandidateFeatures(SerializableModel):
    workflow_key: str
    workflow_id: str
    workflow_version: str
    status: str
    objective_value: float | None
    objective_components: dict[str, float] = field(default_factory=dict)
    workflow_metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowSelectorExample(SerializableModel):
    id: str
    split: str
    group_id: str
    prompt: str
    context: Metadata
    candidates: list[WorkflowCandidateFeatures]
    label: Metadata
    source_run_ids: list[str]
    source_trajectory_ids: list[str]
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class TrainingDataset(SerializableModel):
    id: str
    task: str
    build_config: DatasetBuildConfig
    examples: list[Metadata]
    source_ids: list[str]
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in {"part_selector", "workflow_selector"}:
            raise ValueError("unsupported training dataset task")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source_ids must be unique")

    def split_counts(self) -> dict[str, int]:
        counts = {"train": 0, "validation": 0, "test": 0}
        for example in self.examples:
            split = example.get("split")
            if split in counts:
                counts[split] += 1
        return counts
