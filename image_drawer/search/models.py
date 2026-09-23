"""Serializable experiment records for workflow comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from image_drawer.core import SerializableModel, Trajectory


Metadata = dict[str, Any]


@dataclass(slots=True)
class PromptCase(SerializableModel):
    id: str
    prompt: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSpec(SerializableModel):
    id: str
    prompt_set: list[PromptCase]
    repeats: int = 1
    seed_policy: str = "paired_increment"
    base_seed: int | None = 0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("experiment id must not be empty")
        if not self.prompt_set:
            raise ValueError("experiment prompt_set must not be empty")
        if type(self.repeats) is not int or self.repeats <= 0:
            raise ValueError("repeats must be a positive integer")
        if self.seed_policy not in {"paired_increment", "fixed", "none"}:
            raise ValueError(
                "seed_policy must be paired_increment, fixed, or none"
            )
        if self.seed_policy != "none" and type(self.base_seed) is not int:
            raise ValueError("base_seed must be an integer for seeded policies")


@dataclass(slots=True)
class WorkflowSnapshot(SerializableModel):
    key: str
    workflow_id: str
    workflow_version: str
    canonical_dsl: str
    content_hash: str
    backends: dict[str, str] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentRun(SerializableModel):
    id: str
    workflow_key: str
    prompt_case_id: str
    prompt: str
    repeat_index: int
    seed: int | None
    status: str
    runtime_seconds: float
    trajectory: Trajectory | None = None
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowMetrics(SerializableModel):
    workflow_key: str
    run_count: int
    success_count: int
    failure_count: int
    failure_rate: float
    runtime_seconds_mean: float
    runtime_seconds_min: float
    runtime_seconds_max: float
    evaluation_count: int = 0
    score_overall_mean: float | None = None
    component_means: dict[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentResult(SerializableModel):
    id: str
    spec: ExperimentSpec
    workflows: list[WorkflowSnapshot]
    runs: list[ExperimentRun]
    aggregate_metrics: dict[str, WorkflowMetrics]
    metadata: Metadata = field(default_factory=dict)

    def successful_runs(self) -> list[ExperimentRun]:
        return [run for run in self.runs if run.status == "success"]

    def failed_runs(self) -> list[ExperimentRun]:
        return [run for run in self.runs if run.status == "failure"]
