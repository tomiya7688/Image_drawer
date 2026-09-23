"""Grid/random search over validated WorkflowSpec parameter variants."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from image_drawer.core import SerializableModel, WorkflowSpec
from image_drawer.runtime import validate_workflow
from image_drawer.search.experiment import ExperimentRunner
from image_drawer.search.models import ExperimentResult, ExperimentSpec
from image_drawer.steps import StepRegistry


Metadata = dict[str, Any]


@dataclass(slots=True)
class ParameterDimension(SerializableModel):
    step_id: str
    parameter: str
    values: list[Any]

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("step_id must not be empty")
        if not self.parameter:
            raise ValueError("parameter must not be empty")
        if not self.values:
            raise ValueError("search dimension values must not be empty")


@dataclass(slots=True)
class ObjectiveSpec(SerializableModel):
    weights: dict[str, float]
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("objective weights must not be empty")
        for name, weight in self.weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("objective metric names must be non-empty strings")
            if type(weight) not in (int, float) or not math.isfinite(float(weight)):
                raise ValueError(f"objective weight for {name} must be finite")


@dataclass(slots=True)
class SearchSpec(SerializableModel):
    id: str
    strategy: str
    dimensions: list[ParameterDimension]
    objective: ObjectiveSpec
    sample_count: int | None = None
    random_seed: int = 0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("search id must not be empty")
        if self.strategy not in {"grid", "random"}:
            raise ValueError("strategy must be grid or random")
        if not self.dimensions:
            raise ValueError("search dimensions must not be empty")
        keys = [(item.step_id, item.parameter) for item in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError("search dimensions must be unique")
        if self.strategy == "random":
            if type(self.sample_count) is not int or self.sample_count <= 0:
                raise ValueError(
                    "random search requires a positive sample_count"
                )


@dataclass(slots=True)
class SearchCandidate(SerializableModel):
    key: str
    parameters: dict[str, Any]
    workflow: WorkflowSpec
    valid: bool
    validation_error: str | None = None
    objective_value: float | None = None
    objective_components: dict[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ParameterSearchResult(SerializableModel):
    id: str
    spec: SearchSpec
    base_workflow: WorkflowSpec
    candidates: list[SearchCandidate]
    ranking: list[str]
    best_candidate_key: str | None
    experiment_result: ExperimentResult | None
    metadata: Metadata = field(default_factory=dict)

    def candidate(self, key: str) -> SearchCandidate:
        for item in self.candidates:
            if item.key == key:
                return item
        raise KeyError(f"unknown search candidate: {key}")

    def best_candidate(self) -> SearchCandidate | None:
        if self.best_candidate_key is None:
            return None
        return self.candidate(self.best_candidate_key)


def _json_key(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _configuration_key(configuration: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_json_key(configuration).encode("utf-8")).hexdigest()
    return f"candidate-{digest[:16]}"


def _clone_workflow(workflow: WorkflowSpec) -> WorkflowSpec:
    return WorkflowSpec.from_dict(workflow.to_dict())


def _step_map(workflow: WorkflowSpec):
    return {step.id: step for step in workflow.nodes}


def apply_parameter_configuration(
    base_workflow: WorkflowSpec,
    configuration: Mapping[str, Any],
) -> WorkflowSpec:
    """Return a cloned WorkflowSpec with step.parameter paths applied."""
    workflow = _clone_workflow(base_workflow)
    steps = _step_map(workflow)
    normalized: dict[str, Any] = {}
    for path, value in configuration.items():
        if not isinstance(path, str) or "." not in path:
            raise ValueError(
                "parameter configuration keys must be step_id.parameter"
            )
        step_id, parameter = path.split(".", 1)
        if not step_id or not parameter:
            raise ValueError(
                "parameter configuration keys must be step_id.parameter"
            )
        try:
            step = steps[step_id]
        except KeyError as exc:
            raise KeyError(f"unknown Step in parameter search: {step_id}") from exc
        step.parameters[parameter] = value
        normalized[path] = value

    workflow.metadata = dict(workflow.metadata)
    workflow.metadata["parameter_search"] = {
        "configuration": normalized,
    }
    return workflow


def _grid_configurations(
    dimensions: Sequence[ParameterDimension],
) -> list[dict[str, Any]]:
    paths = [f"{item.step_id}.{item.parameter}" for item in dimensions]
    return [
        dict(zip(paths, values))
        for values in itertools.product(
            *(item.values for item in dimensions)
        )
    ]


def _random_configurations(
    dimensions: Sequence[ParameterDimension],
    *,
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    total = math.prod(len(item.values) for item in dimensions)
    if sample_count >= total:
        all_configs = _grid_configurations(dimensions)
        rng = random.Random(seed)
        rng.shuffle(all_configs)
        return all_configs

    rng = random.Random(seed)
    paths = [f"{item.step_id}.{item.parameter}" for item in dimensions]
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    while len(output) < sample_count:
        configuration = {
            path: rng.choice(dimension.values)
            for path, dimension in zip(paths, dimensions)
        }
        signature = _json_key(configuration)
        if signature in seen:
            continue
        seen.add(signature)
        output.append(configuration)
    return output


def generate_configurations(spec: SearchSpec) -> list[dict[str, Any]]:
    if spec.strategy == "grid":
        return _grid_configurations(spec.dimensions)
    assert spec.sample_count is not None
    return _random_configurations(
        spec.dimensions,
        sample_count=spec.sample_count,
        seed=spec.random_seed,
    )


def generate_candidates(
    base_workflow: WorkflowSpec,
    spec: SearchSpec,
    registry: StepRegistry,
) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    for configuration in generate_configurations(spec):
        key = _configuration_key(configuration)
        try:
            workflow = apply_parameter_configuration(
                base_workflow,
                configuration,
            )
            validate_workflow(workflow, registry)
        except Exception as exc:
            candidates.append(
                SearchCandidate(
                    key=key,
                    parameters=dict(configuration),
                    workflow=_clone_workflow(base_workflow),
                    valid=False,
                    validation_error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        workflow.metadata.setdefault("parameter_search", {}).update(
            {
                "search_id": spec.id,
                "candidate_key": key,
                "strategy": spec.strategy,
                "configuration": dict(configuration),
            }
        )
        candidates.append(
            SearchCandidate(
                key=key,
                parameters=dict(configuration),
                workflow=workflow,
                valid=True,
            )
        )
    return candidates


def _objective_component(metric, name: str) -> float:
    if name == "score_overall_mean":
        if metric.score_overall_mean is None:
            raise ValueError(
                f"workflow {metric.workflow_key} has no score_overall_mean"
            )
        return float(metric.score_overall_mean)
    if name == "failure_rate":
        return float(metric.failure_rate)
    if name == "runtime_seconds_mean":
        return float(metric.runtime_seconds_mean)
    if name.startswith("component:"):
        component = name.split(":", 1)[1]
        if component not in metric.component_means:
            raise ValueError(
                f"workflow {metric.workflow_key} has no component {component}"
            )
        return float(metric.component_means[component])
    raise ValueError(f"unsupported objective metric: {name}")


def score_objective(metric, objective: ObjectiveSpec) -> tuple[float, dict[str, float]]:
    components = {
        name: _objective_component(metric, name)
        for name in objective.weights
    }
    aggregate = sum(
        components[name] * float(weight)
        for name, weight in objective.weights.items()
    )
    return aggregate, components


class ParameterSearchRunner:
    """Generate, validate, execute, and rank workflow parameter variants."""

    def __init__(
        self,
        registry: StepRegistry,
        experiment_runner: ExperimentRunner | None = None,
    ) -> None:
        self.registry = registry
        self.experiment_runner = experiment_runner or ExperimentRunner(registry)

    def run(
        self,
        base_workflow: WorkflowSpec,
        search_spec: SearchSpec,
        experiment_spec: ExperimentSpec,
    ) -> ParameterSearchResult:
        candidates = generate_candidates(
            base_workflow,
            search_spec,
            self.registry,
        )
        valid = [item for item in candidates if item.valid]
        if len(valid) < 2:
            raise ValueError(
                "parameter search requires at least two valid candidates"
            )

        workflows = {item.key: item.workflow for item in valid}
        experiment = self.experiment_runner.run(
            experiment_spec,
            workflows,
        )

        for candidate in valid:
            metric = experiment.aggregate_metrics[candidate.key]
            aggregate, components = score_objective(
                metric,
                search_spec.objective,
            )
            candidate.objective_value = aggregate
            candidate.objective_components = components
            candidate.metadata["workflow_metrics"] = metric.to_dict()

        ranking = [
            item.key
            for item in sorted(
                valid,
                key=lambda item: (
                    -float(item.objective_value),
                    item.key,
                ),
            )
        ]
        best = ranking[0] if ranking else None
        return ParameterSearchResult(
            id=search_spec.id,
            spec=search_spec,
            base_workflow=_clone_workflow(base_workflow),
            candidates=candidates,
            ranking=ranking,
            best_candidate_key=best,
            experiment_result=experiment,
            metadata={
                "valid_candidate_count": len(valid),
                "invalid_candidate_count": len(candidates) - len(valid),
                "objective_weights": dict(search_spec.objective.weights),
                "best_configuration": (
                    dict(candidates_by_key(candidates)[best].parameters)
                    if best is not None
                    else None
                ),
            },
        )


def candidates_by_key(
    candidates: Sequence[SearchCandidate],
) -> dict[str, SearchCandidate]:
    return {candidate.key: candidate for candidate in candidates}


def reconstruct_best_workflow(
    result: ParameterSearchResult,
) -> WorkflowSpec:
    best = result.best_candidate()
    if best is None:
        raise ValueError("search result has no best candidate")
    workflow = apply_parameter_configuration(
        result.base_workflow,
        best.parameters,
    )
    workflow.metadata.setdefault("parameter_search", {}).update(
        {
            "search_id": result.id,
            "candidate_key": best.key,
            "configuration": dict(best.parameters),
        }
    )
    return workflow
