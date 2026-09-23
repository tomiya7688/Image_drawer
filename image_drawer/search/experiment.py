"""Reusable experiment runner for paired workflow comparison."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from statistics import fmean
from time import perf_counter
from typing import Any, Mapping

from image_drawer.core import Trajectory, WorkflowSpec
from image_drawer.dsl import serialize_workflow
from image_drawer.runtime import RuntimeExecutionError, WorkflowRuntime, validate_workflow
from image_drawer.search.models import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    WorkflowMetrics,
    WorkflowSnapshot,
)
from image_drawer.steps import StepRegistry


def _seed_for(
    spec: ExperimentSpec,
    prompt_index: int,
    repeat_index: int,
) -> int | None:
    if spec.seed_policy == "none":
        return None
    assert spec.base_seed is not None
    if spec.seed_policy == "fixed":
        return spec.base_seed
    return spec.base_seed + prompt_index * spec.repeats + repeat_index


def _run_id(
    experiment_id: str,
    workflow_key: str,
    prompt_case_id: str,
    repeat_index: int,
    seed: int | None,
) -> str:
    payload = {
        "experiment_id": experiment_id,
        "workflow_key": workflow_key,
        "prompt_case_id": prompt_case_id,
        "repeat_index": repeat_index,
        "seed": seed,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return f"experiment-run-{digest}"


def _snapshot_workflow(
    key: str,
    workflow: WorkflowSpec,
    registry: StepRegistry,
) -> WorkflowSnapshot:
    validate_workflow(workflow, registry)
    canonical = serialize_workflow(workflow, registry)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    backends = {
        step.id: registry.backend_name(step.type, step.backend)
        for step in workflow.nodes
    }
    return WorkflowSnapshot(
        key=key,
        workflow_id=workflow.id,
        workflow_version=workflow.version,
        canonical_dsl=canonical,
        content_hash=f"sha256:{digest}",
        backends=backends,
        metadata=dict(workflow.metadata),
    )


def _collect_observed_versions(
    runs: list[ExperimentRun],
) -> dict[str, Any]:
    identities: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    models: dict[str, dict[str, Any]] = {}
    step_backends: set[tuple[str, str]] = set()

    for run in runs:
        trajectory = run.trajectory
        if trajectory is None:
            continue

        execution_identities = trajectory.metadata.get("execution_identity", {})
        if isinstance(execution_identities, dict):
            for step_id, identity_group in execution_identities.items():
                if not isinstance(identity_group, dict):
                    continue
                for family, identity in identity_group.items():
                    if not isinstance(identity, dict):
                        continue
                    key = json.dumps(
                        identity,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    identities[family][key] = dict(identity)

        for artifact in trajectory.artifacts:
            if artifact.model or artifact.version or artifact.backend:
                payload = {
                    "model": artifact.model,
                    "version": artifact.version,
                    "backend": artifact.backend,
                    "artifact_type": artifact.artifact_type,
                }
                key = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                )
                models[key] = payload

        for execution in trajectory.executions:
            if execution.backend:
                step_backends.add((execution.step_type, execution.backend))

    return {
        "execution_identities": {
            family: [
                payload
                for _, payload in sorted(values.items())
            ]
            for family, values in sorted(identities.items())
        },
        "artifact_models": [
            payload for _, payload in sorted(models.items())
        ],
        "step_backends": [
            {"step_type": step_type, "backend": backend}
            for step_type, backend in sorted(step_backends)
        ],
    }


def aggregate_workflow_metrics(
    runs: list[ExperimentRun],
    workflow_key: str,
) -> WorkflowMetrics:
    selected = [run for run in runs if run.workflow_key == workflow_key]
    if not selected:
        raise ValueError(f"no runs for workflow {workflow_key}")

    successes = [run for run in selected if run.status == "success"]
    failures = [run for run in selected if run.status == "failure"]
    runtimes = [run.runtime_seconds for run in selected]

    scores = []
    for run in selected:
        if run.trajectory is not None:
            scores.extend(run.trajectory.evaluations)

    overall_values = [
        float(score.overall)
        for score in scores
        if score.overall is not None
    ]
    components: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        for name, value in score.components.items():
            components[name].append(float(value))

    return WorkflowMetrics(
        workflow_key=workflow_key,
        run_count=len(selected),
        success_count=len(successes),
        failure_count=len(failures),
        failure_rate=len(failures) / len(selected),
        runtime_seconds_mean=fmean(runtimes),
        runtime_seconds_min=min(runtimes),
        runtime_seconds_max=max(runtimes),
        evaluation_count=len(scores),
        score_overall_mean=(
            fmean(overall_values) if overall_values else None
        ),
        component_means={
            name: fmean(values)
            for name, values in sorted(components.items())
        },
        metadata={
            "successful_trajectory_ids": [
                run.trajectory.id
                for run in successes
                if run.trajectory is not None
            ],
            "failure_errors": [
                run.error for run in failures if run.error is not None
            ],
        },
    )


class ExperimentRunner:
    """Run validated workflows on a shared prompt/seed schedule."""

    def __init__(
        self,
        registry: StepRegistry,
        runtime: WorkflowRuntime | None = None,
        *,
        environment_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.runtime = runtime or WorkflowRuntime(registry)
        self.environment_metadata = dict(environment_metadata or {})

    def run(
        self,
        spec: ExperimentSpec,
        workflows: Mapping[str, WorkflowSpec],
    ) -> ExperimentResult:
        if len(workflows) < 2:
            raise ValueError(
                "workflow comparison requires at least two workflows"
            )
        if any(not key for key in workflows):
            raise ValueError("workflow comparison keys must not be empty")

        snapshots = [
            _snapshot_workflow(key, workflow, self.registry)
            for key, workflow in workflows.items()
        ]
        snapshot_by_key = {snapshot.key: snapshot for snapshot in snapshots}

        runs: list[ExperimentRun] = []
        for prompt_index, prompt_case in enumerate(spec.prompt_set):
            for repeat_index in range(spec.repeats):
                seed = _seed_for(spec, prompt_index, repeat_index)
                for workflow_key, workflow in workflows.items():
                    input_ids = list(workflow.inputs)
                    run_id = _run_id(
                        spec.id,
                        workflow_key,
                        prompt_case.id,
                        repeat_index,
                        seed,
                    )
                    started = perf_counter()
                    trajectory: Trajectory | None = None
                    error: str | None = None
                    status = "success"

                    try:
                        if len(input_ids) != 1:
                            raise ValueError(
                                "experiment runner requires exactly one "
                                "external workflow input"
                            )
                        result = self.runtime.execute(
                            workflow,
                            external_inputs={
                                input_ids[0]: prompt_case.prompt
                            },
                            run_id=run_id,
                            prompt=prompt_case.prompt,
                            seed=seed,
                        )
                        trajectory = result.trajectory
                    except RuntimeExecutionError as exc:
                        status = "failure"
                        trajectory = exc.trajectory
                        error = str(exc)
                    except Exception as exc:
                        status = "failure"
                        error = f"{type(exc).__name__}: {exc}"

                    duration = perf_counter() - started
                    if trajectory is not None:
                        trajectory.metadata.setdefault(
                            "experiment",
                            {},
                        ).update(
                            {
                                "experiment_id": spec.id,
                                "workflow_key": workflow_key,
                                "prompt_case_id": prompt_case.id,
                                "repeat_index": repeat_index,
                                "seed": seed,
                                "workflow_content_hash": snapshot_by_key[
                                    workflow_key
                                ].content_hash,
                            }
                        )

                    runs.append(
                        ExperimentRun(
                            id=run_id,
                            workflow_key=workflow_key,
                            prompt_case_id=prompt_case.id,
                            prompt=prompt_case.prompt,
                            repeat_index=repeat_index,
                            seed=seed,
                            status=status,
                            runtime_seconds=duration,
                            trajectory=trajectory,
                            error=error,
                            metadata={
                                "prompt_metadata": dict(
                                    prompt_case.metadata
                                ),
                            },
                        )
                    )

        aggregates = {
            key: aggregate_workflow_metrics(runs, key)
            for key in workflows
        }
        environment = dict(self.environment_metadata)
        environment.setdefault(
            "git_commit",
            os.environ.get("IMAGE_DRAWER_GIT_COMMIT"),
        )

        return ExperimentResult(
            id=spec.id,
            spec=spec,
            workflows=snapshots,
            runs=runs,
            aggregate_metrics=aggregates,
            metadata={
                "environment": environment,
                "observed_versions": _collect_observed_versions(runs),
                "comparison_schedule": {
                    "prompt_case_ids": [
                        item.id for item in spec.prompt_set
                    ],
                    "repeats": spec.repeats,
                    "seed_policy": spec.seed_policy,
                    "base_seed": spec.base_seed,
                },
            },
        )
