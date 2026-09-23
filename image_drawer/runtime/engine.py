"""Model-agnostic workflow execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from image_drawer.core import (
    Artifact,
    StepExecution,
    Trajectory,
    WorkflowSpec,
    new_id,
)
from image_drawer.runtime.validation import resolve_parameters, validate_workflow
from image_drawer.steps import StepContext, StepRegistry, StepResult


class ArtifactRegistry:
    """In-memory artifact index for one workflow run."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def add(self, artifact: Artifact) -> None:
        if artifact.id in self._artifacts:
            raise ValueError(f"duplicate artifact id: {artifact.id}")
        self._artifacts[artifact.id] = artifact

    def get(self, artifact_id: str) -> Artifact:
        return self._artifacts[artifact_id]

    def all(self) -> list[Artifact]:
        return list(self._artifacts.values())


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    trajectory: Trajectory
    outputs: dict[str, Artifact]


class RuntimeExecutionError(RuntimeError):
    """Execution failure carrying the partially recorded trajectory."""

    def __init__(self, message: str, trajectory: Trajectory) -> None:
        super().__init__(message)
        self.trajectory = trajectory


class WorkflowRuntime:
    def __init__(self, registry: StepRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        workflow: WorkflowSpec,
        *,
        external_inputs: dict[str, Any],
        run_id: str | None = None,
        prompt: str | None = None,
    ) -> RuntimeResult:
        plan = validate_workflow(workflow, self.registry)
        actual_run_id = run_id or new_id("run")
        node_by_id = {node.id: node for node in workflow.nodes}
        artifacts_by_step: dict[str, Artifact] = {}
        artifact_registry = ArtifactRegistry()

        if prompt is None:
            prompt = next(
                (
                    value
                    for step_id, value in external_inputs.items()
                    if step_id in workflow.inputs and isinstance(value, str)
                ),
                "",
            )

        trajectory = Trajectory(
            id=f"trajectory:{actual_run_id}",
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            prompt=prompt,
            input_metadata={"external_input_ids": sorted(external_inputs)},
        )

        for node_id in plan.order:
            node = node_by_id[node_id]
            implementation = self.registry.resolve(node.type, node.backend)
            backend = self.registry.backend_name(node.type, node.backend)
            params = resolve_parameters(node, self.registry)
            source_ids = plan.incoming[node_id]
            input_artifacts = [artifacts_by_step[source_id] for source_id in source_ids]
            context = StepContext(
                run_id=actual_run_id,
                step_id=node_id,
                backend=backend,
                external_inputs=external_inputs,
            )
            started_at = datetime.now(timezone.utc)
            started_counter = perf_counter()
            execution = StepExecution(
                id=f"execution:{actual_run_id}:{node_id}",
                step_id=node_id,
                step_type=node.type,
                backend=backend,
                parameters=params,
                input_artifact_ids=[artifact.id for artifact in input_artifacts],
                start_time=started_at.isoformat(),
            )

            try:
                step_result = implementation.run(input_artifacts, params, context)
                if isinstance(step_result, StepResult):
                    artifact = step_result.primary
                    produced_artifacts = step_result.all_artifacts()
                elif isinstance(step_result, Artifact):
                    artifact = step_result
                    produced_artifacts = [artifact]
                else:
                    raise TypeError(
                        f"{node_id}: Step returned unsupported value "
                        f"{type(step_result).__name__}"
                    )

                expected_output = implementation.schema.output_type
                if (
                    expected_output != "*"
                    and artifact.artifact_type != expected_output
                ):
                    raise TypeError(
                        f"{node_id}: Step returned {artifact.artifact_type}, "
                        f"expected {expected_output}"
                    )

                for produced in produced_artifacts:
                    artifact_registry.add(produced)
                artifacts_by_step[node_id] = artifact
                execution.output_artifact_ids = [
                    produced.id for produced in produced_artifacts
                ]
                trajectory.artifacts.extend(produced_artifacts)
                for produced in produced_artifacts:
                    trajectory.evaluations.extend(produced.scores)

                execution_identity = artifact.metadata.get("execution_identity")
                if isinstance(execution_identity, dict):
                    execution.metadata["execution_identity"] = execution_identity
                    trajectory.metadata.setdefault("execution_identity", {})[
                        node_id
                    ] = execution_identity

                if node.type in {"SELECT", "SELECT_PARTS"}:
                    trajectory.selections.append(
                        {
                            "step_id": node_id,
                            "input_artifact_ids": execution.input_artifact_ids,
                            "output_artifact_id": artifact.id,
                        }
                    )
            except Exception as exc:
                execution.error = f"{type(exc).__name__}: {exc}"
                trajectory.errors.append(execution.error)
                execution.duration_seconds = perf_counter() - started_counter
                trajectory.executions.append(execution)
                raise RuntimeExecutionError(
                    f"Step {node_id} failed: {exc}",
                    trajectory,
                ) from exc

            execution.duration_seconds = perf_counter() - started_counter
            trajectory.executions.append(execution)

        outputs = {
            output_id: artifacts_by_step[output_id]
            for output_id in workflow.outputs
        }
        trajectory.final_output_ids = [
            artifact.id for artifact in outputs.values()
        ]
        trajectory.timing = {
            "step_count": len(trajectory.executions),
            "total_duration_seconds": sum(
                execution.duration_seconds or 0.0
                for execution in trajectory.executions
            ),
        }
        return RuntimeResult(trajectory=trajectory, outputs=outputs)
