"""Headless-testable workflow workbench state and actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from image_drawer.core import Artifact, StepSpec, WorkflowSpec
from image_drawer.dsl import parse_workflow, serialize_workflow
from image_drawer.runtime import RuntimeResult, WorkflowRuntime, validate_workflow
from image_drawer.steps import StepRegistry
from image_drawer.runtime.validation import resolve_parameters


DEFAULT_WORKFLOW_DSL = """INPUT prompt: Text

parts = RETRIEVE_PARTS(
  prompt,
  category="generic",
  top=20,
)

selected = SELECT_PARTS(
  parts,
  top=5,
)

draft = COMPOSE(
  selected,
)

scores = EVALUATE(
  draft,
  evaluator="mock",
)

OUTPUT draft
"""


@dataclass(frozen=True, slots=True)
class ArtifactRow:
    id: str
    artifact_type: str
    step_id: str | None
    uri: str | None
    parents: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScoreRow:
    score_id: str
    artifact_id: str | None
    evaluator: str
    evaluator_version: str | None
    overall: float | None
    components: dict[str, float]


@dataclass(frozen=True, slots=True)
class CandidateRow:
    artifact_id: str
    status: str
    score_overall: float | None
    components: dict[str, float]
    uri: str | None = None


@dataclass(slots=True)
class RunRecord:
    run_id: str
    prompt: str
    result: RuntimeResult
    artifacts: list[ArtifactRow] = field(default_factory=list)
    scores: list[ScoreRow] = field(default_factory=list)
    candidates: list[CandidateRow] = field(default_factory=list)

    @classmethod
    def from_result(
        cls,
        run_id: str,
        prompt: str,
        result: RuntimeResult,
    ) -> "RunRecord":
        artifacts = [
            ArtifactRow(
                id=artifact.id,
                artifact_type=artifact.artifact_type,
                step_id=artifact.producing_step_id,
                uri=artifact.uri,
                parents=tuple(artifact.parent_artifact_ids),
                metadata=dict(artifact.metadata),
            )
            for artifact in result.trajectory.artifacts
        ]
        scores = [
            ScoreRow(
                score_id=score.id,
                artifact_id=(
                    score.metadata.get("artifact_id")
                    if isinstance(score.metadata.get("artifact_id"), str)
                    else None
                ),
                evaluator=score.evaluator,
                evaluator_version=score.evaluator_version,
                overall=score.overall,
                components=dict(score.components),
            )
            for score in result.trajectory.evaluations
        ]

        selected_ids: set[str] = set()
        for selection in result.trajectory.selections:
            decision = selection.get("decision")
            if isinstance(decision, dict):
                values = decision.get("selected_artifact_ids", [])
                if isinstance(values, list):
                    selected_ids.update(
                        value for value in values if isinstance(value, str)
                    )

        candidate_artifacts: dict[str, Artifact] = {
            artifact.id: artifact
            for artifact in result.trajectory.artifacts
            if artifact.artifact_type == "Image"
        }
        for artifact in result.trajectory.artifacts:
            if artifact.artifact_type != "ImageSet":
                continue
            payload = artifact.metadata.get("candidates")
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                try:
                    candidate = Artifact.from_dict(item)
                except Exception:
                    continue
                if candidate.artifact_type == "Image":
                    candidate_artifacts.setdefault(candidate.id, candidate)

        scored_ids: list[str] = []
        score_by_artifact: dict[str, ScoreRow] = {}
        for artifact in result.trajectory.artifacts:
            if artifact.artifact_type != "ScoreSet":
                continue
            payload = artifact.metadata.get("score_set")
            if not isinstance(payload, dict):
                continue
            candidate_ids = payload.get("candidate_artifact_ids", [])
            score_payloads = payload.get("scores", [])
            if not isinstance(candidate_ids, list) or not isinstance(
                score_payloads, list
            ):
                continue
            for candidate_id, score_payload in zip(
                candidate_ids, score_payloads
            ):
                if not isinstance(candidate_id, str) or not isinstance(
                    score_payload, dict
                ):
                    continue
                scored_ids.append(candidate_id)
                score_by_artifact[candidate_id] = ScoreRow(
                    score_id=str(score_payload.get("id", "")),
                    artifact_id=candidate_id,
                    evaluator=str(score_payload.get("evaluator", "")),
                    evaluator_version=score_payload.get("evaluator_version"),
                    overall=score_payload.get("overall"),
                    components=dict(score_payload.get("components", {})),
                )

        candidates = []
        for candidate_id in dict.fromkeys(scored_ids):
            score = score_by_artifact.get(candidate_id)
            candidates.append(
                CandidateRow(
                    artifact_id=candidate_id,
                    status=(
                        "selected"
                        if candidate_id in selected_ids
                        else "rejected"
                        if selected_ids
                        else "scored"
                    ),
                    score_overall=score.overall if score else None,
                    components=dict(score.components) if score else {},
                    uri=(
                        candidate_artifacts[candidate_id].uri
                        if candidate_id in candidate_artifacts
                        else None
                    ),
                )
            )
        return cls(
            run_id=run_id,
            prompt=prompt,
            result=result,
            artifacts=artifacts,
            scores=scores,
            candidates=candidates,
        )


class WorkbenchController:
    """Mutable GUI state; no Tk dependency so CI can exercise all actions."""

    def __init__(
        self,
        registry: StepRegistry,
        runtime: WorkflowRuntime | None = None,
        *,
        initial_dsl: str = DEFAULT_WORKFLOW_DSL,
        bank_dir: str | Path | None = None,
    ) -> None:
        self.registry = registry
        self.runtime = runtime or WorkflowRuntime(registry)
        self.bank_dir = Path(bank_dir).resolve() if bank_dir else None
        self.workflow = parse_workflow(
            initial_dsl,
            self.registry,
            workflow_id="gui-workflow",
        )
        self.validation_error: str | None = None
        self.run_history: list[RunRecord] = []
        self._run_counter = 0

    def canonical_dsl(self) -> str:
        return serialize_workflow(self.workflow, self.registry)

    def apply_dsl(self, source: str) -> WorkflowSpec:
        try:
            workflow = parse_workflow(
                source,
                self.registry,
                workflow_id=self.workflow.id,
                version=self.workflow.version,
            )
        except Exception as exc:
            self.validation_error = str(exc)
            raise
        self.workflow = workflow
        self.validation_error = None
        return workflow

    def load_workflow(self, path: str | Path) -> WorkflowSpec:
        source = Path(path).read_text(encoding="utf-8")
        workflow = self.apply_dsl(source)
        return workflow

    def save_workflow(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.canonical_dsl(), encoding="utf-8")

    def validate(self) -> str | None:
        try:
            validate_workflow(self.workflow, self.registry)
        except Exception as exc:
            self.validation_error = str(exc)
            return self.validation_error
        self.validation_error = None
        return None

    def step_ids(self) -> list[str]:
        return [node.id for node in self.workflow.nodes]

    def get_step(self, step_id: str) -> StepSpec:
        for node in self.workflow.nodes:
            if node.id == step_id:
                return node
        raise KeyError(f"unknown Step: {step_id}")

    def schema_for(self, step_id: str):
        step = self.get_step(step_id)
        return self.registry.resolve(step.type, step.backend).schema

    def parameter_values(self, step_id: str) -> dict[str, Any]:
        return resolve_parameters(self.get_step(step_id), self.registry)

    def set_parameter(
        self,
        step_id: str,
        name: str,
        value: Any,
    ) -> None:
        step = self.get_step(step_id)
        schema = self.registry.resolve(step.type, step.backend).schema
        if name not in schema.parameters:
            raise KeyError(f"{step_id}: unknown parameter {name}")
        previous = dict(step.parameters)
        step.parameters[name] = value
        try:
            resolve_parameters(step, self.registry)
            validate_workflow(self.workflow, self.registry)
        except Exception:
            step.parameters = previous
            raise
        self.validation_error = None

    def set_parameter_text(
        self,
        step_id: str,
        name: str,
        text: str,
    ) -> None:
        schema = self.schema_for(step_id)
        spec = schema.parameters[name]
        expected = (
            spec.value_type
            if isinstance(spec.value_type, tuple)
            else (spec.value_type,)
        )
        value: Any
        if bool in expected:
            lowered = text.strip().lower()
            if lowered not in {"true", "false"}:
                raise ValueError("boolean parameters must be true or false")
            value = lowered == "true"
        elif int in expected and float not in expected:
            value = int(text)
        elif float in expected and int not in expected:
            value = float(text)
        elif list in expected:
            value = json.loads(text)
            if not isinstance(value, list):
                raise ValueError("list parameter must be a JSON list")
        else:
            value = text
        self.set_parameter(step_id, name, value)

    def _unique_step_id(self, step_type: str) -> str:
        base = step_type.lower()
        used = set(self.step_ids())
        if base not in used:
            return base
        index = 2
        while f"{base}_{index}" in used:
            index += 1
        return f"{base}_{index}"

    def add_step(
        self,
        step_type: str,
        *,
        after_step_id: str | None = None,
    ) -> str:
        schema = self.registry.resolve(step_type).schema
        step_id = self._unique_step_id(step_type)
        parameters = {
            name: spec.default
            for name, spec in schema.parameters.items()
            if spec.has_default
        }
        step = StepSpec(id=step_id, type=step_type, parameters=parameters)

        if after_step_id is None:
            insert_index = len(self.workflow.nodes)
        else:
            insert_index = self.step_ids().index(after_step_id) + 1
        self.workflow.nodes.insert(insert_index, step)

        if after_step_id is not None and len(schema.input_types) == 1:
            outgoing = [
                edge
                for edge in self.workflow.edges
                if edge[0] == after_step_id
            ]
            if len(outgoing) <= 1:
                if outgoing:
                    successor = outgoing[0][1]
                    self.workflow.edges.remove(outgoing[0])
                    self.workflow.edges.append((after_step_id, step_id))
                    self.workflow.edges.append((step_id, successor))
                else:
                    self.workflow.edges.append((after_step_id, step_id))

        self.validate()
        return step_id

    def delete_step(self, step_id: str) -> None:
        step = self.get_step(step_id)
        if step.type in {"INPUT", "OUTPUT"}:
            raise ValueError("INPUT/OUTPUT Steps cannot be deleted in GUI v0")
        incoming = [edge for edge in self.workflow.edges if edge[1] == step_id]
        outgoing = [edge for edge in self.workflow.edges if edge[0] == step_id]
        self.workflow.nodes.remove(step)
        self.workflow.edges = [
            edge
            for edge in self.workflow.edges
            if step_id not in edge
        ]
        if len(incoming) == 1 and len(outgoing) == 1:
            reconnect = (incoming[0][0], outgoing[0][1])
            if reconnect not in self.workflow.edges:
                self.workflow.edges.append(reconnect)
        self.validate()

    def move_step(self, step_id: str, delta: int) -> None:
        if delta not in {-1, 1}:
            raise ValueError("delta must be -1 or 1")
        index = self.step_ids().index(step_id)
        target = index + delta
        if target < 0 or target >= len(self.workflow.nodes):
            return
        self.workflow.nodes[index], self.workflow.nodes[target] = (
            self.workflow.nodes[target],
            self.workflow.nodes[index],
        )

    def run(self, prompt: str) -> RunRecord:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        error = self.validate()
        if error is not None:
            raise ValueError(error)
        input_ids = list(self.workflow.inputs)
        if len(input_ids) != 1:
            raise ValueError(
                "GUI v0 Run supports exactly one external workflow input"
            )
        self._run_counter += 1
        run_id = f"gui-run-{self._run_counter:04d}"
        result = self.runtime.execute(
            self.workflow,
            external_inputs={input_ids[0]: prompt},
            run_id=run_id,
            prompt=prompt,
        )
        record = RunRecord.from_result(run_id, prompt, result)
        self.run_history.insert(0, record)
        return record

    def artifact_path(self, artifact: ArtifactRow) -> Path | None:
        if artifact.uri is None:
            return None
        path = Path(artifact.uri)
        if path.is_absolute():
            return path
        if self.bank_dir is None:
            return None
        resolved = (self.bank_dir / path).resolve()
        if not resolved.is_relative_to(self.bank_dir):
            return None
        return resolved
