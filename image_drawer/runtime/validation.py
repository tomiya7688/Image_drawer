"""WorkflowSpecのstatic validationとdependency resolution。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from image_drawer.core import StepSpec, WorkflowSpec
from image_drawer.steps import ParameterSpec, StepRegistry


class WorkflowValidationError(ValueError):
    """Workflowを安全に実行できない場合に送出する。"""


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    order: tuple[str, ...]
    incoming: dict[str, tuple[str, ...]]


def _matches_type(actual: str, expected: str | tuple[str, ...]) -> bool:
    if actual == "*" or expected == "*":
        return True
    if isinstance(expected, tuple):
        return actual in expected
    return actual == expected


def _parameter_matches(value: Any, spec: ParameterSpec) -> bool:
    expected = spec.value_type
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if int in expected_types and bool not in expected_types and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def resolve_parameters(step: StepSpec, registry: StepRegistry) -> dict[str, Any]:
    """指定parameterをvalidateし、schema defaultを適用する。"""
    implementation = registry.resolve(step.type, step.backend)
    schema = implementation.schema

    unknown = set(step.parameters) - set(schema.parameters)
    if unknown:
        raise WorkflowValidationError(
            f"{step.id}: unknown parameters: {', '.join(sorted(unknown))}"
        )

    resolved: dict[str, Any] = {}
    for name, parameter_spec in schema.parameters.items():
        if name in step.parameters:
            value = step.parameters[name]
        elif parameter_spec.has_default:
            value = parameter_spec.default
        elif parameter_spec.required:
            raise WorkflowValidationError(
                f"{step.id}: missing required parameter {name}"
            )
        else:
            continue

        if not _parameter_matches(value, parameter_spec):
            expected = parameter_spec.value_type
            raise WorkflowValidationError(
                f"{step.id}: parameter {name} expected {expected}, "
                f"got {type(value).__name__}"
            )
        if (
            parameter_spec.choices is not None
            and value not in parameter_spec.choices
        ):
            raise WorkflowValidationError(
                f"{step.id}: parameter {name} must be one of "
                f"{parameter_spec.choices}"
            )
        resolved[name] = value

    return resolved


def validate_workflow(
    workflow: WorkflowSpec,
    registry: StepRegistry,
) -> ExecutionPlan:
    """graph構造、Step contract、type、parameterをvalidateする。"""
    nodes: dict[str, StepSpec] = {}
    declaration_index: dict[str, int] = {}
    for index, node in enumerate(workflow.nodes):
        if node.id in nodes:
            raise WorkflowValidationError(f"duplicate Step id: {node.id}")
        nodes[node.id] = node
        declaration_index[node.id] = index
        try:
            registry.resolve(node.type, node.backend)
        except KeyError as exc:
            raise WorkflowValidationError(f"{node.id}: {exc.args[0]}") from exc
        resolve_parameters(node, registry)

    if not nodes:
        raise WorkflowValidationError("workflow has no Steps")

    incoming_lists: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    seen_edges: set[tuple[str, str]] = set()

    for source, target in workflow.edges:
        edge = (source, target)
        if edge in seen_edges:
            raise WorkflowValidationError(f"duplicate edge: {source} -> {target}")
        seen_edges.add(edge)
        if source not in nodes:
            raise WorkflowValidationError(f"edge references unknown source: {source}")
        if target not in nodes:
            raise WorkflowValidationError(f"edge references unknown target: {target}")
        incoming_lists[target].append(source)
        outgoing[source].append(target)

    indegree = {node_id: len(sources) for node_id, sources in incoming_lists.items()}
    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=declaration_index.__getitem__,
    )
    order: list[str] = []

    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=declaration_index.__getitem__)

    if len(order) != len(nodes):
        raise WorkflowValidationError("workflow contains a cycle")

    output_types: dict[str, str] = {}
    for node_id in order:
        node = nodes[node_id]
        schema = registry.resolve(node.type, node.backend).schema
        sources = incoming_lists[node_id]

        if len(sources) != len(schema.input_types):
            raise WorkflowValidationError(
                f"{node_id}: expected {len(schema.input_types)} inputs, "
                f"got {len(sources)}"
            )

        for position, (source_id, expected_type) in enumerate(
            zip(sources, schema.input_types)
        ):
            actual_type = output_types[source_id]
            if not _matches_type(actual_type, expected_type):
                raise WorkflowValidationError(
                    f"{node_id}: input {position} expects {expected_type}, "
                    f"got {actual_type} from {source_id}"
                )

        output_types[node_id] = schema.output_type

    for input_id in workflow.inputs:
        if input_id not in nodes:
            raise WorkflowValidationError(
                f"workflow input references unknown Step: {input_id}"
            )
        if nodes[input_id].type != "INPUT":
            raise WorkflowValidationError(
                f"workflow input {input_id} must reference an INPUT Step"
            )

    for output_id in workflow.outputs:
        if output_id not in nodes:
            raise WorkflowValidationError(
                f"workflow output references unknown Step: {output_id}"
            )
        if nodes[output_id].type != "OUTPUT":
            raise WorkflowValidationError(
                f"workflow output {output_id} must reference an OUTPUT Step"
            )

    return ExecutionPlan(
        order=tuple(order),
        incoming={
            node_id: tuple(sources)
            for node_id, sources in incoming_lists.items()
        },
    )
