"""Image Drawer WorkflowSpec record向けcanonical serializer。"""

from __future__ import annotations

import json
from typing import Any

from image_drawer.core import StepSpec, WorkflowSpec
from image_drawer.runtime import validate_workflow
from image_drawer.steps import StepRegistry


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TypeError(
        "DSL parameters support only string, integer, float, boolean, and list"
    )


def _dsl_metadata(step: StepSpec) -> dict[str, Any]:
    value = step.metadata.get("dsl", {})
    return value if isinstance(value, dict) else {}


def serialize_workflow(
    workflow: WorkflowSpec,
    registry: StepRegistry,
) -> str:
    """validation済みWorkflowSpecをstable canonical DSL textへserializeする。"""
    plan = validate_workflow(workflow, registry)
    node_by_id = {node.id: node for node in workflow.nodes}
    lines: list[str] = []

    for node_id in plan.order:
        step = node_by_id[node_id]
        schema = registry.resolve(step.type, step.backend).schema
        metadata = _dsl_metadata(step)

        if step.type == "INPUT":
            declared_type = metadata.get("input_type", schema.output_type)
            lines.append(f"INPUT {step.id}: {declared_type}")
            lines.append("")
            continue

        if step.type == "OUTPUT":
            source_ids = plan.incoming[step.id]
            if len(source_ids) != 1:
                raise ValueError(f"{step.id}: OUTPUT must have exactly one input")
            lines.append(f"OUTPUT {source_ids[0]}")
            lines.append("")
            continue

        incoming = list(plan.incoming[step.id])
        stored_bindings = metadata.get("inputs")
        bindings: list[dict[str, str | None]]
        if (
            isinstance(stored_bindings, list)
            and len(stored_bindings) == len(incoming)
            and all(isinstance(item, dict) for item in stored_bindings)
        ):
            bindings = stored_bindings
        else:
            bindings = [
                {"source": source_id, "name": None}
                for source_id in incoming
            ]

        parameter_names = list(schema.parameters)
        parameter_names.extend(
            sorted(
                name
                for name in step.parameters
                if name not in schema.parameters
            )
        )

        arguments: list[str] = []
        for binding, source_id in zip(bindings, incoming):
            name = binding.get("name")
            stored_source = binding.get("source")
            source = stored_source if stored_source == source_id else source_id
            if name:
                arguments.append(f"{name}={source}")
            else:
                arguments.append(source)

        for name in parameter_names:
            if name in step.parameters:
                arguments.append(
                    f"{name}={_format_value(step.parameters[name])}"
                )

        if not arguments:
            lines.append(f"{step.id} = {step.type}()")
            lines.append("")
            continue

        lines.append(f"{step.id} = {step.type}(")
        for argument in arguments:
            lines.append(f"  {argument},")
        lines.append(")")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
