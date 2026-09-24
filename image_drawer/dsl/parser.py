"""意図的に小さく保ったImage Drawer Workflow DSLのparser。"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from image_drawer.core import StepSpec, WorkflowSpec
from image_drawer.runtime import WorkflowValidationError, validate_workflow
from image_drawer.steps import StepRegistry

_INPUT_RE = re.compile(
    r"^\s*INPUT\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:#.*)?$"
)
_OUTPUT_RE = re.compile(
    r"^\s*OUTPUT\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:#.*)?$"
)
_STEP_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class DslError(ValueError):
    """parse失敗とDSL validation失敗の基底class。"""


@dataclass(eq=False)
class DslSyntaxError(DslError):
    message: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        location = ""
        if self.line is not None:
            location = f"line {self.line}"
            if self.column is not None:
                location += f", column {self.column}"
            location += ": "
        return location + self.message


class DslValidationError(DslError):
    """parse済み構文がWorkflow/Step contractへ違反した場合に送出する。"""


def _preprocess(source: str) -> str:
    """INPUT/OUTPUT宣言をPython形式のsentinel callへ変換する。

    AST診断を元DSL行へ直接対応させるため、行数は維持する。
    """
    transformed: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("INPUT"):
            match = _INPUT_RE.match(line)
            if match is None:
                raise DslSyntaxError(
                    "expected 'INPUT <name>: <Type>'",
                    line=line_number,
                )
            name, artifact_type = match.groups()
            indent = line[: len(line) - len(stripped)]
            transformed.append(
                f'{indent}{name} = __DSL_INPUT__("{artifact_type}")'
            )
            continue

        if stripped.startswith("OUTPUT"):
            match = _OUTPUT_RE.match(line)
            if match is None:
                raise DslSyntaxError(
                    "expected 'OUTPUT <name>'",
                    line=line_number,
                )
            name = match.group(1)
            indent = line[: len(line) - len(stripped)]
            transformed.append(f"{indent}__DSL_OUTPUT__({name})")
            continue

        transformed.append(line)

    return "\n".join(transformed)


def _literal(node: ast.AST) -> Any:
    """DSL v1がsupportするprimitive parameterだけをdecodeする。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)):
            return node.value
        raise DslSyntaxError(
            "parameters support only string, integer, float, boolean, and list",
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
        )

    if isinstance(node, ast.Name):
        if node.id in {"true", "True"}:
            return True
        if node.id in {"false", "False"}:
            return False
        raise DslSyntaxError(
            f"'{node.id}' is not a literal parameter value",
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
        )

    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value

    raise DslSyntaxError(
        "arbitrary expressions are not allowed in parameters",
        line=getattr(node, "lineno", None),
        column=getattr(node, "col_offset", None),
    )


def _name_reference(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id not in {
        "true",
        "false",
        "True",
        "False",
    }:
        return node.id
    return None


def parse_workflow(
    source: str,
    registry: StepRegistry,
    *,
    workflow_id: str = "dsl-workflow",
    version: str = "v1",
) -> WorkflowSpec:
    """DSL textをparseし、validation済みWorkflowSpecへ変換する。

    positional call argumentはArtifact dependencyとして扱う。
    Step parameter名と一致するnamed argumentはliteral parameterとする。
    その他のnamed argumentはscores=line_scoresのような名前付きdependencyとしてbindingできる。
    """
    try:
        tree = ast.parse(_preprocess(source), mode="exec")
    except SyntaxError as exc:
        raise DslSyntaxError(
            exc.msg,
            line=exc.lineno,
            column=exc.offset,
        ) from exc

    nodes: list[StepSpec] = []
    edges: list[tuple[str, str]] = []
    workflow_inputs: list[str] = []
    workflow_outputs: list[str] = []
    defined: set[str] = set()
    output_count = 0

    def require_defined(name: str, node: ast.AST) -> None:
        if name not in defined:
            raise DslSyntaxError(
                f"undefined variable '{name}'",
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
            )

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                raise DslSyntaxError(
                    "each Step must assign to exactly one variable",
                    line=statement.lineno,
                    column=statement.col_offset,
                )

            step_id = statement.targets[0].id
            if step_id in defined:
                raise DslSyntaxError(
                    f"variable '{step_id}' is already defined",
                    line=statement.lineno,
                    column=statement.col_offset,
                )
            if not isinstance(statement.value, ast.Call):
                raise DslSyntaxError(
                    "assignment right-hand side must be a Step call",
                    line=statement.lineno,
                    column=statement.col_offset,
                )
            call = statement.value
            if not isinstance(call.func, ast.Name):
                raise DslSyntaxError(
                    "Step calls must use a simple operation name",
                    line=call.lineno,
                    column=call.col_offset,
                )

            step_type = call.func.id
            if step_type == "__DSL_INPUT__":
                if call.keywords or len(call.args) != 1:
                    raise DslSyntaxError(
                        "invalid INPUT declaration",
                        line=call.lineno,
                        column=call.col_offset,
                    )
                declared_type = _literal(call.args[0])
                if not isinstance(declared_type, str):
                    raise DslSyntaxError(
                        "INPUT type must be an identifier",
                        line=call.lineno,
                        column=call.col_offset,
                    )
                try:
                    input_schema = registry.resolve("INPUT").schema
                except KeyError as exc:
                    raise DslValidationError(
                        "registry does not provide an INPUT Step"
                    ) from exc
                if (
                    input_schema.output_type != "*"
                    and declared_type != input_schema.output_type
                ):
                    raise DslValidationError(
                        f"{step_id}: INPUT declares {declared_type}, "
                        f"but registry INPUT produces {input_schema.output_type}"
                    )
                nodes.append(
                    StepSpec(
                        id=step_id,
                        type="INPUT",
                        metadata={"dsl": {"input_type": declared_type}},
                    )
                )
                workflow_inputs.append(step_id)
                defined.add(step_id)
                continue

            if not _STEP_TYPE_RE.match(step_type):
                raise DslSyntaxError(
                    f"invalid Step operation '{step_type}'",
                    line=call.lineno,
                    column=call.col_offset,
                )

            try:
                schema = registry.resolve(step_type).schema
            except KeyError as exc:
                raise DslValidationError(
                    f"{step_id}: unknown Step type: {step_type}"
                ) from exc

            input_bindings: list[dict[str, str | None]] = []
            parameters: dict[str, Any] = {}

            for argument in call.args:
                if isinstance(argument, ast.Starred):
                    raise DslSyntaxError(
                        "starred arguments are not supported",
                        line=argument.lineno,
                        column=argument.col_offset,
                    )
                reference = _name_reference(argument)
                if reference is None:
                    raise DslSyntaxError(
                        "positional Step arguments must reference variables",
                        line=getattr(argument, "lineno", None),
                        column=getattr(argument, "col_offset", None),
                    )
                require_defined(reference, argument)
                edges.append((reference, step_id))
                input_bindings.append({"source": reference, "name": None})

            for keyword in call.keywords:
                if keyword.arg is None:
                    raise DslSyntaxError(
                        "expanded keyword arguments are not supported",
                        line=keyword.value.lineno,
                        column=keyword.value.col_offset,
                    )

                if keyword.arg in schema.parameters:
                    try:
                        parameters[keyword.arg] = _literal(keyword.value)
                    except DslSyntaxError as exc:
                        reference = _name_reference(keyword.value)
                        if reference is not None:
                            raise DslSyntaxError(
                                f"parameter '{keyword.arg}' requires a literal "
                                f"value, not variable '{reference}'",
                                line=keyword.value.lineno,
                                column=keyword.value.col_offset,
                            ) from exc
                        raise
                    continue

                reference = _name_reference(keyword.value)
                if reference is not None:
                    require_defined(reference, keyword.value)
                    edges.append((reference, step_id))
                    input_bindings.append(
                        {"source": reference, "name": keyword.arg}
                    )
                    continue

                parameters[keyword.arg] = _literal(keyword.value)

            nodes.append(
                StepSpec(
                    id=step_id,
                    type=step_type,
                    parameters=parameters,
                    metadata={"dsl": {"inputs": input_bindings}},
                )
            )
            defined.add(step_id)
            continue

        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "__DSL_OUTPUT__"
        ):
            call = statement.value
            if call.keywords or len(call.args) != 1:
                raise DslSyntaxError(
                    "OUTPUT expects exactly one variable",
                    line=statement.lineno,
                    column=statement.col_offset,
                )
            reference = _name_reference(call.args[0])
            if reference is None:
                raise DslSyntaxError(
                    "OUTPUT expects a variable name",
                    line=statement.lineno,
                    column=statement.col_offset,
                )
            require_defined(reference, call.args[0])
            output_id = f"__dsl_output__:{output_count}"
            output_count += 1
            nodes.append(
                StepSpec(
                    id=output_id,
                    type="OUTPUT",
                    metadata={"dsl": {"output_source": reference}},
                )
            )
            edges.append((reference, output_id))
            workflow_outputs.append(output_id)
            continue

        raise DslSyntaxError(
            "only INPUT, Step assignments, OUTPUT, and comments are allowed",
            line=getattr(statement, "lineno", None),
            column=getattr(statement, "col_offset", None),
        )

    if not nodes:
        raise DslValidationError("workflow is empty")
    if not workflow_outputs:
        raise DslValidationError("workflow must declare at least one OUTPUT")

    workflow = WorkflowSpec(
        id=workflow_id,
        version=version,
        nodes=nodes,
        edges=edges,
        inputs=workflow_inputs,
        outputs=workflow_outputs,
    )
    try:
        validate_workflow(workflow, registry)
    except WorkflowValidationError as exc:
        raise DslValidationError(str(exc)) from exc

    return workflow
