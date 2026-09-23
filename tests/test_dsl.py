from pathlib import Path

from image_drawer.core import Artifact
from image_drawer.dsl import (
    DslSyntaxError,
    DslValidationError,
    parse_workflow,
    serialize_workflow,
)
from image_drawer.runtime import WorkflowRuntime
from image_drawer.steps import (
    ParameterSpec,
    Step,
    StepContext,
    StepSchema,
    create_mock_registry,
)

FIXTURES = Path(__file__).parent / "fixtures" / "workflows"


def test_basic_fixture_parses_serializes_and_round_trips_stably():
    registry = create_mock_registry()
    source = (FIXTURES / "valid_basic.dsl").read_text(encoding="utf-8")

    workflow = parse_workflow(source, registry)
    canonical = serialize_workflow(workflow, registry)

    assert canonical == """INPUT prompt: Text

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

    reparsed = parse_workflow(canonical, registry)
    assert serialize_workflow(reparsed, registry) == canonical
    assert reparsed.edges == workflow.edges


def test_parsed_workflow_executes_in_runtime():
    registry = create_mock_registry()
    source = (FIXTURES / "valid_basic.dsl").read_text(encoding="utf-8")
    workflow = parse_workflow(source, registry)

    result = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "cat"},
        run_id="dsl-runtime",
    )

    output = next(iter(result.outputs.values()))
    assert output.artifact_type == "Image"
    assert output.metadata["value"].startswith("mock-image[generic:cat:")
    assert [execution.step_type for execution in result.trajectory.executions] == [
        "INPUT",
        "RETRIEVE_PARTS",
        "SELECT_PARTS",
        "COMPOSE",
        "EVALUATE",
        "OUTPUT",
    ]


def test_invalid_fixture_reports_undefined_variable_with_line():
    registry = create_mock_registry()
    source = (FIXTURES / "invalid_undefined.dsl").read_text(encoding="utf-8")

    try:
        parse_workflow(source, registry)
    except DslSyntaxError as error:
        assert "undefined variable 'parts'" in str(error)
        assert "line 3" in str(error)
    else:
        raise AssertionError("undefined variable should fail parsing")


def test_invalid_fixture_reports_bad_parameter_type():
    registry = create_mock_registry()
    source = (FIXTURES / "invalid_parameter.dsl").read_text(encoding="utf-8")

    try:
        parse_workflow(source, registry)
    except DslValidationError as error:
        assert "parameter top" in str(error)
        assert "str" in str(error)
    else:
        raise AssertionError("bad parameter type should fail validation")


class ParamsStep(Step):
    backend = "test"
    schema = StepSchema(
        step_type="PARAMS",
        input_types=("Text",),
        output_type="Text",
        parameters={
            "text": ParameterSpec(str, required=True),
            "integer": ParameterSpec(int, required=True),
            "ratio": ParameterSpec(float, required=True),
            "enabled": ParameterSpec(bool, required=True),
            "values": ParameterSpec(list, required=True),
        },
    )

    def run(
        self,
        inputs: list[Artifact],
        params: dict,
        context: StepContext,
    ) -> Artifact:
        return context.make_artifact(
            "Text",
            parent_artifact_ids=[inputs[0].id],
            metadata={"value": params},
        )


def test_all_v1_primitive_parameter_types_are_supported():
    registry = create_mock_registry()
    registry.register(ParamsStep(), default=True)
    source = """INPUT prompt: Text

value = PARAMS(
  prompt,
  text="hello",
  integer=-2,
  ratio=1.5,
  enabled=true,
  values=[1, "x", false, -3.25],
)

OUTPUT value
"""
    workflow = parse_workflow(source, registry)
    step = next(node for node in workflow.nodes if node.id == "value")

    assert step.parameters == {
        "text": "hello",
        "integer": -2,
        "ratio": 1.5,
        "enabled": True,
        "values": [1, "x", False, -3.25],
    }

    canonical = serialize_workflow(workflow, registry)
    assert 'text="hello"' in canonical
    assert "integer=-2" in canonical
    assert "ratio=1.5" in canonical
    assert "enabled=true" in canonical
    assert 'values=[1, "x", false, -3.25]' in canonical


def test_named_artifact_dependency_is_preserved():
    registry = create_mock_registry()
    source = """INPUT prompt: Text

parts = RETRIEVE_PARTS(prompt, top=2)
draft = COMPOSE(parts)
scores = EVALUATE(draft)
best = SELECT(draft, scores=scores, top=1)

OUTPUT best
"""

    workflow = parse_workflow(source, registry)
    canonical = serialize_workflow(workflow, registry)

    assert "  draft," in canonical
    assert "  scores=scores," in canonical
    assert ("draft", "best") in workflow.edges
    assert ("scores", "best") in workflow.edges


def test_comments_are_accepted_and_removed_by_canonical_formatting():
    registry = create_mock_registry()
    source = """# 前
INPUT prompt: Text  # input宣言

parts = RETRIEVE_PARTS(
  prompt,  # dependency
  top=1,   # parameter
)
# call後
OUTPUT parts
"""

    canonical = serialize_workflow(parse_workflow(source, registry), registry)

    assert "#" not in canonical
    assert canonical.endswith("OUTPUT parts\n")


def test_arbitrary_expression_and_control_flow_are_rejected():
    registry = create_mock_registry()

    expression = """INPUT prompt: Text
parts = RETRIEVE_PARTS(prompt, top=1 + 1)
OUTPUT parts
"""
    try:
        parse_workflow(expression, registry)
    except DslSyntaxError as error:
        assert "arbitrary expressions" in str(error)
    else:
        raise AssertionError("arithmetic expression should be rejected")

    control_flow = """INPUT prompt: Text
if prompt:
    OUTPUT prompt
"""
    try:
        parse_workflow(control_flow, registry)
    except DslSyntaxError as error:
        assert "only INPUT" in str(error)
    else:
        raise AssertionError("control flow should be rejected")


def test_parameter_cannot_be_bound_to_a_variable():
    registry = create_mock_registry()
    source = """INPUT prompt: Text
parts = RETRIEVE_PARTS(prompt, top=prompt)
OUTPUT parts
"""

    try:
        parse_workflow(source, registry)
    except DslSyntaxError as error:
        assert "parameter 'top' requires a literal" in str(error)
    else:
        raise AssertionError("parameter variable binding should be rejected")
