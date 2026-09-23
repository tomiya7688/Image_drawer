from image_drawer.core import StepSpec, WorkflowSpec
from image_drawer.runtime import (
    RuntimeExecutionError,
    WorkflowRuntime,
    WorkflowValidationError,
    validate_workflow,
)
from image_drawer.steps import Step, StepContext, StepRegistry, StepSchema, create_mock_registry


def build_full_mock_workflow():
    return WorkflowSpec(
        id="workflow-mock",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(
                id="retrieve",
                type="RETRIEVE_PARTS",
                parameters={"category": "generic", "top": 4},
            ),
            StepSpec(
                id="select_parts",
                type="SELECT_PARTS",
                parameters={"top": 2},
            ),
            StepSpec(id="compose", type="COMPOSE"),
            StepSpec(id="evaluate", type="EVALUATE"),
            StepSpec(id="select", type="SELECT", parameters={"top": 1}),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "retrieve"),
            ("retrieve", "select_parts"),
            ("select_parts", "compose"),
            ("compose", "evaluate"),
            ("compose", "select"),
            ("evaluate", "select"),
            ("select", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
    )


def test_full_mock_workflow_validates_and_executes_end_to_end():
    registry = create_mock_registry()
    workflow = build_full_mock_workflow()

    plan = validate_workflow(workflow, registry)
    assert plan.order == (
        "prompt",
        "retrieve",
        "select_parts",
        "compose",
        "evaluate",
        "select",
        "output",
    )

    result = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "cat"},
        run_id="runtime-test",
    )

    trajectory = result.trajectory
    assert trajectory.prompt == "cat"
    assert [execution.step_id for execution in trajectory.executions] == list(
        plan.order
    )
    assert len(trajectory.artifacts) == len(workflow.nodes)
    assert all(execution.error is None for execution in trajectory.executions)
    assert len(trajectory.evaluations) == 1
    assert trajectory.evaluations[0].components["determinism"] == 1.0
    assert [selection["step_id"] for selection in trajectory.selections] == [
        "select_parts",
        "select",
    ]

    output = result.outputs["output"]
    assert output.artifact_type == "Image"
    assert output.metadata["value"] == (
        "mock-image[generic:cat:0|generic:cat:1]"
    )
    assert output.id == "artifact:runtime-test:output"
    assert trajectory.final_output_ids == [output.id]

    by_step = {
        artifact.producing_step_id: artifact
        for artifact in trajectory.artifacts
    }
    assert by_step["retrieve"].parent_artifact_ids == [by_step["prompt"].id]
    assert by_step["select"].parent_artifact_ids == [
        by_step["compose"].id,
        by_step["evaluate"].id,
    ]
    assert by_step["output"].parent_artifact_ids == [by_step["select"].id]


def test_validator_rejects_incompatible_artifact_types():
    workflow = WorkflowSpec(
        id="invalid-types",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(id="compose", type="COMPOSE"),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "compose"),
            ("compose", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
    )

    try:
        validate_workflow(workflow, create_mock_registry())
    except WorkflowValidationError as error:
        assert "expects PartSet" in str(error)
        assert "got Text" in str(error)
    else:
        raise AssertionError("invalid artifact type should be rejected")


def test_validator_rejects_cycles():
    workflow = WorkflowSpec(
        id="cycle",
        version="v1",
        nodes=[
            StepSpec(id="a", type="OUTPUT"),
            StepSpec(id="b", type="OUTPUT"),
        ],
        edges=[("a", "b"), ("b", "a")],
    )

    try:
        validate_workflow(workflow, create_mock_registry())
    except WorkflowValidationError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("cycle should be rejected")


def test_validator_rejects_unknown_and_bad_parameter_types():
    registry = create_mock_registry()

    unknown = WorkflowSpec(
        id="unknown-param",
        version="v1",
        nodes=[
            StepSpec(
                id="retrieve",
                type="RETRIEVE_PARTS",
                parameters={"unknown": 1},
            ),
        ],
    )
    try:
        validate_workflow(unknown, registry)
    except WorkflowValidationError as error:
        assert "unknown parameters" in str(error)
    else:
        raise AssertionError("unknown parameter should be rejected")

    bad_type = WorkflowSpec(
        id="bad-type",
        version="v1",
        nodes=[
            StepSpec(
                id="retrieve",
                type="RETRIEVE_PARTS",
                parameters={"top": True},
            ),
        ],
    )
    try:
        validate_workflow(bad_type, registry)
    except WorkflowValidationError as error:
        assert "parameter top" in str(error)
    else:
        raise AssertionError("bool must not satisfy an integer parameter")


class UpperStep(Step):
    backend = "test"
    schema = StepSchema(
        step_type="UPPER",
        input_types=("Text",),
        output_type="Text",
    )

    def run(self, inputs, params, context: StepContext):
        return context.make_artifact(
            "Text",
            parent_artifact_ids=[inputs[0].id],
            metadata={"value": inputs[0].metadata["value"].upper()},
        )


def test_new_step_can_be_registered_without_runtime_changes():
    registry = create_mock_registry()
    registry.register(UpperStep(), default=True)
    workflow = WorkflowSpec(
        id="custom-step",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(id="upper", type="UPPER"),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "upper"),
            ("upper", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
    )

    result = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "hello"},
        run_id="custom-step-test",
    )

    assert result.outputs["output"].metadata["value"] == "HELLO"
    assert [execution.step_type for execution in result.trajectory.executions] == [
        "INPUT",
        "UPPER",
        "OUTPUT",
    ]


def test_runtime_records_step_failure_in_partial_trajectory():
    registry = create_mock_registry()
    workflow = WorkflowSpec(
        id="missing-input",
        version="v1",
        nodes=[StepSpec(id="prompt", type="INPUT")],
        inputs=["prompt"],
    )

    try:
        WorkflowRuntime(registry).execute(
            workflow,
            external_inputs={},
            run_id="failure-test",
        )
    except RuntimeExecutionError as error:
        trajectory = error.trajectory
        assert len(trajectory.executions) == 1
        assert trajectory.executions[0].step_id == "prompt"
        assert "missing external input" in trajectory.executions[0].error
        assert trajectory.errors == [trajectory.executions[0].error]
    else:
        raise AssertionError("runtime failure should surface with trajectory")


def test_generic_evaluation_records_raw_score_and_selection_decision_once():
    registry = create_mock_registry()
    workflow = build_full_mock_workflow()

    result = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "cat"},
        run_id="evaluation-runtime-test",
    )

    trajectory = result.trajectory
    assert len(trajectory.evaluations) == 1
    score = trajectory.evaluations[0]
    assert score.evaluator == "mock"
    assert score.evaluator_version == "v1"
    assert set(score.components) == {
        "quality",
        "prompt_alignment",
        "determinism",
    }

    select_record = next(
        item for item in trajectory.selections
        if item["step_id"] == "select"
    )
    decision = select_record["decision"]
    assert decision["strategy"] == "best"
    assert decision["selected_artifact_ids"] == [
        "artifact:evaluation-runtime-test:compose"
    ]
    assert decision["evaluator"] == "mock"
    assert decision["evaluator_version"] == "v1"

    evaluate_execution = next(
        item for item in trajectory.executions
        if item.step_id == "evaluate"
    )
    assert evaluate_execution.metadata["execution_identity"] == {
        "evaluator": {"name": "mock", "version": "v1"}
    }
