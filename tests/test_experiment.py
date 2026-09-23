import csv
import json
from pathlib import Path

from image_drawer.core import StepSpec, WorkflowSpec
from image_drawer.search import (
    ExperimentResult,
    ExperimentRunner,
    ExperimentSpec,
    PromptCase,
    export_experiment,
    load_experiment,
    load_prompt_set,
)
from image_drawer.steps import (
    Step,
    StepContext,
    StepRegistry,
    StepSchema,
    create_mock_registry,
)


def workflow_a():
    return WorkflowSpec(
        id="workflow-a",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(
                id="parts",
                type="RETRIEVE_PARTS",
                parameters={"category": "generic", "top": 4},
            ),
            StepSpec(
                id="selected",
                type="SELECT_PARTS",
                parameters={"top": 2},
            ),
            StepSpec(id="draft", type="COMPOSE"),
            StepSpec(
                id="scores",
                type="EVALUATE",
                parameters={"evaluator": "mock"},
            ),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "parts"),
            ("parts", "selected"),
            ("selected", "draft"),
            ("draft", "scores"),
            ("draft", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
        metadata={"variant": "a"},
    )


def workflow_b():
    return WorkflowSpec(
        id="workflow-b",
        version="v2",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(
                id="parts",
                type="RETRIEVE_PARTS",
                parameters={"category": "generic", "top": 3},
            ),
            StepSpec(
                id="selected",
                type="SELECT_PARTS",
                parameters={"top": 1},
            ),
            StepSpec(id="draft", type="COMPOSE"),
            StepSpec(
                id="scores",
                type="EVALUATE",
                parameters={"evaluator": "mock"},
            ),
            StepSpec(
                id="best",
                type="SELECT",
                parameters={"strategy": "best", "top": 1, "weights": []},
            ),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "parts"),
            ("parts", "selected"),
            ("selected", "draft"),
            ("draft", "scores"),
            ("draft", "best"),
            ("scores", "best"),
            ("best", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
        metadata={"variant": "b"},
    )


def experiment_spec():
    return ExperimentSpec(
        id="comparison-fixture",
        prompt_set=[
            PromptCase(id="cat", prompt="cat"),
            PromptCase(id="dog", prompt="dog"),
        ],
        repeats=2,
        seed_policy="paired_increment",
        base_seed=100,
        metadata={
            "dataset_version": "fixture-v1",
            "index_version": "mock-v1",
        },
    )


def test_two_workflows_run_on_same_prompt_seed_schedule():
    registry = create_mock_registry()
    result = ExperimentRunner(
        registry,
        environment_metadata={
            "dataset_version": "fixture-v1",
            "host": "test",
        },
    ).run(
        experiment_spec(),
        {"a": workflow_a(), "b": workflow_b()},
    )

    assert len(result.runs) == 8
    assert set(result.aggregate_metrics) == {"a", "b"}
    assert all(run.status == "success" for run in result.runs)

    by_pair = {}
    for run in result.runs:
        key = (run.prompt_case_id, run.repeat_index)
        by_pair.setdefault(key, []).append(run)
    assert set(by_pair) == {
        ("cat", 0),
        ("cat", 1),
        ("dog", 0),
        ("dog", 1),
    }
    expected_seeds = {
        ("cat", 0): 100,
        ("cat", 1): 101,
        ("dog", 0): 102,
        ("dog", 1): 103,
    }
    for key, paired_runs in by_pair.items():
        assert {run.seed for run in paired_runs} == {expected_seeds[key]}
        for run in paired_runs:
            assert run.trajectory is not None
            assert run.trajectory.input_metadata["seed"] == expected_seeds[key]
            assert all(
                execution.seed == expected_seeds[key]
                for execution in run.trajectory.executions
            )

    assert result.metadata["comparison_schedule"] == {
        "prompt_case_ids": ["cat", "dog"],
        "repeats": 2,
        "seed_policy": "paired_increment",
        "base_seed": 100,
    }
    assert result.metadata["environment"]["dataset_version"] == "fixture-v1"


def test_workflow_snapshots_capture_dsl_versions_backends_and_hashes():
    registry = create_mock_registry()
    result = ExperimentRunner(registry).run(
        ExperimentSpec(
            id="snapshot",
            prompt_set=[PromptCase(id="one", prompt="one")],
        ),
        {"a": workflow_a(), "b": workflow_b()},
    )

    snapshots = {snapshot.key: snapshot for snapshot in result.workflows}
    assert snapshots["a"].workflow_id == "workflow-a"
    assert snapshots["a"].workflow_version == "v1"
    assert snapshots["b"].workflow_version == "v2"
    assert snapshots["a"].content_hash.startswith("sha256:")
    assert "parts = RETRIEVE_PARTS(" in snapshots["a"].canonical_dsl
    assert snapshots["a"].backends["scores"] == "evaluation"
    assert snapshots["b"].backends["best"] == "selection"

    observed = result.metadata["observed_versions"]
    assert observed["execution_identities"]["evaluator"] == [
        {"name": "mock", "version": "v1"}
    ]
    assert any(
        item["model"] == "mock-compose" and item["version"] == "v1"
        for item in observed["artifact_models"]
    )


def test_aggregate_metrics_are_computed_per_workflow():
    registry = create_mock_registry()
    result = ExperimentRunner(registry).run(
        experiment_spec(),
        {"a": workflow_a(), "b": workflow_b()},
    )

    for key in ("a", "b"):
        metric = result.aggregate_metrics[key]
        assert metric.workflow_key == key
        assert metric.run_count == 4
        assert metric.success_count == 4
        assert metric.failure_count == 0
        assert metric.failure_rate == 0.0
        assert metric.runtime_seconds_min >= 0
        assert metric.runtime_seconds_max >= metric.runtime_seconds_min
        assert metric.runtime_seconds_mean >= 0
        assert metric.evaluation_count == 4
        assert metric.score_overall_mean is not None
        assert "quality" in metric.component_means
        assert "prompt_alignment" in metric.component_means
        assert "determinism" in metric.component_means


class FailingStep(Step):
    backend = "test"
    schema = StepSchema(
        step_type="FAIL",
        input_types=("Text",),
        output_type="Text",
    )

    def run(self, inputs, params, context: StepContext):
        raise RuntimeError("intentional experiment failure")


def failing_workflow():
    return WorkflowSpec(
        id="workflow-failing",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(id="fail", type="FAIL"),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "fail"),
            ("fail", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
    )


def test_runtime_failure_is_recorded_without_aborting_comparison():
    registry = create_mock_registry()
    registry.register(FailingStep(), default=True)
    result = ExperimentRunner(registry).run(
        ExperimentSpec(
            id="failure-comparison",
            prompt_set=[PromptCase(id="one", prompt="one")],
            repeats=1,
            base_seed=9,
        ),
        {"healthy": workflow_a(), "failing": failing_workflow()},
    )

    assert len(result.runs) == 2
    healthy = next(run for run in result.runs if run.workflow_key == "healthy")
    failed = next(run for run in result.runs if run.workflow_key == "failing")
    assert healthy.status == "success"
    assert failed.status == "failure"
    assert "intentional experiment failure" in failed.error
    assert failed.trajectory is not None
    assert failed.trajectory.errors
    assert failed.trajectory.executions[-1].step_id == "fail"
    assert failed.trajectory.executions[-1].seed == 9

    metric = result.aggregate_metrics["failing"]
    assert metric.run_count == 1
    assert metric.success_count == 0
    assert metric.failure_count == 1
    assert metric.failure_rate == 1.0
    assert metric.metadata["failure_errors"]


def test_experiment_result_json_round_trip():
    registry = create_mock_registry()
    result = ExperimentRunner(registry).run(
        ExperimentSpec(
            id="round-trip",
            prompt_set=[PromptCase(id="one", prompt="one")],
        ),
        {"a": workflow_a(), "b": workflow_b()},
    )

    restored = ExperimentResult.from_json(result.to_json())
    assert restored == result


def test_export_persists_trajectories_workflows_and_csvs(tmp_path):
    registry = create_mock_registry()
    result = ExperimentRunner(registry).run(
        ExperimentSpec(
            id="export",
            prompt_set=[PromptCase(id="one", prompt="one")],
        ),
        {"a": workflow_a(), "b": workflow_b()},
    )

    root = export_experiment(result, tmp_path / "result")

    assert (root / "experiment.json").is_file()
    assert (root / "runs.csv").is_file()
    assert (root / "workflow_metrics.csv").is_file()
    assert (root / "workflows" / "a.dsl").is_file()
    assert (root / "workflows" / "b.dsl").is_file()
    assert len(list((root / "trajectories").glob("*.json"))) == 2

    loaded = load_experiment(root)
    assert loaded == result

    with (root / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["workflow_key"] for row in rows} == {"a", "b"}
    assert all(row["status"] == "success" for row in rows)

    with (root / "workflow_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert {row["workflow_key"] for row in metric_rows} == {"a", "b"}
    assert "component:quality" in metric_rows[0]


def test_load_prompt_set_supports_text_jsonl_and_json(tmp_path):
    text_path = tmp_path / "prompts.txt"
    text_path.write_text(
        "# comment\ncat portrait\n\n dog portrait \n",
        encoding="utf-8",
    )
    text_prompts = load_prompt_set(text_path)
    assert [item.prompt for item in text_prompts] == [
        "cat portrait",
        "dog portrait",
    ]
    assert text_prompts[0].metadata["source_line"] == 2

    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "id": "cat",
                "prompt": "cat",
                "metadata": {"group": "animal"},
            }
        )
        + "\n"
        + json.dumps("dog")
        + "\n",
        encoding="utf-8",
    )
    jsonl_prompts = load_prompt_set(jsonl_path)
    assert jsonl_prompts[0].id == "cat"
    assert jsonl_prompts[0].metadata == {"group": "animal"}
    assert jsonl_prompts[1].prompt == "dog"

    json_path = tmp_path / "prompts.json"
    json_path.write_text(
        json.dumps(
            [
                {"id": "a", "prompt": "alpha"},
                {"id": "b", "prompt": "beta"},
            ]
        ),
        encoding="utf-8",
    )
    json_prompts = load_prompt_set(json_path)
    assert [(item.id, item.prompt) for item in json_prompts] == [
        ("a", "alpha"),
        ("b", "beta"),
    ]


def test_fixed_and_none_seed_policies():
    registry = create_mock_registry()

    fixed = ExperimentRunner(registry).run(
        ExperimentSpec(
            id="fixed",
            prompt_set=[
                PromptCase(id="a", prompt="a"),
                PromptCase(id="b", prompt="b"),
            ],
            repeats=2,
            seed_policy="fixed",
            base_seed=77,
        ),
        {"a": workflow_a(), "b": workflow_b()},
    )
    assert {run.seed for run in fixed.runs} == {77}

    unseeded = ExperimentRunner(registry).run(
        ExperimentSpec(
            id="none",
            prompt_set=[PromptCase(id="a", prompt="a")],
            seed_policy="none",
            base_seed=None,
        ),
        {"a": workflow_a(), "b": workflow_b()},
    )
    assert {run.seed for run in unseeded.runs} == {None}
