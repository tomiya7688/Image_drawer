import json

from image_drawer.core import StepSpec, WorkflowSpec
from image_drawer.dsl import serialize_workflow
from image_drawer.search import (
    ExperimentSpec,
    ObjectiveSpec,
    ParameterDimension,
    ParameterSearchResult,
    ParameterSearchRunner,
    PromptCase,
    SearchSpec,
    export_parameter_search,
    generate_candidates,
    generate_configurations,
    load_parameter_search,
    reconstruct_best_workflow,
)
from image_drawer.steps import create_mock_registry


def searchable_workflow():
    return WorkflowSpec(
        id="searchable",
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
            StepSpec(
                id="compose",
                type="COMPOSE",
                parameters={"count": 1},
            ),
            StepSpec(
                id="scores",
                type="EVALUATE",
                parameters={"evaluator": "mock"},
            ),
            StepSpec(
                id="select",
                type="SELECT",
                parameters={
                    "strategy": "weighted",
                    "top": 1,
                    "weights": ["quality=1.0"],
                },
            ),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "parts"),
            ("parts", "selected"),
            ("selected", "compose"),
            ("compose", "scores"),
            ("compose", "select"),
            ("scores", "select"),
            ("select", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
    )


def experiment_spec():
    return ExperimentSpec(
        id="parameter-search-experiment",
        prompt_set=[
            PromptCase(id="cat", prompt="cat"),
            PromptCase(id="dog", prompt="dog"),
        ],
        repeats=1,
        seed_policy="paired_increment",
        base_seed=50,
    )


def test_grid_generates_initial_search_dimensions():
    spec = SearchSpec(
        id="grid",
        strategy="grid",
        dimensions=[
            ParameterDimension("parts", "top", [3, 4]),
            ParameterDimension("selected", "top", [1, 2]),
            ParameterDimension("compose", "count", [1, 2]),
            ParameterDimension(
                "select",
                "weights",
                [
                    ["quality=1.0"],
                    ["prompt_alignment=1.0"],
                ],
            ),
        ],
        objective=ObjectiveSpec(
            weights={"score_overall_mean": 1.0}
        ),
    )

    configurations = generate_configurations(spec)

    assert len(configurations) == 16
    assert {
        tuple(configuration["select.weights"])
        for configuration in configurations
    } == {
        ("quality=1.0",),
        ("prompt_alignment=1.0",),
    }
    assert {configuration["parts.top"] for configuration in configurations} == {
        3,
        4,
    }
    assert {
        configuration["selected.top"] for configuration in configurations
    } == {1, 2}
    assert {
        configuration["compose.count"] for configuration in configurations
    } == {1, 2}


def test_random_search_is_reproducible_and_unique():
    spec = SearchSpec(
        id="random",
        strategy="random",
        sample_count=5,
        random_seed=1234,
        dimensions=[
            ParameterDimension("parts", "top", [2, 3, 4]),
            ParameterDimension("selected", "top", [1, 2]),
            ParameterDimension("compose", "count", [1, 2]),
        ],
        objective=ObjectiveSpec(
            weights={"score_overall_mean": 1.0}
        ),
    )

    first = generate_configurations(spec)
    second = generate_configurations(spec)

    assert first == second
    assert len(first) == 5
    assert len(
        {
            json.dumps(item, sort_keys=True)
            for item in first
        }
    ) == 5


def test_invalid_parameter_candidate_is_rejected_before_execution():
    registry = create_mock_registry()
    spec = SearchSpec(
        id="invalid",
        strategy="grid",
        dimensions=[
            ParameterDimension("parts", "top", [2, 3, "bad"]),
        ],
        objective=ObjectiveSpec(
            weights={"score_overall_mean": 1.0}
        ),
    )

    candidates = generate_candidates(
        searchable_workflow(),
        spec,
        registry,
    )

    valid = [item for item in candidates if item.valid]
    invalid = [item for item in candidates if not item.valid]
    assert len(valid) == 2
    assert len(invalid) == 1
    assert invalid[0].parameters == {"parts.top": "bad"}
    assert "parameter top" in invalid[0].validation_error


def test_search_runs_only_valid_candidates_through_experiment_runner():
    registry = create_mock_registry()
    search_spec = SearchSpec(
        id="mixed-validity",
        strategy="grid",
        dimensions=[
            ParameterDimension("parts", "top", [2, 3, "bad"]),
        ],
        objective=ObjectiveSpec(
            weights={
                "score_overall_mean": 1.0,
                "failure_rate": -1.0,
            }
        ),
    )

    result = ParameterSearchRunner(registry).run(
        searchable_workflow(),
        search_spec,
        experiment_spec(),
    )

    assert len(result.candidates) == 3
    assert result.metadata["valid_candidate_count"] == 2
    assert result.metadata["invalid_candidate_count"] == 1
    assert result.experiment_result is not None
    assert len(result.experiment_result.workflows) == 2
    assert len(result.experiment_result.runs) == 4
    executed_keys = {
        run.workflow_key
        for run in result.experiment_result.runs
    }
    assert executed_keys == {
        candidate.key
        for candidate in result.candidates
        if candidate.valid
    }


def test_objective_preserves_raw_components_and_ranks_candidates():
    registry = create_mock_registry()
    search_spec = SearchSpec(
        id="objective",
        strategy="grid",
        dimensions=[
            ParameterDimension("parts", "top", [2, 4]),
            ParameterDimension("compose", "count", [1, 2]),
        ],
        objective=ObjectiveSpec(
            weights={
                "component:quality": 0.7,
                "component:prompt_alignment": 0.3,
                "failure_rate": -5.0,
            }
        ),
    )

    result = ParameterSearchRunner(registry).run(
        searchable_workflow(),
        search_spec,
        experiment_spec(),
    )

    valid = [candidate for candidate in result.candidates if candidate.valid]
    assert len(valid) == 4
    assert len(result.ranking) == 4
    assert result.best_candidate_key == result.ranking[0]
    assert set(result.ranking) == {candidate.key for candidate in valid}

    objective_values = {
        candidate.key: candidate.objective_value
        for candidate in valid
    }
    assert result.ranking == sorted(
        result.ranking,
        key=lambda key: (-objective_values[key], key),
    )
    for candidate in valid:
        assert set(candidate.objective_components) == {
            "component:quality",
            "component:prompt_alignment",
            "failure_rate",
        }
        raw = candidate.metadata["workflow_metrics"]
        assert raw["component_means"]["quality"] == (
            candidate.objective_components["component:quality"]
        )
        assert raw["component_means"]["prompt_alignment"] == (
            candidate.objective_components[
                "component:prompt_alignment"
            ]
        )


def test_best_workflow_is_reconstructable_from_result_metadata():
    registry = create_mock_registry()
    base = searchable_workflow()
    search_spec = SearchSpec(
        id="reconstruct",
        strategy="grid",
        dimensions=[
            ParameterDimension("parts", "top", [2, 5]),
            ParameterDimension("selected", "top", [1, 2]),
        ],
        objective=ObjectiveSpec(
            weights={"score_overall_mean": 1.0}
        ),
    )

    result = ParameterSearchRunner(registry).run(
        base,
        search_spec,
        experiment_spec(),
    )
    best = result.best_candidate()
    assert best is not None

    reconstructed = reconstruct_best_workflow(result)

    assert result.metadata["best_configuration"] == best.parameters
    assert reconstructed.metadata["parameter_search"]["configuration"] == (
        best.parameters
    )
    assert serialize_workflow(reconstructed, registry) == serialize_workflow(
        best.workflow,
        registry,
    )
    steps = {step.id: step for step in reconstructed.nodes}
    for path, value in best.parameters.items():
        step_id, parameter = path.split(".", 1)
        assert steps[step_id].parameters[parameter] == value


def test_search_result_round_trip_and_export(tmp_path):
    registry = create_mock_registry()
    search_spec = SearchSpec(
        id="export-search",
        strategy="random",
        sample_count=3,
        random_seed=99,
        dimensions=[
            ParameterDimension("parts", "top", [2, 3, 4]),
            ParameterDimension("compose", "count", [1, 2]),
        ],
        objective=ObjectiveSpec(
            weights={
                "score_overall_mean": 1.0,
                "runtime_seconds_mean": -0.0,
            }
        ),
        metadata={"purpose": "fixture"},
    )

    result = ParameterSearchRunner(registry).run(
        searchable_workflow(),
        search_spec,
        experiment_spec(),
    )

    restored = ParameterSearchResult.from_json(result.to_json())
    assert restored == result

    root = export_parameter_search(
        result,
        tmp_path / "search-output",
        registry,
    )
    loaded = load_parameter_search(root)
    assert loaded == result

    assert (root / "search.json").is_file()
    assert (root / "best_workflow.dsl").is_file()
    assert (root / "best_configuration.json").is_file()
    assert (root / "experiment" / "experiment.json").is_file()
    assert (root / "experiment" / "workflow_metrics.csv").is_file()
    assert len(list((root / "candidates").glob("*.json"))) == 3
    assert len(list((root / "candidates").glob("*.dsl"))) == 3

    best_payload = json.loads(
        (root / "best_configuration.json").read_text(encoding="utf-8")
    )
    assert best_payload["candidate_key"] == result.best_candidate_key
    assert best_payload["parameters"] == (
        result.best_candidate().parameters
    )
