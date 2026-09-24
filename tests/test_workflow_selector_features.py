import json

from image_drawer.training import (
    WorkflowSelectorFeatureSchema,
    WorkflowSelectorTrainConfig,
    encode_workflow_selector_candidate,
    load_workflow_selector_train_config,
    workflow_static_record,
)


def candidate(key="portrait", objective=1.0, status="success"):
    return {
        "workflow_key": key,
        "workflow_id": f"workflow-{key}",
        "workflow_version": "v1",
        "status": status,
        "objective_value": objective,
        "objective_components": {
            "component:quality": objective,
            "failure_rate": 0.0,
        },
        "workflow_metadata": {
            "content_hash": f"sha256:{key}",
            "backends": {
                "parts": "mock",
                "compose": "mock",
            },
            "metadata": {
                "template": key,
                "task": key,
            },
        },
    }


def test_feature_encoding_is_deterministic_and_fixed_size():
    schema = WorkflowSelectorFeatureSchema(text_dimensions=16)
    context = {
        "prompt_metadata": {"style": "portrait"},
        "resource_budget": "low",
    }

    first = encode_workflow_selector_candidate(
        "portrait request",
        candidate(),
        schema,
        context=context,
    )
    second = encode_workflow_selector_candidate(
        "portrait request",
        candidate(),
        schema,
        context=context,
    )

    assert first == second
    assert len(first) == schema.total_dimensions
    assert schema.total_dimensions == 48
    assert schema.identity_hash.startswith("sha256:")


def test_objective_and_run_status_do_not_leak_into_features():
    schema = WorkflowSelectorFeatureSchema(text_dimensions=16)
    first = candidate(objective=0.1, status="failure")
    second = candidate(objective=999.0, status="success")
    second["objective_components"] = {
        "component:quality": 999.0,
        "failure_rate": -1.0,
    }

    assert encode_workflow_selector_candidate(
        "portrait",
        first,
        schema,
    ) == encode_workflow_selector_candidate(
        "portrait",
        second,
        schema,
    )


def test_static_record_strips_label_and_run_fields():
    static = workflow_static_record(
        candidate(objective=0.7, status="failure")
    )

    assert set(static) == {
        "workflow_key",
        "workflow_id",
        "workflow_version",
        "workflow_metadata",
    }
    assert "objective_value" not in static
    assert "status" not in static


def test_prompt_workflow_interaction_changes_features():
    schema = WorkflowSelectorFeatureSchema(text_dimensions=16)

    portrait = encode_workflow_selector_candidate(
        "portrait request",
        candidate("portrait"),
        schema,
    )
    landscape = encode_workflow_selector_candidate(
        "portrait request",
        candidate("landscape"),
        schema,
    )

    assert portrait != landscape


def test_training_config_loads_relative_paths_from_toml(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "workflow.toml"
    config_path.write_text(
        """
[training]
id = "fixture"
dataset_path = "dataset.json"
output_dir = "run"
seed = 7
epochs = 3
batch_size = 4
learning_rate = 0.02
hidden_dimensions = [16]
text_dimensions = 8
top_k = 2
tie_tolerance = 0.05
selection_metric = "mean_objective_regret"
final_quality_metric = "component:quality"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_workflow_selector_train_config(config_path)

    assert isinstance(config, WorkflowSelectorTrainConfig)
    assert config.dataset_path == str(dataset.resolve())
    assert config.output_dir == str((tmp_path / "run").resolve())
    assert config.hidden_dimensions == [16]
    assert config.tie_tolerance == 0.05
    assert config.selection_metric == "mean_objective_regret"
