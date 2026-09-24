import pytest

torch = pytest.importorskip("torch")

from image_drawer.training import (
    DatasetBuildConfig,
    SplitConfig,
    TrainingDataset,
    WorkflowSelectorTrainConfig,
    evaluate_workflow_selector_checkpoint,
    load_workflow_selector_checkpoint,
    score_workflow_selector_checkpoint,
    train_workflow_selector,
)


WORKFLOWS = (
    ("portrait", "portrait"),
    ("landscape", "landscape"),
    ("generic", "generic"),
)


def workflow_candidate(key, task, objective):
    return {
        "workflow_key": key,
        "workflow_id": f"workflow-{key}",
        "workflow_version": "v1",
        "status": "success",
        "objective_value": objective,
        "objective_components": {
            "component:quality": objective,
            "failure_rate": 0.0,
        },
        "workflow_metadata": {
            "content_hash": f"sha256:{key}",
            "backends": {
                "retrieve": "mock",
                "compose": "mock",
                "evaluate": "evaluation",
            },
            "metadata": {
                "template": key,
                "task": task,
            },
        },
    }


def add_example(examples, split, index, target, *, tie=False):
    prompt = f"{target} request sample {index}"
    objectives = {
        "portrait": 0.15,
        "landscape": 0.15,
        "generic": 0.05,
    }
    objectives[target] = 1.0
    if tie:
        alternate = "generic"
        objectives[alternate] = 0.99

    candidates = [
        workflow_candidate(key, task, objectives[key])
        for key, task in WORKFLOWS
    ]
    examples.append(
        {
            "id": f"workflow-example-{split}-{index}",
            "split": split,
            "group_id": f"group-{split}-{index}",
            "prompt": prompt,
            "context": {
                "experiment_id": "fixture-experiment",
                "prompt_case_id": f"prompt-{split}-{index}",
                "repeat_index": 0,
                "prompt_metadata": {
                    "task": target,
                    "style": target,
                },
            },
            "candidates": candidates,
            "label": {
                "selected_workflow_key": target,
                "objective": {
                    "weights": {
                        "component:quality": 1.0,
                    },
                    "metadata": {},
                },
            },
            "source_run_ids": [
                f"run-{split}-{index}-{key}"
                for key, _ in WORKFLOWS
            ],
            "source_trajectory_ids": [
                f"trajectory-{split}-{index}-{key}"
                for key, _ in WORKFLOWS
            ],
            "metadata": {
                "candidate_count": len(candidates),
            },
        }
    )


def make_dataset():
    examples = []
    for index in range(6):
        add_example(
            examples,
            "train",
            index,
            "portrait" if index % 2 == 0 else "landscape",
        )
    add_example(examples, "validation", 10, "portrait")
    add_example(examples, "validation", 11, "landscape", tie=True)
    add_example(examples, "test", 20, "portrait")
    add_example(examples, "test", 21, "landscape")
    return TrainingDataset(
        id="workflow-ranker-fixture-v1",
        task="workflow_selector",
        build_config=DatasetBuildConfig(
            id="workflow-fixture-build",
            split=SplitConfig(
                train=0.6,
                validation=0.2,
                test=0.2,
                seed=1,
            ),
        ),
        examples=examples,
        source_ids=[
            f"run-{split}-{index}-{key}"
            for split, indexes in (
                ("train", range(6)),
                ("validation", (10, 11)),
                ("test", (20, 21)),
            )
            for index in indexes
            for key, _ in WORKFLOWS
        ],
        metadata={
            "generated_at": "fixed",
            "source_experiment_id": "fixture-experiment",
            "objective": {
                "weights": {
                    "component:quality": 1.0,
                },
                "metadata": {
                    "name": "quality-only",
                },
            },
        },
    )


def config(tmp_path, name="run"):
    return WorkflowSelectorTrainConfig(
        id="workflow-selector-fixture",
        dataset_path="unused",
        output_dir=str(tmp_path / name),
        seed=123,
        epochs=12,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        hidden_dimensions=[16],
        text_dimensions=8,
        top_k=2,
        tie_tolerance=0.02,
        selection_metric="top1_accuracy",
        final_quality_metric="component:quality",
    )


def test_training_saves_checkpoint_objective_and_separate_test_metrics(tmp_path):
    dataset = make_dataset()
    result = train_workflow_selector(
        dataset,
        config(tmp_path),
    )

    checkpoint = tmp_path / "run" / "best.pt"
    assert checkpoint.is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert (tmp_path / "run" / "training_result.json").is_file()
    assert result.dataset_id == dataset.id
    assert result.best_epoch >= 1
    assert result.best_validation_metrics.example_count == 2
    assert result.best_validation_metrics.tied_example_count == 1
    assert result.best_validation_metrics.top1_accuracy is not None
    assert result.best_validation_metrics.top_k_accuracy is not None
    assert result.best_validation_metrics.mean_objective_regret is not None
    assert result.test_metrics is not None
    assert result.test_metrics.split == "test"
    assert result.test_metrics.example_count == 2
    assert result.test_metrics.downstream_final_quality_mean is not None

    model, schema, metadata = load_workflow_selector_checkpoint(checkpoint)
    assert schema.version == "workflow-selector-feature-v1"
    assert metadata["feature_schema_hash"] == schema.identity_hash
    assert metadata["dataset_id"] == dataset.id
    assert metadata["model_type"] == "workflow-selector-mlp"
    assert metadata["best_epoch"] == result.best_epoch
    assert metadata["label_objective"] == dataset.metadata["objective"]
    assert {
        item["workflow_key"]
        for item in metadata["known_workflows"]
    } == {"portrait", "landscape", "generic"}

    held_out = evaluate_workflow_selector_checkpoint(
        checkpoint,
        dataset,
        split="test",
        top_k=2,
    )
    assert held_out == result.test_metrics


def test_fixed_seed_training_is_reproducible(tmp_path):
    dataset = make_dataset()

    first = train_workflow_selector(
        dataset,
        config(tmp_path, "first"),
    )
    second = train_workflow_selector(
        dataset,
        config(tmp_path, "second"),
    )

    assert first.best_epoch == second.best_epoch
    assert (
        first.best_validation_metrics
        == second.best_validation_metrics
    )
    assert first.test_metrics == second.test_metrics


def test_checkpoint_scores_known_workflows_for_new_prompt_without_labels(tmp_path):
    dataset = make_dataset()
    result = train_workflow_selector(
        dataset,
        config(tmp_path, "score"),
    )

    ranking = score_workflow_selector_checkpoint(
        result.checkpoint_path,
        "portrait request for a new user",
        context={
            "prompt_metadata": {
                "task": "portrait",
                "style": "portrait",
            }
        },
    )

    assert len(ranking) == 3
    assert {item["workflow_key"] for item in ranking} == {
        "portrait",
        "landscape",
        "generic",
    }
    assert all("score" in item for item in ranking)
    assert all("objective_value" not in item for item in ranking)
    assert all("status" not in item for item in ranking)
    assert ranking == sorted(
        ranking,
        key=lambda item: (-item["score"], item["workflow_key"]),
    )
