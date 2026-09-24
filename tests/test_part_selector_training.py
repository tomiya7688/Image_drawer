import pytest

torch = pytest.importorskip("torch")

from image_drawer.training import (
    DatasetBuildConfig,
    PartCandidateFeatures,
    PartSelectorFeatureSchema,
    PartSelectorTrainConfig,
    SplitConfig,
    TrainingDataset,
    evaluate_part_selector_checkpoint,
    load_part_selector_checkpoint,
    train_part_selector,
)


def make_candidate(part_id, tags, retrieval_score, quality=None):
    return PartCandidateFeatures(
        part_id=part_id,
        retrieval_score=retrieval_score,
        part_metadata={
            "category": "face",
            "bbox": [0, 0, 16, 16],
            "tags": list(tags),
            "attributes": {},
            "quality": quality or {},
            "extraction_method": "fixture",
            "extraction_version": "v1",
            "metadata": {"search_text": " ".join(tags)},
        },
        embedding_identity={
            "family": "metadata-text",
            "model": "feature-hash",
            "version": "v1",
            "dimensions": 32,
        },
    ).to_dict()


def add_group(examples, split, index):
    prompt = f"target token {index}"
    group_id = f"group-{split}-{index}"
    retrieval_id = f"retrieval-{split}-{index}"
    positive = make_candidate(
        f"{group_id}-positive",
        ["target", "token", str(index)],
        1.0,
        {"downstream_evaluator_score": 0.9},
    )
    negative_a = make_candidate(
        f"{group_id}-negative-a",
        ["other", "blue"],
        -0.5,
        {"downstream_evaluator_score": 0.2},
    )
    negative_b = make_candidate(
        f"{group_id}-negative-b",
        ["unrelated", "green"],
        -1.0,
        {"downstream_evaluator_score": 0.1},
    )
    candidates = [positive, negative_a, negative_b]
    for candidate in candidates:
        selected = candidate["part_id"] == positive["part_id"]
        examples.append(
            {
                "id": f"point-{candidate['part_id']}",
                "split": split,
                "group_id": group_id,
                "example_kind": "pointwise",
                "prompt": prompt,
                "context": {
                    "category": "face",
                    "retrieval_artifact_id": retrieval_id,
                    "retrieval_filters": {},
                },
                "candidates": [candidate],
                "label": {
                    "selected": selected,
                    "value": 1 if selected else 0,
                },
                "source_trajectory_ids": [f"trajectory-{group_id}"],
                "metadata": {},
            }
        )
    for negative in (negative_a, negative_b):
        examples.append(
            {
                "id": f"pair-{positive['part_id']}-{negative['part_id']}",
                "split": split,
                "group_id": group_id,
                "example_kind": "pairwise",
                "prompt": prompt,
                "context": {
                    "category": "face",
                    "retrieval_artifact_id": retrieval_id,
                    "retrieval_filters": {},
                },
                "candidates": [positive, negative],
                "label": {
                    "preferred_part_id": positive["part_id"],
                    "rejected_part_id": negative["part_id"],
                },
                "source_trajectory_ids": [f"trajectory-{group_id}"],
                "metadata": {},
            }
        )


def make_dataset():
    examples = []
    for index in range(4):
        add_group(examples, "train", index)
    for index in range(2):
        add_group(examples, "validation", index + 10)
    for index in range(2):
        add_group(examples, "test", index + 20)
    return TrainingDataset(
        id="ranker-fixture-v1",
        task="part_selector",
        build_config=DatasetBuildConfig(
            id="fixture-build",
            split=SplitConfig(
                train=0.5,
                validation=0.25,
                test=0.25,
                seed=1,
            ),
            part_example_mode="both",
        ),
        examples=examples,
        source_ids=[
            f"trajectory-group-{split}-{index}"
            for split, count in (
                ("train", 4),
                ("validation", 2),
                ("test", 2),
            )
            for index in range(count)
        ],
        metadata={"generated_at": "fixed"},
    )


def config(tmp_path, name="run", objective="pairwise"):
    return PartSelectorTrainConfig(
        id=f"fixture-{objective}",
        dataset_path="unused",
        output_dir=str(tmp_path / name),
        seed=123,
        epochs=8,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        hidden_dimensions=[16],
        objective=objective,
        text_dimensions=8,
        top_k=2,
        selection_metric="mrr",
    )


def test_pairwise_training_saves_best_checkpoint_and_test_is_separate(tmp_path):
    dataset = make_dataset()
    result = train_part_selector(
        dataset,
        config(tmp_path),
    )

    checkpoint = tmp_path / "run" / "best.pt"
    assert checkpoint.is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert (tmp_path / "run" / "training_result.json").is_file()
    assert result.dataset_id == dataset.id
    assert result.best_epoch >= 1
    assert result.best_validation_metrics.group_count == 2
    assert result.best_validation_metrics.pairwise_accuracy is not None
    assert result.best_validation_metrics.mrr is not None
    assert result.test_metrics is not None
    assert result.test_metrics.split == "test"
    assert result.test_metrics.group_count == 2

    model, schema, metadata = load_part_selector_checkpoint(checkpoint)
    assert schema.version == "part-selector-feature-v1"
    assert metadata["feature_schema_hash"] == schema.identity_hash
    assert metadata["dataset_id"] == dataset.id
    assert metadata["model_type"] == "part-selector-mlp"
    assert metadata["best_epoch"] == result.best_epoch

    held_out = evaluate_part_selector_checkpoint(
        checkpoint,
        dataset,
        split="test",
        top_k=2,
    )
    assert held_out == result.test_metrics
    assert held_out.downstream_evaluator_score_mean is not None


def test_fixed_seed_training_is_reproducible_for_debugging(tmp_path):
    dataset = make_dataset()

    first = train_part_selector(
        dataset,
        config(tmp_path, "first"),
    )
    second = train_part_selector(
        dataset,
        config(tmp_path, "second"),
    )

    assert first.best_epoch == second.best_epoch
    assert (
        first.best_validation_metrics
        == second.best_validation_metrics
    )
    assert first.test_metrics == second.test_metrics


def test_pointwise_fallback_trains_and_evaluates(tmp_path):
    dataset = make_dataset()
    result = train_part_selector(
        dataset,
        config(tmp_path, "pointwise", objective="pointwise"),
    )

    assert (tmp_path / "pointwise" / "best.pt").is_file()
    assert result.best_validation_metrics.top1_accuracy is not None
    assert result.best_validation_metrics.top_k_recall is not None
    assert result.best_validation_metrics.ndcg is not None
