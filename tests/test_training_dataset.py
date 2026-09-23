import json

from image_drawer.core import (
    Artifact,
    Part,
    Score,
    SourceImage,
    StepExecution,
    StepSpec,
    Trajectory,
    WorkflowSpec,
)
from image_drawer.search import (
    ExperimentRunner,
    ExperimentSpec,
    ObjectiveSpec,
    PromptCase,
)
from image_drawer.steps import create_mock_registry
from image_drawer.training import (
    DatasetBuildConfig,
    SplitConfig,
    TrainingDataset,
    build_part_selector_dataset,
    build_workflow_selector_dataset,
    export_training_dataset,
    load_training_dataset,
    normalize_prompt_group,
    prompt_group_id,
)


class FakePartRepository:
    def __init__(self):
        self.parts = {
            "part-a": Part(
                id="part-a",
                source_image_id="source-1",
                category="face",
                crop_uri="parts/a.png",
                bbox=(0, 0, 16, 16),
                embedding_refs={"mock:v1": "embeddings/a.json"},
                tags=["cat", "red"],
                attributes={"pose": "front"},
                quality={"usable": True},
                extraction_method="fixture",
                extraction_version="v1",
                metadata={"split": "train"},
            ),
            "part-b": Part(
                id="part-b",
                source_image_id="source-1",
                category="face",
                crop_uri="parts/b.png",
                bbox=(16, 0, 16, 16),
                embedding_refs={"mock:v1": "embeddings/b.json"},
                tags=["dog", "blue"],
                extraction_method="fixture",
                extraction_version="v1",
                metadata={"split": "train"},
            ),
            "part-c": Part(
                id="part-c",
                source_image_id="source-1",
                category="face",
                crop_uri="parts/c.png",
                bbox=(32, 0, 16, 16),
                tags=["bird"],
                extraction_method="fixture",
                extraction_version="v1",
                metadata={"split": "train"},
            ),
        }

    def get(self, part_id):
        if part_id not in self.parts:
            raise KeyError(part_id)
        return self.parts[part_id]


def make_part_trajectory(
    trajectory_id: str,
    prompt: str,
    selected=("part-a",),
):
    retrieval = Artifact(
        id=f"{trajectory_id}:retrieval",
        artifact_type="PartSet",
        producing_step_id="retrieve",
        metadata={
            "value": ["part-a", "part-b", "part-c"],
            "retrieval_scores": [0.9, 0.7, 0.2],
            "query_id": f"query:{trajectory_id}",
            "category": "face",
            "part_set": {
                "id": f"partset:{trajectory_id}",
                "query_id": f"query:{trajectory_id}",
                "part_ids": ["part-a", "part-b", "part-c"],
                "retrieval_scores": [0.9, 0.7, 0.2],
                "category": "face",
                "metadata": {
                    "filters": {"split": "train"},
                    "embedding": {
                        "family": "metadata-text",
                        "model": "feature-hash",
                        "version": "v1",
                        "dimensions": 64,
                    },
                    "index": {
                        "backend": "bruteforce-cosine",
                        "version": "v1",
                    },
                },
            },
        },
    )
    selected_artifact = Artifact(
        id=f"{trajectory_id}:selected",
        artifact_type="PartSet",
        producing_step_id="select_parts",
        parent_artifact_ids=[retrieval.id],
        metadata={
            "value": list(selected),
            "retrieval_scores": [
                0.9 if item == "part-a" else 0.7
                for item in selected
            ],
            "selected": True,
        },
    )
    evaluation = Score(
        id=f"{trajectory_id}:score",
        evaluator="mock",
        evaluator_version="v1",
        overall=0.8,
        components={"quality": 0.8, "determinism": 1.0},
    )
    return Trajectory(
        id=trajectory_id,
        workflow_id="workflow-parts",
        workflow_version="v1",
        prompt=prompt,
        input_metadata={"seed": 7},
        executions=[
            StepExecution(
                id=f"{trajectory_id}:exec-retrieve",
                step_id="retrieve",
                step_type="RETRIEVE_PARTS",
                backend="part-bank",
                output_artifact_ids=[retrieval.id],
                seed=7,
                metadata={
                    "execution_identity": {
                        "embedding": {
                            "family": "metadata-text",
                            "model": "feature-hash",
                            "version": "v1",
                            "dimensions": 64,
                        },
                        "index": {
                            "backend": "bruteforce-cosine",
                            "version": "v1",
                        },
                    }
                },
            ),
            StepExecution(
                id=f"{trajectory_id}:exec-select",
                step_id="select_parts",
                step_type="SELECT_PARTS",
                backend="mock",
                input_artifact_ids=[retrieval.id],
                output_artifact_ids=[selected_artifact.id],
                seed=7,
            ),
        ],
        artifacts=[retrieval, selected_artifact],
        evaluations=[evaluation],
        selections=[
            {
                "step_id": "select_parts",
                "input_artifact_ids": [retrieval.id],
                "output_artifact_id": selected_artifact.id,
            }
        ],
        metadata={
            "execution_identity": {
                "retrieve": {
                    "embedding": {
                        "family": "metadata-text",
                        "model": "feature-hash",
                        "version": "v1",
                        "dimensions": 64,
                    },
                    "index": {
                        "backend": "bruteforce-cosine",
                        "version": "v1",
                    },
                }
            }
        },
    )


def split_config(seed=13):
    return SplitConfig(
        train=0.6,
        validation=0.2,
        test=0.2,
        seed=seed,
    )


def test_part_selector_builds_pointwise_and_pairwise_examples():
    trajectory = make_part_trajectory("trajectory-1", "red cat")
    config = DatasetBuildConfig(
        id="parts-both",
        split=split_config(),
        part_example_mode="both",
    )

    dataset = build_part_selector_dataset(
        [trajectory],
        config,
        repository=FakePartRepository(),
        source_metadata={"dataset_version": "fixture-v2"},
        generated_at="2026-09-24T00:00:00+00:00",
    )

    assert dataset.task == "part_selector"
    assert dataset.source_ids == ["trajectory-1"]
    pointwise = [
        item for item in dataset.examples
        if item["example_kind"] == "pointwise"
    ]
    pairwise = [
        item for item in dataset.examples
        if item["example_kind"] == "pairwise"
    ]
    assert len(pointwise) == 3
    assert len(pairwise) == 2

    labels = {
        item["candidates"][0]["part_id"]: item["label"]["selected"]
        for item in pointwise
    }
    assert labels == {
        "part-a": True,
        "part-b": False,
        "part-c": False,
    }
    assert {
        (
            item["label"]["preferred_part_id"],
            item["label"]["rejected_part_id"],
        )
        for item in pairwise
    } == {
        ("part-a", "part-b"),
        ("part-a", "part-c"),
    }

    selected = next(
        item for item in pointwise
        if item["candidates"][0]["part_id"] == "part-a"
    )
    features = selected["candidates"][0]
    assert features["retrieval_score"] == 0.9
    assert features["part_metadata"]["category"] == "face"
    assert features["part_metadata"]["tags"] == ["cat", "red"]
    assert features["embedding_refs"] == {
        "mock:v1": "embeddings/a.json"
    }
    assert features["embedding_identity"]["model"] == "feature-hash"

    assert dataset.metadata["source_metadata"] == {
        "dataset_version": "fixture-v2"
    }
    assert dataset.metadata["observed_versions"]["index"] == [
        {"backend": "bruteforce-cosine", "version": "v1"}
    ]
    assert dataset.metadata["observed_versions"]["embedding"][0][
        "dimensions"
    ] == 64
    assert dataset.metadata["evaluator_versions"] == [
        {"name": "mock", "version": "v1"}
    ]


def test_part_split_happens_before_example_expansion_and_prompt_variants_group():
    first = make_part_trajectory("trajectory-a", "Red CAT!")
    second = make_part_trajectory("trajectory-b", "  red   cat ")
    config = DatasetBuildConfig(
        id="parts-pointwise",
        split=split_config(seed=88),
        part_example_mode="pointwise",
    )

    dataset = build_part_selector_dataset(
        [first, second],
        config,
        generated_at="fixed",
    )

    assert normalize_prompt_group(first.prompt) == "red cat"
    assert prompt_group_id(first.prompt) == prompt_group_id(second.prompt)
    assert len({item["group_id"] for item in dataset.examples}) == 1
    assert len({item["split"] for item in dataset.examples}) == 1
    assert {item["source_trajectory_ids"][0] for item in dataset.examples} == {
        "trajectory-a",
        "trajectory-b",
    }


def test_part_pair_count_is_capped_deterministically():
    trajectory = make_part_trajectory(
        "trajectory-cap",
        "cat",
        selected=("part-a", "part-b"),
    )
    config = DatasetBuildConfig(
        id="pair-cap",
        split=split_config(),
        part_example_mode="pairwise",
        max_pairs_per_group=1,
    )

    first = build_part_selector_dataset(
        [trajectory],
        config,
        generated_at="fixed",
    )
    second = build_part_selector_dataset(
        [trajectory],
        config,
        generated_at="fixed",
    )

    assert first == second
    assert len(first.examples) == 1
    assert first.examples[0]["label"] == {
        "preferred_part_id": "part-a",
        "rejected_part_id": "part-c",
    }


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
        metadata={"template": "a"},
    )


def workflow_b():
    workflow = WorkflowSpec.from_dict(workflow_a().to_dict())
    workflow.id = "workflow-b"
    workflow.version = "v2"
    workflow.metadata = {"template": "b"}
    parts = next(node for node in workflow.nodes if node.id == "parts")
    parts.parameters["top"] = 3
    return workflow


def make_experiment():
    registry = create_mock_registry()
    return ExperimentRunner(
        registry,
        environment_metadata={
            "dataset_version": "fixture-v3",
            "git_commit": "abc123",
        },
    ).run(
        ExperimentSpec(
            id="workflow-comparison",
            prompt_set=[
                PromptCase(
                    id="cat",
                    prompt="Cat Portrait",
                    metadata={"style": "portrait"},
                ),
                PromptCase(
                    id="dog",
                    prompt="Dog Portrait",
                    metadata={"style": "portrait"},
                ),
            ],
            repeats=2,
            seed_policy="paired_increment",
            base_seed=20,
        ),
        {"a": workflow_a(), "b": workflow_b()},
    )


def test_workflow_selector_examples_keep_comparison_candidates_together():
    experiment = make_experiment()
    config = DatasetBuildConfig(
        id="workflow-dataset",
        split=split_config(seed=4),
    )
    objective = ObjectiveSpec(
        weights={
            "component:determinism": 1.0,
            "failure_rate": -2.0,
        }
    )

    dataset = build_workflow_selector_dataset(
        experiment,
        config,
        objective,
        generated_at="2026-09-24T00:00:00+00:00",
    )

    assert dataset.task == "workflow_selector"
    assert len(dataset.examples) == 4
    for example in dataset.examples:
        assert len(example["candidates"]) == 2
        assert {
            item["workflow_key"]
            for item in example["candidates"]
        } == {"a", "b"}
        # determinismは同点なのでstable keyのaが選ばれる。
        assert example["label"]["selected_workflow_key"] == "a"
        assert len(example["source_run_ids"]) == 2
        assert len(example["source_trajectory_ids"]) == 2
        assert all(
            candidate["objective_components"][
                "component:determinism"
            ] == 1.0
            for candidate in example["candidates"]
        )

    cat_examples = [
        item for item in dataset.examples
        if item["context"]["prompt_case_id"] == "cat"
    ]
    assert len({item["split"] for item in cat_examples}) == 1
    assert len({item["group_id"] for item in cat_examples}) == 1
    assert dataset.metadata["objective"] == objective.to_dict()
    assert dataset.metadata["experiment_environment"][
        "dataset_version"
    ] == "fixture-v3"
    assert dataset.metadata["source_experiment_id"] == "workflow-comparison"


def test_workflow_selector_dataset_is_deterministic_for_fixed_inputs():
    experiment = make_experiment()
    config = DatasetBuildConfig(
        id="workflow-deterministic",
        split=split_config(seed=777),
    )
    objective = ObjectiveSpec(
        weights={"component:determinism": 1.0}
    )

    first = build_workflow_selector_dataset(
        experiment,
        config,
        objective,
        generated_at="fixed",
    )
    second = build_workflow_selector_dataset(
        experiment,
        config,
        objective,
        generated_at="fixed",
    )

    assert first == second


def test_training_dataset_json_round_trip_and_jsonl_export(tmp_path):
    dataset = build_part_selector_dataset(
        [make_part_trajectory("trajectory-export", "cat")],
        DatasetBuildConfig(
            id="export",
            split=split_config(seed=2),
            part_example_mode="both",
        ),
        repository=FakePartRepository(),
        generated_at="fixed",
    )

    restored = TrainingDataset.from_json(dataset.to_json())
    assert restored == dataset

    root = export_training_dataset(dataset, tmp_path / "dataset")
    loaded = load_training_dataset(root)
    assert loaded == dataset

    assert (root / "dataset.json").is_file()
    assert (root / "build_config.json").is_file()
    assert (root / "manifest.json").is_file()
    assert (root / "train.jsonl").is_file()
    assert (root / "validation.jsonl").is_file()
    assert (root / "test.jsonl").is_file()

    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_id"] == dataset.id
    assert manifest["split_counts"] == dataset.split_counts()

    exported = []
    for split in ("train", "validation", "test"):
        for line in (root / f"{split}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line:
                item = json.loads(line)
                assert item["split"] == split
                exported.append(item)
    assert len(exported) == len(dataset.examples)
    assert {item["id"] for item in exported} == {
        item["id"] for item in dataset.examples
    }


def test_split_ratios_must_sum_to_one():
    try:
        SplitConfig(train=0.8, validation=0.2, test=0.2)
    except ValueError as error:
        assert "sum to 1.0" in str(error)
    else:
        raise AssertionError("invalid split ratios should fail")
