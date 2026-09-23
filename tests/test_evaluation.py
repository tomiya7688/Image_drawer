from image_drawer.core import Artifact, Score, ScoreSet
from image_drawer.evaluation import (
    DeterministicMockEvaluator,
    create_default_evaluator_registry,
    parse_component_weights,
    select_from_scores,
)
from image_drawer.steps import EvaluateStep, SelectStep, StepContext


def make_candidates():
    first = Artifact(
        id="image-a",
        artifact_type="Image",
        uri="runs/a.png",
        metadata={
            "value": "a",
            "mock_scores": {
                "quality": 0.9,
                "prompt_alignment": 0.1,
            },
        },
    )
    second = Artifact(
        id="image-b",
        artifact_type="Image",
        uri="runs/b.png",
        metadata={
            "value": "b",
            "mock_scores": {
                "quality": 0.4,
                "prompt_alignment": 0.8,
            },
        },
    )
    return first, second


def make_image_set():
    first, second = make_candidates()
    return Artifact(
        id="images",
        artifact_type="ImageSet",
        metadata={
            "candidates": [first.to_dict(), second.to_dict()],
            "value": [first.id, second.id],
        },
    )


def context(step_id: str):
    return StepContext(
        run_id="evaluation-test",
        step_id=step_id,
        backend="test",
        external_inputs={"prompt": "fixture prompt"},
    )


def test_scoreset_round_trip_preserves_candidate_score_mapping():
    first, second = make_candidates()
    evaluator = DeterministicMockEvaluator()

    score_set = evaluator.evaluate(
        [first, second],
        context={"prompt": "fixture"},
    )
    restored = ScoreSet.from_json(score_set.to_json())

    assert restored == score_set
    assert restored.candidate_artifact_ids == ["image-a", "image-b"]
    assert restored.score_for("image-a").metadata["artifact_id"] == "image-a"


def test_batch_evaluate_imageset_preserves_raw_components_and_identity():
    step = EvaluateStep(create_default_evaluator_registry())

    result = step.run(
        [make_image_set()],
        {"evaluator": "mock"},
        context("evaluate"),
    )

    assert result.artifact_type == "ScoreSet"
    score_set = ScoreSet.from_dict(result.metadata["score_set"])
    assert score_set.evaluator == "mock"
    assert score_set.evaluator_version == "v1"
    assert score_set.candidate_artifact_ids == ["image-a", "image-b"]
    assert score_set.scores[0].components == {
        "quality": 0.9,
        "prompt_alignment": 0.1,
    }
    assert score_set.scores[1].components == {
        "quality": 0.4,
        "prompt_alignment": 0.8,
    }
    assert result.metadata["execution_identity"] == {
        "evaluator": {"name": "mock", "version": "v1"}
    }


def test_best_selection_uses_evaluator_overall_and_returns_image():
    evaluate = EvaluateStep(create_default_evaluator_registry())
    select = SelectStep()
    image_set = make_image_set()
    scores = evaluate.run(
        [image_set],
        {"evaluator": "mock"},
        context("evaluate"),
    )

    selected = select.run(
        [image_set, scores],
        {"strategy": "best", "top": 1, "weights": []},
        context("select"),
    )

    assert selected.artifact_type == "Image"
    assert selected.metadata["selected_from_artifact_id"] == "image-b"
    decision = selected.metadata["selection"]
    assert decision["selected_artifact_ids"] == ["image-b"]
    assert decision["ranking"] == ["image-b", "image-a"]
    assert decision["evaluator"] == "mock"
    assert selected.scores[0].components == {
        "quality": 0.4,
        "prompt_alignment": 0.8,
    }


def test_top_k_selection_returns_serialized_imageset():
    evaluate = EvaluateStep(create_default_evaluator_registry())
    select = SelectStep()
    image_set = make_image_set()
    scores = evaluate.run(
        [image_set],
        {"evaluator": "mock"},
        context("evaluate"),
    )

    selected = select.run(
        [image_set, scores],
        {"strategy": "top_k", "top": 2, "weights": []},
        context("select"),
    )

    assert selected.artifact_type == "ImageSet"
    assert selected.metadata["value"] == ["image-b", "image-a"]
    assert [
        item["id"] for item in selected.metadata["candidates"]
    ] == ["image-b", "image-a"]
    assert len(selected.scores) == 2


def test_weighted_selection_uses_components_not_overall():
    evaluate = EvaluateStep(create_default_evaluator_registry())
    select = SelectStep()
    image_set = make_image_set()
    scores = evaluate.run(
        [image_set],
        {"evaluator": "mock"},
        context("evaluate"),
    )

    selected = select.run(
        [image_set, scores],
        {
            "strategy": "weighted",
            "top": 1,
            "weights": ["quality=1.0", "prompt_alignment=0.0"],
        },
        context("select"),
    )

    assert selected.artifact_type == "Image"
    assert selected.metadata["selected_from_artifact_id"] == "image-a"
    decision = selected.metadata["selection"]
    assert decision["selected_artifact_ids"] == ["image-a"]
    assert decision["weights"] == {
        "quality": 1.0,
        "prompt_alignment": 0.0,
    }
    assert decision["aggregate_scores"]["image-a"] == 0.9
    assert decision["aggregate_scores"]["image-b"] == 0.4


def test_weight_parser_and_missing_component_fail_clearly():
    assert parse_component_weights(["quality=0.7", "color=0.3"]) == {
        "quality": 0.7,
        "color": 0.3,
    }
    score_set = ScoreSet(
        id="scores",
        evaluator="mock",
        evaluator_version="v1",
        candidate_artifact_ids=["a"],
        scores=[
            Score(
                id="score-a",
                evaluator="mock",
                evaluator_version="v1",
                overall=0.5,
                components={"quality": 0.5},
            )
        ],
    )

    try:
        select_from_scores(
            score_set,
            strategy="weighted",
            top=1,
            weights={"missing": 1.0},
        )
    except ValueError as error:
        assert "lacks weighted components" in str(error)
    else:
        raise AssertionError("missing weighted component should fail")


def test_mock_evaluator_is_deterministic_without_explicit_fixture_scores():
    artifact = Artifact(
        id="generated",
        artifact_type="Image",
        uri="runs/generated.png",
        metadata={"value": "same"},
    )
    evaluator = DeterministicMockEvaluator()

    first = evaluator.evaluate([artifact], context={"prompt": "same"})
    second = evaluator.evaluate([artifact], context={"prompt": "same"})

    assert first == second
    assert first.scores[0].components["determinism"] == 1.0
