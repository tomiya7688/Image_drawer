"""評価済み候補の選択と不整合の拒否を確認する。"""

import pytest

from image_drawer.core import Artifact, Score, ScoreSet
from image_drawer.steps import StepContext
from image_drawer.steps.mock import SelectStep


def make_inputs():
    candidates = [
        Artifact(id="b", artifact_type="Image", uri="b.png", metadata={"value": "B"}),
        Artifact(
            id="a",
            artifact_type="Image",
            uri="a.png",
            metadata={"value": "A", "execution_identity": {"generator": "old"}},
        ),
    ]
    images = Artifact(
        id="set",
        artifact_type="ImageSet",
        metadata={"candidates": [candidate.to_dict() for candidate in candidates]},
    )
    score_set = ScoreSet(
        id="scores",
        evaluator="mock",
        evaluator_version="v1",
        candidate_artifact_ids=["b", "a"],
        scores=[
            Score(id="score-b", evaluator="mock", evaluator_version="v1", overall=0.4),
            Score(id="score-a", evaluator="mock", evaluator_version="v1", overall=0.8),
        ],
    )
    scores = Artifact(
        id="scores",
        artifact_type="ScoreSet",
        parent_artifact_ids=["set"],
        scores=score_set.scores,
        metadata={"score_set": score_set.to_dict()},
    )
    return images, scores


def test_select_returns_highest_scored_image():
    images, scores = make_inputs()
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    result = SelectStep().run([images, scores], {"top": 1}, context)
    assert result.uri == "a.png"
    assert result.metadata["value"] == "A"
    assert result.metadata["selection"]["candidate_artifact_id"] == "a"
    assert result.metadata["selection"]["score_id"] == "score-a"
    assert "execution_identity" not in result.metadata


def test_ties_are_resolved_by_candidate_id():
    images, scores = make_inputs()
    scores.metadata["score_set"]["scores"][0]["overall"] = 0.8
    scores.scores[0].overall = 0.8
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    result = SelectStep().run([images, scores], {"top": 1}, context)
    assert result.metadata["selection"]["candidate_artifact_id"] == "a"


def test_select_rejects_disagreement_with_artifact_scores():
    images, scores = make_inputs()
    scores.scores = [Score(id="other", evaluator="mock", evaluator_version="v1", overall=0.1)]
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    with pytest.raises(ValueError, match="artifact scores do not match"):
        SelectStep().run([images, scores], {"top": 1}, context)


def test_select_rejects_missing_artifact_scores():
    images, scores = make_inputs()
    scores.scores = []
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    with pytest.raises(ValueError, match="artifact scores do not match"):
        SelectStep().run([images, scores], {"top": 1}, context)


def test_select_rejects_changed_score_value():
    images, scores = make_inputs()
    scores.scores[0].overall = 0.9
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    with pytest.raises(ValueError, match="artifact scores do not match"):
        SelectStep().run([images, scores], {"top": 1}, context)


@pytest.mark.parametrize(
    "case", ["foreign", "missing", "misaligned", "top", "no_overall", "bool_top"]
)
def test_select_rejects_inconsistent_inputs(case):
    images, scores = make_inputs()
    top = 1
    if case == "foreign":
        scores.parent_artifact_ids = ["another"]
    elif case == "missing":
        scores.metadata["score_set"]["candidate_artifact_ids"][0] = "other"
    elif case == "misaligned":
        scores.metadata["score_set"]["scores"][0]["metadata"] = {"candidate_artifact_id": "a"}
        scores.scores[0].metadata = {"candidate_artifact_id": "a"}
    elif case == "no_overall":
        scores.metadata["score_set"]["scores"][0]["overall"] = None
        scores.scores[0].overall = None
    elif case == "bool_top":
        top = True
    else:
        top = 2
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    with pytest.raises(ValueError):
        SelectStep().run([images, scores], {"top": top}, context)
