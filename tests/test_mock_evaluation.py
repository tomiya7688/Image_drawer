"""モック評価の候補対応と保存形式を確認する。"""

import pytest

from image_drawer.core import Artifact, ScoreSet
from image_drawer.steps import StepContext
from image_drawer.steps.mock import EvaluateStep


def test_batch_evaluation_preserves_candidate_order_and_scores():
    candidates = [
        Artifact(id="b", artifact_type="Image", metadata={"value": "B"}),
        Artifact(id="a", artifact_type="Image", metadata={"value": "A"}),
    ]
    source = Artifact(
        id="images",
        artifact_type="ImageSet",
        metadata={"candidates": [candidate.to_dict() for candidate in candidates]},
    )
    context = StepContext(run_id="run", step_id="evaluate", backend="mock", external_inputs={})
    output = EvaluateStep().run([source], {"evaluator": "mock"}, context)
    restored = ScoreSet.from_dict(output.metadata["score_set"])
    assert restored.candidate_artifact_ids == ["b", "a"]
    assert [score.overall for score in restored.scores] == [0.066, 0.065]
    assert [score.metadata["candidate_artifact_id"] for score in output.scores] == ["b", "a"]
    assert output.parent_artifact_ids == ["images"]
    assert restored.scores == output.scores


@pytest.mark.parametrize("payload", [[], "invalid", [{"id": "x", "artifact_type": "Text"}]])
def test_invalid_image_sets_are_rejected(payload):
    source = Artifact(id="images", artifact_type="ImageSet", metadata={"candidates": payload})
    context = StepContext(run_id="run", step_id="evaluate", backend="mock", external_inputs={})
    with pytest.raises((TypeError, ValueError)):
        EvaluateStep().run([source], {"evaluator": "mock"}, context)


def test_unknown_evaluator_is_rejected():
    source = Artifact(id="image", artifact_type="Image")
    context = StepContext(run_id="run", step_id="evaluate", backend="mock", external_inputs={})
    with pytest.raises(ValueError, match="unknown mock evaluator"):
        EvaluateStep().run([source], {"evaluator": "other"}, context)
