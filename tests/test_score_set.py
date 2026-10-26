"""候補と評価器固有得点の保存境界を確認する。"""

import math

import pytest

from image_drawer.core import Score, ScoreSet


# {
#   責務: [
#     make_score_set: 二候補の得点集合を構築する
#   ]
#   処理: [
#     異なる成分と由来を持つ二件を同順で保持する
#   ]
#   引数: [
#     なし
#   ]
#   戻り値: [
#     score_set: 検証用の得点集合
#   ]
# }
def make_score_set() -> ScoreSet:
    return ScoreSet(
        id="scores-1",
        evaluator="mock",
        evaluator_version="v1",
        candidate_artifact_ids=["image-a", "image-b"],
        scores=[
            Score(
                id="score-a",
                evaluator="mock",
                evaluator_version="v1",
                overall=0.4,
                components={"quality": 0.3, "detail": 0.5},
                metadata={"source": "a"},
            ),
            Score(
                id="score-b",
                evaluator="mock",
                evaluator_version="v1",
                overall=None,
                components={"quality": 0.8, "detail": 0.2},
                metadata={"source": "b"},
            ),
        ],
        metadata={"dataset": "fixture"},
    )


# {
#   責務: [
#     test_score_set_round_trip: 複数候補と生得点の由来をJSON往復で保持する
#   ]
#   処理: [
#     候補の順序・成分・評価器・任意のoverallを比較する
#   ]
#   引数: [
#     なし
#   ]
#   戻り値: [
#     なし
#   ]
# }
def test_score_set_round_trip():
    original = make_score_set()
    restored = ScoreSet.from_json(original.to_json())
    assert restored.to_dict() == original.to_dict()
    assert restored.scores_by_candidate()["image-a"].components == {"quality": 0.3, "detail": 0.5}
    assert restored.score_for("image-b").overall is None
    assert restored.score_for("image-b").metadata == {"source": "b"}
    with pytest.raises(KeyError):
        restored.score_for("missing")


# {
#   責務: [
#     test_empty_score_set: 空のバッチを明示的な空集合として保持する
#   ]
#   処理: [
#     空集合を往復し、未登録候補の参照を拒否する
#   ]
#   引数: [
#     なし
#   ]
#   戻り値: [
#     なし
#   ]
# }
def test_empty_score_set():
    empty = ScoreSet(id="empty", evaluator="mock", evaluator_version="v1")
    assert ScoreSet.from_json(empty.to_json()).scores_by_candidate() == {}
    with pytest.raises(KeyError):
        empty.score_for("absent")


# {
#   責務: [
#     test_reject_inconsistent_payloads: 壊れた保存データを読み込まない
#   ]
#   処理: [
#     候補、得点、評価器、数値の不整合をそれぞれ拒否する
#   ]
#   引数: [
#     mutation: 保存データを破損させる操作
#   ]
#   戻り値: [
#     なし
#   ]
# }
@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("candidate_artifact_ids", "image-a"),
        lambda p: p.__setitem__("scores", {}),
        lambda p: p.__setitem__("metadata", None),
        lambda p: p["scores"][0].__setitem__("components", []),
        lambda p: p["scores"][0].__setitem__("metadata", None),
        lambda p: p["candidate_artifact_ids"].pop(),
        lambda p: p["candidate_artifact_ids"].__setitem__(1, "image-a"),
        lambda p: p["candidate_artifact_ids"].__setitem__(0, " "),
        lambda p: p["scores"][1].__setitem__("id", "score-a"),
        lambda p: p["scores"][0].__setitem__("evaluator", "other"),
        lambda p: p["scores"][0].__setitem__("evaluator_version", "v2"),
        lambda p: p["scores"][0].__setitem__("overall", math.nan),
        lambda p: p["scores"][0]["components"].__setitem__("quality", math.inf),
        lambda p: p["scores"][0]["components"].__setitem__("quality", True),
        lambda p: p["scores"][0]["components"].__setitem__("quality", 10**400),
        lambda p: p["scores"][0]["components"].__setitem__("", 0.5),
        lambda p: p.__setitem__("evaluator_version", ""),
    ],
)
def test_reject_inconsistent_payloads(mutation):
    payload = make_score_set().to_dict()
    mutation(payload)
    with pytest.raises((ValueError, TypeError)):
        ScoreSet.from_dict(payload)


# {
#   責務: [
#     test_revalidate_after_mutation: 作成後の変更による不整合を読み出しと保存で拒否する
#   ]
#   処理: [
#     対応表と子得点を破損させ、それぞれの境界を確認する
#   ]
#   引数: [
#     なし
#   ]
#   戻り値: [
#     なし
#   ]
# }
def test_revalidate_after_mutation():
    scores = make_score_set()
    scores.candidate_artifact_ids.pop()
    with pytest.raises(ValueError):
        scores.scores_by_candidate()
    scores = make_score_set()
    scores.scores[0].components["quality"] = math.nan
    with pytest.raises(ValueError):
        scores.to_json()
