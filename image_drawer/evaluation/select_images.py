"""評価済み画像候補から上位k件を選ぶ。基準は総合得点・評価軸・重み付き平均から選べる。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from image_drawer.core import Artifact, Score, ScoreSet
from image_drawer.evaluation.mock import image_candidates


# {
#   責務: [
#     ImageSelection: 選ばれた画像と、順位づけの根拠を保持する
#   ]
#   フィールド: [
#     candidates: 選ばれた画像。順位順
#     scores: 選ばれた画像に対応する生得点。candidatesと同順
#     score_set: 評価器の情報を持つ得点集合
#     method: 順位づけの基準。weighted、component、overall のいずれか
#     weights: weighted のときの評価軸ごとの重み。それ以外は空
#     component: component のときの評価軸名。それ以外はNone
#     ranking: 全候補の順位。候補ID、得点ID、集約値、生の成分を持つ
#   ]
# }
@dataclass(slots=True)
class ImageSelection:
    candidates: list[Artifact]
    scores: list[Score]
    score_set: ScoreSet
    method: str
    weights: dict[str, float] = field(default_factory=dict)
    component: str | None = None
    ranking: list[dict[str, Any]] = field(default_factory=list)


# {
#   責務: [
#     normalize_weights: 重みを検証し、評価軸ごとの有限な非負数にする
#   ]
#   引数: [
#     weights: 評価軸名から重みへの辞書
#   ]
#   戻り値: [
#     normalized: 検証済みの重み
#   ]
#   エラー: [
#     ValueError: 軸名が空、重みが数値でない・有限でない・負、または合計が0以下
#   ]
# }
def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("weight names must be non-empty strings")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {name} must be a finite non-negative number")
        normalized[name] = float(weight)
    if normalized and sum(normalized.values()) <= 0:
        raise ValueError("weights must have a positive sum")
    return normalized


# {
#   責務: [
#     aggregate_score: 1件の得点を、選択基準に従って1つの数値にする
#   ]
#   処理: [
#     1: 重みがあれば、重み付き平均を計算する
#     2: 評価軸の指定があれば、その成分を使う
#     3: どちらもなければ、評価器が定義する総合得点を使う
#   ]
#   引数: [
#     score: 候補の生得点
#     weights: 検証済みの重み
#     component: 評価軸名。使わなければ空
#   ]
#   戻り値: [
#     value: 順位づけに使う数値
#   ]
#   エラー: [
#     ValueError: 必要な評価軸または総合得点がない
#   ]
# }
def aggregate_score(score: Score, weights: dict[str, float], component: str) -> float:
    if weights:
        missing = sorted(set(weights) - set(score.components))
        if missing:
            raise ValueError(f"score {score.id} has no component(s): {missing}")
        total = sum(weights.values())
        return sum(w * score.components[name] for name, w in weights.items()) / total
    if component:
        if component not in score.components:
            raise ValueError(f"score {score.id} has no component: {component}")
        return score.components[component]
    if score.overall is None:
        raise ValueError("SELECT requires an overall score for each candidate")
    return score.overall


# {
#   責務: [
#     select_images: 得点集合に対応する候補から、上位top件を決定的に選ぶ
#   ]
#   処理: [
#     1: 得点集合の種類・由来・候補IDの一致と、保存形式内の得点の一致を検証する
#     2: 選択基準を決め、各候補を集約して順位づけする。同点は候補IDの昇順にする
#     3: 上位top件と、全候補の順位・生の成分を返す
#   ]
#   引数: [
#     source: 画像または画像集合
#     scores_artifact: sourceを評価した得点集合Artifact
#     top: 選ぶ件数
#     weights: 評価軸ごとの重み。使わなければ空
#     component: 評価軸名。使わなければ空
#   ]
#   戻り値: [
#     selection: 選ばれた画像と順位づけの根拠
#   ]
#   エラー: [
#     TypeError: 得点集合の型が違う
#     ValueError: 由来・候補ID・得点の不一致、topの範囲外、重みと評価軸の同時指定、集約値が有限でない
#   ]
# }
def select_images(
    source: Artifact,
    scores_artifact: Artifact,
    *,
    top: int = 1,
    weights: dict[str, Any] | None = None,
    component: str = "",
) -> ImageSelection:
    if scores_artifact.artifact_type != "ScoreSet":
        raise TypeError("SELECT expects a ScoreSet artifact")
    if source.id not in scores_artifact.parent_artifact_ids:
        raise ValueError("ScoreSet does not originate from the input image artifact")
    score_set = ScoreSet.from_dict(scores_artifact.metadata.get("score_set"))
    # Trajectoryに保存される得点と、選択に使う得点を一致させる。
    if scores_artifact.scores != score_set.scores:
        raise ValueError("ScoreSet artifact scores do not match metadata.score_set")
    candidates = image_candidates(source)
    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    scores_by_id = score_set.scores_by_candidate()
    if len(candidates_by_id) != len(candidates) or set(candidates_by_id) != set(scores_by_id):
        raise ValueError("candidate IDs must match ScoreSet exactly")
    # 明示された候補IDがある場合は、保存形式内の順序ずれを拒否する。
    for candidate_id, score in scores_by_id.items():
        recorded_id = score.metadata.get("candidate_artifact_id")
        if recorded_id is not None and recorded_id != candidate_id:
            raise ValueError("score candidate ID does not match ScoreSet order")
    if type(top) is not int or not 1 <= top <= len(candidates):
        raise ValueError(f"top must be an integer between 1 and {len(candidates)}")
    normalized = normalize_weights(weights or {})
    if normalized and component:
        raise ValueError("weights and component cannot be combined")

    aggregates = {
        candidate_id: aggregate_score(score, normalized, component)
        for candidate_id, score in scores_by_id.items()
    }
    # 集約の途中でオーバーフローしたNaN・無限大は順位を不定にするため拒否する。
    if not all(math.isfinite(value) for value in aggregates.values()):
        raise ValueError("aggregate score must be finite")
    order = sorted(aggregates, key=lambda candidate_id: (-aggregates[candidate_id], candidate_id))
    return ImageSelection(
        candidates=[candidates_by_id[candidate_id] for candidate_id in order[:top]],
        scores=[scores_by_id[candidate_id] for candidate_id in order[:top]],
        score_set=score_set,
        method="weighted" if normalized else "component" if component else "overall",
        weights=normalized,
        component=component or None,
        ranking=[
            {
                "candidate_artifact_id": candidate_id,
                "score_id": scores_by_id[candidate_id].id,
                "aggregate": aggregates[candidate_id],
                "components": dict(scores_by_id[candidate_id].components),
                "overall": scores_by_id[candidate_id].overall,
            }
            for candidate_id in order
        ],
    )
