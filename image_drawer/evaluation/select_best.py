"""評価済み画像候補から最良の一枚を選ぶ。"""

from __future__ import annotations

from image_drawer.core import Artifact, Score, ScoreSet
from image_drawer.evaluation.mock import image_candidates


# {
#   責務: [
#     select_best_image: 得点集合に対応する候補から最良の画像を選ぶ
#   ]
#   処理: [
#     入力の由来と候補IDを検証し、総合得点とIDで決定的に選ぶ
#   ]
#   引数: [
#     source: 画像または画像集合
#     scores_artifact: sourceを評価した得点集合Artifact
#   ]
#   戻り値: [
#     candidate: 選択された画像
#     score: 対応する生得点
#     score_set: 評価器の情報を持つ得点集合
#   ]
# }
def select_best_image(
    source: Artifact, scores_artifact: Artifact
) -> tuple[Artifact, Score, ScoreSet]:
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
        if score.overall is None:
            raise ValueError("SELECT requires an overall score for each candidate")
    best_id = min(
        scores_by_id, key=lambda candidate_id: (-scores_by_id[candidate_id].overall, candidate_id)
    )
    return candidates_by_id[best_id], scores_by_id[best_id], score_set
