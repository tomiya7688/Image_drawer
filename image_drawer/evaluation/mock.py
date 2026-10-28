"""候補画像ごとに再現可能なテスト用得点を生成する。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from image_drawer.core import Artifact, Score, ScoreSet

if TYPE_CHECKING:
    from image_drawer.steps.base import StepContext


# {
#   責務: [
#     image_candidates: 評価対象の画像を同じ形式で取り出す
#   ]
#   処理: [
#     Imageを一件に、ImageSetの保存済み候補を画像レコードに変換する
#   ]
#   引数: [
#     artifact: 評価対象のImageまたはImageSet
#   ]
#   戻り値: [
#     candidates: 元の順序を保った画像候補
#   ]
# }
def image_candidates(artifact: Artifact) -> list[Artifact]:
    if artifact.artifact_type == "Image":
        return [artifact]
    if artifact.artifact_type != "ImageSet":
        raise TypeError("evaluation expects Image or ImageSet")
    payload = artifact.metadata.get("candidates")
    if not isinstance(payload, list) or not payload:
        raise ValueError("ImageSet metadata.candidates must be a non-empty list")
    # 保存形式を検証してから復元し、候補以外のArtifactを混入させない。
    candidates = []
    for value in payload:
        if not isinstance(value, dict):
            raise TypeError("ImageSet candidates must be serialized Artifact objects")
        candidate = Artifact.from_dict(value)
        if candidate.artifact_type != "Image":
            raise ValueError("ImageSet candidates must have type Image")
        candidates.append(candidate)
    return candidates


# {
#   責務: [
#     MockEvaluator: 画像候補ごとに決定的なテスト用得点を生成する
#   ]
#   フィールド: [
#     なし
#   ]
#   処理: [
#     候補の順序と得点の順序を揃えたScoreSetを返す
#   ]
# }
class MockEvaluator:
    name = "mock"
    version = "v1"

    # {
    #   責務: [
    #     evaluate: 候補の文字列表現を個別に評価する
    #   ]
    #   処理: [
    #     各候補の得点と候補IDを同じ位置に保存する
    #   ]
    #   引数: [
    #     candidates: 評価する画像候補
    #     context: 実行と得点の識別子
    #   ]
    #   戻り値: [
    #     score_set: 評価器と候補に対応する得点集合
    #   ]
    # }
    def evaluate(self, candidates: list[Artifact], context: StepContext) -> ScoreSet:
        scores = []
        for index, candidate in enumerate(candidates):
            rendered = str(candidate.metadata.get("value", ""))
            quality = round((sum(rendered.encode("utf-8")) % 1000) / 1000, 6)
            scores.append(
                Score(
                    id=context.score_id(str(index)),
                    evaluator=self.name,
                    evaluator_version=self.version,
                    overall=quality,
                    components={"quality": quality, "determinism": 1.0},
                    metadata={"candidate_artifact_id": candidate.id},
                )
            )
        return ScoreSet(
            id=context.artifact_id(),
            evaluator=self.name,
            evaluator_version=self.version,
            candidate_artifact_ids=[candidate.id for candidate in candidates],
            scores=scores,
        )
