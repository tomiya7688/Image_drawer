"""評価済み画像候補から最良の一枚を選ぶ。"""

from __future__ import annotations

from image_drawer.core import Artifact, Score, ScoreSet
from image_drawer.evaluation.select_images import select_images


# {
#   責務: [
#     select_best_image: 得点集合に対応する候補から総合得点が最良の画像を選ぶ
#   ]
#   処理: [
#     select_imagesをtop=1・総合得点で呼び、先頭の候補を返す
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
    selection = select_images(source, scores_artifact, top=1)
    return selection.candidates[0], selection.scores[0], selection.score_set
