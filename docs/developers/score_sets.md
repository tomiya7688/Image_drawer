# 評価結果の保存形式

## 背景と目的

複数の画像候補を異なる評価器で比較するには、候補と生得点を対応付け、評価器の名前とバージョンを保持する必要があります。`ScoreSet`は一つの評価器による得点集合を表します。

## 使い方

`image_drawer.core.ScoreSet`を作成し、候補IDと`Score`を同じ順序で指定してください。`scores_by_candidate()`はまとめて参照するための対応表を返します。`to_json()`と`from_json()`で保存・復元できます。

```python
from image_drawer.core import Score, ScoreSet

scores = ScoreSet(
    id="evaluation-1", evaluator="mock", evaluator_version="v1",
    candidate_artifact_ids=["image-1"],
    scores=[Score(id="score-1", evaluator="mock", evaluator_version="v1",
                  components={"quality": 0.8}, overall=0.8)],
)
restored = ScoreSet.from_json(scores.to_json())
assert restored.score_for("image-1").components["quality"] == 0.8
```

候補IDと得点IDの重複、件数の不一致、評価器名やバージョンの不一致、非数値・真偽値・非有限値は拒否します。作成後に内容を変更した場合も、対応表の取得と保存時に再検証します。空集合は保存できます。複数候補を参照する場合は、`score_for()`を繰り返す代わりに`scores_by_candidate()`を一度取得してください。

`Score`には候補IDがないため、件数が一致していても得点の順序が誤っている場合は検出できません。作成側で同じ候補のIDと得点を同じ位置に追加してください。評価処理との接続時には、この対応が保たれることを確認する必要があります。`metadata`の内容はJSON互換性を検証しないため、保存できる値を指定してください。

この段階ではデータ形式だけを提供します。Runtimeの`EVALUATE`、`SELECT`、重み付き選択にはまだ接続していません。
