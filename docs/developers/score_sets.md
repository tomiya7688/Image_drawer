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

`Score`には候補IDがないため、件数が一致していても得点の順序が誤っている場合は検出できません。作成側で同じ候補のIDと得点を同じ位置に追加してください。モック評価器は候補の順番でIDと得点を追加します。`metadata`の内容はJSON互換性を検証しないため、保存できる値を指定してください。

Runtimeのモック`EVALUATE`は、`Image`または`ImageSet`を受け取り、候補ごとに得点を生成します。`ImageSet`の`metadata.candidates`には、`Image`型のArtifactを`to_dict()`で変換した非空の配列を指定してください。出力Artifactの`metadata.score_set`から`ScoreSet.from_dict()`で得点集合を復元できます。Trajectoryとの互換性のため、同じ得点をArtifactの`scores`にも保持します。評価器名は`mock`のみを受け付け、それ以外はエラーになります。

`SELECT`は総合得点が最も高い画像を選び、同点は候補IDの昇順で決めます。得点集合の由来と候補IDが入力画像と一致しない場合、Artifactの`scores`と`metadata.score_set`が一致しない場合、総合得点が欠ける場合は失敗します。選択した候補IDと得点IDはTrajectoryに記録されます。現在は`top=1`のみ対応し、top-kと重み付き選択は未対応です。モック得点は画像品質を測定するものではありません。

## OpenCLIPによる相対評価

`openclip_quality`バックエンドは、画像と「高品質なイラスト」「低品質なイラスト」の英文とのcosine類似度を計算し、その差を`overall`に保存します。数値は文章と画像の相対的な一致度であり、人間の評価に較正された品質点や確率ではありません。元の2つの類似度も`components`に保存します。

利用者は`python -m pip install -e ".[openclip]"`を実行し、OpenCLIP ViT-B-32 `laion2b_s34b_b79k`のsafetensors重みをローカルの`models/`に用意してください。`EVALUATE`のbackendに`openclip_quality`、`weights_path`にそのファイルを指定します。評価対象のImage Artifactにはローカル画像のパスを`uri`として設定します。モデルや画像はGitへ追加しないでください。

重みのSHA-256を評価器のバージョンとして記録します。透明部分は白背景に合成し、OpenCLIPの標準前処理で中央を切り出します。そのため、画像の端は評価に含まれない場合があります。
