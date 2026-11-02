> **対象読者:** DrawingPolicy・Trainer・Dataset builder・評価基盤の実装者
> **目的:** 「高精度」を空約束でなく、モデルの具体的な入出力・損失・学習段階・検証条件へ落とす。
> **状態:** 実装前の規範的Baseline設計案。性能値は未測定であり、性能保証ではない。

# DrawingPolicy v1：高精度を狙う学習設計と評価プロトコル

## 0. 何を高精度と呼ぶか

最重要の課題は**実際の絵師が行う操作・判断を逐次学習すること**。完成PNGの画素類似だけでは、工程の学習を証明できない。一方、次行動の分類正解率が高いだけでも長時間の自律描画ができるとは限らない。

評価を次の4段階に分ける。

1. **データ忠実性**：人間の実際の操作が正しい順番・レイヤー・色・軌跡で再生できる。ここは100%一致が要求できる工学的な不変条件。
2. **教師強制（offline）**：実演のprefixを入力したとき、次の行動・色・筆跡をどれだけ当てられるか。
3. **自律描画（closed-loop）**：モデル自身が選んだ操作後の状態から次の操作を決め、修正・終了まで破綻せず進められるか。
4. **実用的な画質と工程**：絵師の工程に沿い、ブラインドな人間評価でまともな作品になるか。

上の1は完全一致を合格条件にできる。しかし2〜4は未知の人間データや未知のpromptに依存し、**「絶対に高精度」を保証する方法は存在しない**。ベースライン・信頼区間・失敗例を測定し、改善が再現した場合だけ性能向上を主張する。

## 1. Policyの厳密な入出力

```python
class DrawingPolicy:
    def predict(self, observation: DrawingObservation, rng: RandomState) -> DrawingAction:
        ...
```

`DrawingObservation`に**次の決定前**の情報だけを入れる：

| 入力 | 必要な情報 |
| --- | --- |
| `prompt` | 作品の指示文。将来の操作説明・完成画像の説明を混ぜない |
| `canvas` | 現在の合成プレビューRGBとcanvas寸法 |
| `layers` | **全レイヤーの順序・RGBA・作業レイヤーindex**。合成画像だけでは重なり／消去を予測できない |
| `tool_state` | `selected_color_id`、`selected_brush_id`、`phase`、`finished` |
| `history` | `events[:t]`のタイプ・パラメータ・actor・時刻。`events[t:]`は**絶対に渡さない** |
| `allowed_actions` | `DrawingEngine`から生成した合法なAction typeと引数候補のmask |
| `budget` | これまでのAction数、残りイベント数・point予算、実装上限 |

実装上の視覚特徴は小さくしてよいが、**全レイヤーと選択状態に由来する情報を失ったまま**学習の最終版としない。最初はキャンバス全体の低解像度特徴に加え、各レイヤーの不透明度・ID・順序情報を別埋め込みする。位置精度が不足したら直近筆先付近の高解像度cropを追加する。既存の実画素は保持し、モデル用のresizeでActionの正本座標を書き換えない。

## 2. 階層型のAction decoder（Baselineの必須構造）

モデルは次の確率を分解して予測する。

```text
p(action | observation)
  = p(type | observation)
    * p(payload | type, observation)

type : 8種（create_layer, select_layer, set_phase,
              select_color, select_brush, stroke, erase, finish）
payload:
  create_layer  -> NEW_LAYER（作成順indexから決定的なID/nameを生成）
  select_layer  -> existing layer index（最大16候補）
  set_phase     -> 5分類
  select_color  -> **固定256色の分類**
  select_brush  -> 別の分類（v1は1候補）
  stroke        -> width_px, opacity_u8, POINT_0, ..., POINT_END
  erase         -> width_px, POINT_0, ..., POINT_END
  finish        -> payloadなし
```

**多種類の出力を一つの巨大な「色×ブラシ×座標×線幅」分類にしない。** 色とブラシは独立し、色の意味は全実行・全Datasetで固定。

### 2.1 ストローク内の可変長点列

ストローク全体のJSONを一回で出すのではなく、`point decoder`で点を1つずつ出して`POINT_END`で終了させる。Baselineは**整数絶対座標**を使い、xとyをそれぞれ`0..1023`の1024クラスの分類として予測する（キャンバス外値はmask）。対応する座標ペアを両方出して初めて1点とする。1〜4096点の制約、開始直後のEND禁止、最大点数到達での不正END禁止を強制する。

- 入力には直前の点、現在canvas寸法、作業レイヤー、周辺の画像特徴を含める。
- `POINT_END`は点数が1以上のときだけ合法。学習時は実演の実点数でTeacher forcingする。
- 点の位置の表現改善（直前からの差分、mixture density、曲線制御点等）は**同じ評価条件**で比較する候補。勝手に正本座標を別の形式へ変更しない。
- 推論時の全点を受理した**後に1つの`stroke`イベント**としてRendererへ渡す。部分的に点を描いてから途中失敗を隠さない。
- 人間のstrokeの分割・点数分布をDatasetで測定し、点列が長すぎるものを訓練から無断削除しない。上限に達したことを失敗として集計する。

### 2.2 Layer IDと文字列を無理にモデル予測させない

実演ではレイヤー名に自由文を使えるが、モデルが新しい文字列IDを自由生成すると衝突する。**学習時だけ**`layer_id`をそのエピソード内の作成順`layer_index=0..15`へ正規化する。原始イベントのID/nameは変更しない。

- `create_layer`は次の未使用indexが決まるので、`layer_id`を`layer_<index>`、`name`を`レイヤー<index>`として決定的に生成。
- `select_layer`は既存のindexだけを選択。作成順・参照関係を保つ。
- 教師の元IDが任意でも、同じレイヤーの指示なら**正規化後のindex**で損失と正解を計算する。
- ブラシは単一種類のv1では常にその1種類。ブラシのaccuracy=100%を高性能の根拠にしない。

## 3. 合法Actionのmaskと失敗記録

**状態から合法候補を決定する純粋なValidator**を描画器と共用する。学習時と推論時で違う条件を使わない。

- active layerなし→`stroke/erase`無効、色未選択→`stroke`無効、ブラシ未選択→`stroke/erase`無効、phase未設定→`stroke/erase`無効。
- layers=16→`create_layer`無効。既存レイヤー外の`select_layer`無効。
- finished=true→全Action無効。budget不足→候補は制限するが、勝手にfinishを生成して成功扱いしない。
- 色番号は常に`0..255`。x/yは各canvasの範囲、太さは`1..64`、opacityは`1..255`。点列は`1..4096`。
- 全候補が無効／モデルが不正形式を出す場合は`invalid_prediction`として記録し、モデル成功に置換しない。推論時の再試行回数・回復策は構成に記録する。

**maskによる「有効率100%」と、モデルの意思決定が正しかったかは別の指標**。maskなしのraw提案違反率も残す。

## 4. 学習用sampleと目的関数

人間実演`events=[a0,a1,...]`について、各tで**実演の直前まで**を再生し、`(prompt, state_t, history[:t], target=a_t)`を生成する。`actor=human`で許諾があるActionのみ教師ラベルにする。混在episodeに含まれるhuman correctionは独立した通常の実演とは別datasetに分けて評価する。

Baselineの損失（条件付きhead以外は計算しない）：

```text
L_sample = CE(type)
         + 1[payloadあり] * mean(CE(各離散payload field))
         + 1[type∈{stroke,erase}] * mean(CE(x), CE(y), CE(POINT_END))
```

- `mean`は**1イベント単位**で取る。長いstrokeだけが学習損失を独占しないよう、point数による単純な合計はしない。
- 座標・色・線幅・不透明度の正解値は`DrawingEpisode`から直接得る。使わないheadへ損失を計算しない。
- class imbalanceにはtraining側の出現頻度を使って重みを算出し、clipされた重み・seedをconfigへ残す。validation/testの分布から重みを決めない。
- 教師Actionと異なるが同じ描画結果になる行動は存在するため、exact matchだけを唯一のlossとしない。**モデルから得られる確定的なstroke maskの一致**を別指標とする。
- 差分計算可能なRendererや画像知覚損失は**将来の補助損失候補**。その追加によって人間の操作ラベルを捨てない。

データ増強（回転・左右反転・間引き）は初期Baselineで**無効**。特に「左に描く」などのprompt意味を壊すため、入れるときはprompt・座標・レイヤー・状態を一貫して変換し、ReplayとGolden testを通してから別versionで比較する。

## 5. 学習ステージの順序（通過条件）

| Stage | 目的 | ここで合格とするもの |
| --- | --- | --- |
| A. Contract | schema、geometry、Replay、データ出所検証 | **入力と全中間画素の完全一致** |
| B. Tiny overfit | syntheticの小規模Action列を過学習できることを確認 | loss低下、checkpoint復元、teacher-forced復元。**実演の高精度証明ではない** |
| C. Human imitation | 許諾済みhuman traceから状態→Actionを学ぶ | 未見作品・未見作者に対しBaselineより良いこと |
| D. Closed-loop | モデル自身の各出力を実際に描画器へ適用 | 有効Actionだけでなく修正・終了・品質が成立 |
| E. Correction data | モデルが失敗したStateを人間が訂正した実行履歴を追加 | actorと混在文脈を維持し、修正による改善を検証 |

Cが無い段階では「人間の制作過程を学習した」と報告しない。Dが無ければ「自律制作の精度が高い」と報告しない。

Closed-loopは実演の状態分布とズレるため、Dの失敗Stateを人間が再ラベルする**dataset aggregation（DAgger系）**を後続の候補にする。人間が実際に与えた訂正だけをexpert signalとして扱い、モデルの推定修正を人間ラベルに昇格させない。モデルの失敗に対する人間の修正は有用だが、独立した絵師の実演と統計的に混ぜない。

## 6. 比較対象・実験設定・checkpoint

最低限の比較：

1. `Frequency baseline`：Action typeと色の学習集合の最多値を常に選ぶ。
2. `History baseline`：直近Action type・前回の色・選択レイヤーを使う規則モデル。
3. `Imitation baseline`：この文書の階層型条件付きAction予測。
4. `Ablation`：canvas抜き、layer別情報抜き、history抜き、error-recoveryデータ抜き、色分類 vs 連続RGBの**比較実験のみ**（本番の固定256色は変更しない）。

Dataset、artist/work split、palette hash、renderer version、tokenization version、checkpointとseedを固定し、**最低3 seed**で全比較を実施。学習設定・モデルparameter数・train/validation/test件数・ハードウェアを必ず報告する。特定のGPU性能を想定して未計測の速度を保証しない。

Checkpointには少なくともモデル重み、optimizer/step、random seed状態、model/tokenizer/config version、split hash、palette SHA、renderer/brush versionを含める。入力契約が違えばload拒否し、推論時に黙って変換しない。

## 7. 最低限の評価表（全列を別々に保存）

| グループ | 評価項目 | 計算法・注意 |
| --- | --- | --- |
| Operation | Macro-F1、Action type NLL、top-1 | 頻出strokeだけで精度を水増ししない |
| 色 | 256色top-1/top-5、色選択時の知覚距離分布 | 分母は人間の`select_color`数。色のないepisodeではN/A |
| 幾何 | stroke/eraseの点列終了正解率、点数誤差、画素mask IoU | 正解の直前状態へ予測strokeを適用して比較 |
| Layer/phase | 選択indexの正解率、フェーズ誤り、修正Actionのrecall | layer数・phase別の分母を明示 |
| Closed-loop | 完了率、max_actions超過、raw無効提案率、Replay失敗率 | 最終画面・各中間状態は同じRendererで再生 |
| 視覚品質 | ブラインド人間A/B評価、作品適合、線・色・修正の評価 | 評価者とpromptを隠さず、どの条件をブラインドにしたか報告 |
| 効率 | 有効操作数、学習時間、推論時間、メモリ | 同じ計測環境で比較 |

**特に区別**：教師強制時の次アクション正解率はClosed-loop完了率ではない。valid action maskで作った100%合法率は制作能力でもない。レイヤーや色選択の正解は、最終画像一致に吸収せず独立評価する。

## 8. 精度に関する公開ゲート（実データ確保後）

- 事前にpromptカテゴリ、想定canvasサイズ、作品難度、artist数、作業時間、比較Baselineと**主要指標**をpre-registerし、固定したtestへ戻ってhyperparameterを決めない。
- 未見作品・未見作者の2系統で評価。各指標にサンプル数・分母・**artist/work group単位のbootstrap 95%信頼区間**を付ける。単一色やブラシの正解率だけで評価しない。
- **「ベースラインより精度改善」と書いてよいのは**：主要指標の差について事前に決めた方向の改善があり、group bootstrapの95%CIの下端が0を超え、Closed-loopの重要な安全・破綻指標に悪化がない場合。優位性がなければ実験失敗／未確認と報告する。
- 実際の高い画質という主張には、独立のブラインドA/B人間評価も必要。評価者・作品数・採点方法・好みの分布を公開し、恣意的な画像だけを提示しない。
- 十分なhuman datasetがなければtestの改善やCI成功を「必ず良い絵を描ける」と言い換えない。必要数はpilot varianceを測って決める。データ不足は**計画の不確実性**として明示する。

## 9. 画質の上限を決める未実装要素とバージョン移行

**注意：v1の`hard_round_v1`、整数画素座標、アンチエイリアスなし、固定8bit不透明度だけでは、高品質な一般イラストを保証できない。** これらは最初の意味論と再生テストを確立するための基準。人間が筆圧、傾き、質感、スムーズな曲線を使う制作過程の精度は、その情報を記録していないモデルには学習できない。

高品質化を実証する際の次version候補は次を**別契約**として設計・比較する。

| 追加対象 | 記録すべき情報 | 正本との整合 |
| --- | --- | --- |
| サブピクセルの軌跡 | 元ペン座標の高分解能時系列。座標の単位と量子化bit幅を固定 | v1の整数`points`へ**偽の精度**を付けない |
| 筆圧とブラシ | 各サンプルの本当の筆圧有無・測定値、`brush_id`、筆先shape、stroke幅・硬さ、renderer version | 固定256色は維持し、色とブラシを独立の選択にする |
| アンチエイリアス | supersample位置、coverage→alphaの丸め、レイヤー合成規則 | v1のgoldenを破壊せず、新versionで別goldenを作る |
| Undo・修正 | 元Action参照とレイヤー復元の意味、undo/redo履歴の順序 | 画素だけ巻き戻し、実際の行動履歴を消さない |
| モデル用画素特徴 | 高解像度ROIと全レイヤー文脈の両立 | 最終画像だけに情報を潰さない |

**まず実演データの測定でどれが最も大きな誤差を生むか順位付け**し、改善候補を別versionでA/B比較する。色数を増やしたりRGB直接予測へ戻したりせず、**256色の固定パレットの範囲内で**視覚品質を改善する。将来の高精細Rendererが学習しやすくなっても、実演不足・モデルの判断誤りは別問題として残る。

## 10. 適用研究・仕様へ取り込まない部分

- [Sketch-RNN](https://research.google/pubs/a-neural-representation-of-sketch-drawings/)：人間のペン移動を可変長の逐次系列として扱う考え方の参考。固定256色・複数レイヤーは本PJ側で別途扱う。
- [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)：学習済みpolicyが遭遇したStateへの専門家訂正と、分布ずれ対策の参考。
- [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset)：筆跡時系列の収集例。**そのままで色・レイヤー・修正ラベルが揃っているわけではない**。

これらの研究を参照することと、Image Drawerで学習精度が出たことは別。**測定値がない段階で性能保証をしない。**
