> **対象読者:** Workflow DSL・Step adapter・GUIを実装する開発者
> **目的:** Workflowと筆跡イベントの責務を分け、実装されていないOperationを実装済みと誤認しない。

# Workflow DSL（上位工程の保存仕様）

## 1. 最重要：DSLは筆跡の学習データではない

- **Workflow DSL**：Workflow Step（入力、描画セッションの起動、評価、選択、出力）の**接続やパラメータ**を保存する。
- **DrawingEpisode JSON**：1枚の制作中に行った**全ての色選択・ブラシ選択・レイヤー操作・ストローク・消去・工程移行・終了**を保存する。
- **DrawingPolicy**：現在の描画状態と履歴を見て次の`DrawingAction`を予測する。

これらを混同しない。`SKETCH`や`COLOR`を画像変換Stepとして一度だけ呼び、筆跡履歴が一切ない場合は、Image Drawerの目的を満たさない。

色・道具・レイヤー・ストロークの厳密な定義は [制作過程 v1](drawing_process_v1.md) にのみ置く。

## 2. 現在の実装済みDSLと将来の目標を区別

**基幹のWorkflowはtxt2img専用の入力契約を持つ**。学習時は[画像のみ自己描画学習](image_only_learning_v1.md)に従い、生成時は[固定制作工程](fixed_drawing_workflow_v2.md)の**ラフ→下書き→本描き→基本色→陰影・仕上げ→修正・確認→完成**を順に実行する。推論時の唯一の必須外部入力は`INPUT prompt: Text`であり、`Image`・`LineArt`・`DrawingEpisode`等の外部入力は必須ではない。`DRAW_EPISODE`は外部画像を読まず、白紙から生成を開始する。`REPLAY_EPISODE`はモデル自身が直前に生成したEpisodeを画像にする。主たるワークフロー出力は`Image`。Episodeはその親Artifactとして保存される。

この入力契約を持たないimg2img／既存画面の編集／人間の実演Replayは**別モード**。モデル自身のActionによる生成なしでPart合成や固定Scriptを実行し、`OUTPUT image`を返すだけではtxt2imgの合格にならない。


現在の`image_drawer/dsl/parser.py` は`INPUT`、Step呼び出し、`OUTPUT`を`WorkflowSpec`に変換する。Step名が使えるかどうかは`StepRegistry`に依存する。既存の主要なregistryにある`RETRIEVE_PARTS`、`SELECT_PARTS`、`COMPOSE`は**Part Bank用の補助機能**であり、以下の描画用目標Stepが既に実装されたという意味ではない。

目標となる描画用Workflowの概念例（**現時点で実行可能なDSLとは主張しない**）：

```text
INPUT prompt: Text
episode = DRAW_EPISODE(prompt, policy="drawing-policy-v1", max_actions=1000)
image = REPLAY_EPISODE(episode)
OUTPUT image
```

- `DRAW_EPISODE`（**新設予定**）：PromptとPolicyを使い、固定StageControllerの**ラフ→下書き→本描き→基本色→陰影・仕上げ→修正・確認→完成**に沿って1行動ずつ`DrawingEngine`へ適用し、`finish`までの`DrawingEpisode`を返す。Stageを飛ばす・逆行する単一画像生成は不可。
- `REPLAY_EPISODE`（**新設予定**）：保存済み`DrawingEpisode`を厳密に再生して最終`Image`を返す。関連するEpisode URIとIDをArtifact系譜へ残す。
- **v2固定工程**：`rough/sketch/final_drawing/base_color/shading/refine`の順序は[工程v2](fixed_drawing_workflow_v2.md)で固定する。`SKETCH`／`LINE`／`COLOR`／`CORRECT`は旧互換名または各工程内部の説明に過ぎず、**任意の順序をAIに選ばせない**。工程移行は新v2の`advance_stage`で記録する。旧v1の`set_phase`は画素Replay互換だけに残す。
- `EVALUATE`／`SELECT`（一部既存）：上位Artifactの比較に使う。描画時の次アクション推論を代替しない。

Stepの**契約案**（実装IssueでStepSchemaに反映するまで未実装）：

| Operation | Input Artifact | 必須/既定parameter | 主出力 | 不合格となる挙動 |
| --- | --- | --- | --- | --- |
| `DRAW_EPISODE` | `Text` | `policy: str`必須、`max_actions: int=1000`（1..19999、`finish`を除くAction上限。Stage移行も含む） | `DrawingEpisode` | Pixelだけを直接書き換える／行動ログを作らない |
| `REPLAY_EPISODE` | `DrawingEpisode` | なし | `Image` | 正本eventsを再生せず画像をそのまま返す |

`max_actions`の上限に達して`finish`が出ない場合、エラーを記録し**完成エピソードと偽らない**。既存Workflowの`Artifact.uri`を使ってJSONを参照し、画像は追加Artifactとして独立保持してよい。

## 3. DSL構文上の規則（現行Parserに合わせる）

- `INPUT <name>: <Type>` で入力を宣言し、`OUTPUT <variable>`で出力する。
- Stepは`name = UPPERCASE_STEP(dependency, parameter=value)`の形。
- 位置引数は先行する変数への依存。名前付き引数でSchemaにあるものはパラメータ、その他の先行変数参照は名前付き依存として扱う。
- パラメータはstring/int/float/bool/list/string-key dictionaryのリテラル。任意式、関数呼び出し、分岐・繰り返しを許さない。
- Parsed`WorkflowSpec`を静的validateし、GUIとCanonical serializerでround-tripできること。
- 不明なStep、存在しない変数、引数の型・個数違い、依存の不一致をrejectする。

このDSLは**ActionEventの配列を入れる場所ではない**。色や座標を`DRAW_EPISODE`のDSLパラメータに直書きして固定の筆跡を再生する仕組みを、学習済みPolicyと呼んではいけない。

## 4. 単位ごとの保存先

```text
WorkflowSpec/DSL
    └─ DRAW_EPISODE Step
          ├─ DrawingPolicy（次の行動を逐次予測）
          ├─ DrawingEngine（Actionを適用）
          └─ DrawingEpisode（時系列events、URIとして保存）
    └─ REPLAY_EPISODE Step
          └─ Image Artifact（Episode Artifactを親として記録）
    └─ EVALUATE / OUTPUT（必要な場合）
```

GUIから絵師が直接描いた場合も、同じ`DrawingEpisode`を生成する。Workflowを使っていない場合は`WorkflowSpec`を偽造する必要はない。Runtimeから起動した場合は`Trajectory`に関連ID・実行環境を記録する。

## 5. 実装・受け入れ基準

- 新Stepの入出力Artifact typeはSchemaに明記し、未登録ならDSL parse時に失敗すること。
- `DRAW_EPISODE`で予測した各行動が`DrawingEpisode`に残り、`REPLAY_EPISODE`が完全再生できること。
- 描画結果が同じでも、履歴が欠落していれば不合格。
- `COLOR`の名前だけで区域の塗りつぶしを実装しないこと。
- ストロークの意味論は[受け入れ試験](drawing_process_acceptance.md)に従う。色選択をWorkflowの固定parameterへ置き換えないこと。

既存のPart用DSL・mock workflowは破壊せず補助機能として維持する。描画学習側で必要なStepが増える場合でも、DSL syntax全体をモデル固有に変更しない。
