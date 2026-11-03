> **対象読者：** 固定工程を実装する人・契約テストを書く人
> **目的：** [初見向けの7工程表](fixed_drawing_workflow_v2.md)を、実装差のない状態遷移・レイヤー・失敗処理に落とす。
> **状態：** v2設計案。実際のStageControllerは**未実装**。画像を用いた自己描画学習も未実装。

# 工程Controllerの動作契約 v2

## 1. ここでいう「固定」とは

工程の順序は **ラフ → 下書き → 本描き → 基本色 → 陰影・仕上げ → 修正・確認 → 完成**。画像生成時は文章、学習時は画像から自動抽出した文章を受け、**描画Policyが各工程の中で操作を学ぶ**。工程名・順序と必要なレイヤーはControllerが管理する。

この文書は**実際に絵をうまく描けること**を保証しない。自動採点と学習の正本は[画像だけで学ぶ仕様](image_only_learning_v1.md)。

### 状態と登場人物

| 名前 | 意味 | 具体例 |
| --- | --- | --- |
| `DrawingState.stage_id` | 現在の工程 | `rough` |
| `Layer.role` | そのレイヤーの役割 | `final_drawing` |
| `active_layer_id` | **次の筆跡を書く先** | `layer_003` |
| `DrawingPolicy` | 次の色・筆・座標・工程移行を**提案** | Gペン、5px、線を引く |
| `StageController` | 工程移行・役割・表示順を**検証して実行** | ラフから下書きへ |
| `DrawingEngine` | 許可された描画Actionを正確な画素へ適用 | `stroke_v2` |
| `Evaluator` | 学習時だけ目標画像と比較 | 形が近くなったか |

ユーザーには学習画像しか求めない。Controllerは対象画像を参照しない。Policyも対象画像・対象画像特徴・その画像を見て作った中間正解画像・同一trial内でのtarget依存Rewardを直接観測しない。

## 2. 入場時に何を用意するか

初期状態は白紙、`stage_id=rough`、空の行動履歴。Controllerが**工程入場を履歴へ記録し、必要なレイヤーを作る**。新規レイヤーは透明。

| 入った工程 | Controllerが作る役割レイヤー | AIが追加作成できる役割 |
| --- | --- | --- |
| ラフ | `rough` | `rough` |
| 下書き | `sketch` | `sketch` |
| 本描き | `final_drawing` | `final_drawing` |
| 基本色 | `base_color` | `base_color` |
| 陰影・仕上げ | `shade` | `shade`, `highlight` |
| 修正・確認 | **新しいレイヤー不要** | `correction` |

`refine`では、それまでの可視レイヤーを**選んで直すことも、修正用レイヤーを追加することもできる**。

**工程が変わるたびに`active_layer_id`を未選択にする。** それまでのブラシ・色・サイズ・不透明度は保持してよいが、AIが書込可能なレイヤーを明示的に選び直さなければ次の筆跡は拒否する。Controllerの自動レイヤー作成・選択リセット・表示変更も、Replayで復元できるよう履歴へ記録する。

## 3. 操作を受理する順番

Controllerは次の順序で1つの提案Actionを審査する。

1. **基本の正当性**：型・必須キー・数値・ID・Action数や点数の上限・`finish`済みでないこと。
2. **工程による権限**：現在のstage、ブラシmode、レイヤーrole、可視/非表示、書込可能な役割。
3. **描画器による実行**：許可したActionだけEngineで適用。**1Actionの途中失敗では画素を半分だけ変更しない**。
4. **結果の記録**：受理済みActionとそのactor、状態差分を記録。`seq`は受理したActionだけで進める。不正提案はTrialログへ理由とattempt番号を保存し、Episodeの`seq`は消費しない。
5. **工程完了**：`advance_stage`提案時だけ、現在工程の退出判定を行う。**どの工程へ飛ぶかをAIに選ばせない**。

### 終了判定の原則

- ①②③は、その工程専用のroleで**受理済みの有効`stroke_v2`が1つ以上**かつ退出時に画素が残っていること。消去・ブラシ選択だけでは不合格。
- ④⑤は、許可レイヤーに実際に描くか、対象の指示が明示的に要求する場合だけ**レビューのみで退出**できる。
- ⑥は、実際に修正してもよく、**修正が必要なければレビューのみで退出**できる。わざと傷をつけて直したふりをさせない。
- ⑦はControllerだけが`finish`を実行し、次のActionを受理しない。

**有効な筆跡**は、許可roleレイヤーの少なくとも1画素を変更し、その工程の終了時にも視認可能な描画が残ること。重ね描きしたが完全に同じRGBAのままなら有効数に含めない。**一筆だけで工程が完成したと言えるわけではない**ので、画質と工程内成果は別のEvaluatorで判定する。

## 4. レビューだけで工程を終える場合

`skip_stage`というActionは**存在しない**。工程に入ったあと、`advance_stage({"review_reason": ...})`を提案し、Controllerが条件を確認する。

| 現在の工程 | 使用できる理由コード | 認める例 |
| --- | --- | --- |
| 基本色 | `monochrome_line_art_only` | 「白黒の線画だけを描く」と**明示**されている |
| 陰影・仕上げ | `flat_style_no_shading` | 「影を付けずフラットに塗る」と**明示**されている |
| 修正・確認 | `no_correction_needed` | 修正が不要だとモデルが判断して記録した |

最初の2つは**文章からの作品指定**に裏付けられている必要がある。単にモデルが「もう描きたくない」と選んでも許さない。

- 推論中にtarget画像を使うことはできないので、文章だけから得た`StyleIntent`（「線画のみ」「影なしフラット」等）をControllerが確認する。`StyleIntent`はprompt→設定を読む**別のversion付きの判定器**で作り、曖昧ならfalse（レビューは拒否）。人間に新たな必須入力は求めない。
- 学習中も**StyleIntentの入力に目標画像を入れない**。目標画像を使った採点で「このレビューは画質に寄与したか」を別途評価できるが、推論時には入手できない情報を審査に使わない。
- 修正不要という判断は完全に機械で証明できない。Controllerは理由を記録し、別の画質評価で「早すぎる完成」を失敗として数える。必要ならtargetなしの状態Criticを後続で導入してよいが、**未実装の判定器を実装済みと扱わない**。
- `review_reason`は**その工程で筆跡を一本も描かなかった場合にだけ**使用可能。描いた工程には通常の`advance_stage({})`を用いる。

## 5. 一回の工程移行は全部まとめて確定する

たとえば下書きを終えて本描きに進む場合：

```text
AIが advance_stage({}) を提案
    ↓
Controllerが「sketchレイヤーに有効な筆跡が残る」を確認
    ↓
一つの原子的な記録グループとして
  1. actor=model   advance_stage
  2. actor=system  exit_stage(sketch)
  3. actor=system  set_layer_visibility(必要な場合のみ)
  4. actor=system  enter_stage(final_drawing)
  5. actor=system  create_layer(role=final_drawing)
  6. 選択レイヤーを未選択にする
    ↓
完了したグループだけをReplay・外部公開する
```

最後の`refine→finish`では`advance_stage`、`exit_stage(refine)`、`finish`を一つの原子的記録グループとして確定。途中で落ちた場合、**部分的に工程が変わった完成Episodeを公開しない**。Recoveryで最後の確定グループへ戻す。グループIDと永続化方式はEpisode v2設計に必ず含める。

旧v1のJSONイベント`set_phase`や、旧「create_layerは常に最上層」の再生を**書き換えない**。上記はv2の契約。

## 6. レイヤーを重ねる順番

**描画する順番と、最後に表示する上下は違う**。

```text
（上） correction → highlight → final_drawing → shade
     → base_color → sketch → rough → 白背景（下）
```

- 本描きの線より後に色を塗っても、基本色は**本描きの下**へ合成される。
- `rough`は本描き入場時、`sketch`は基本色入場時に非表示。**削除しない**。Replayで表示状態も再現する。
- 最終画像のRendererはRoleのz順を使用。レイヤー作成順だけで重ねる実装は不合格。
- 同じroleの複数レイヤーは作成順でそのrole内に重ねる。非表示roleのレイヤーへ書き込めない。
- レイヤーの細かなアルファ合成・ブラシの数値は別のRenderer v2契約で定義する。**v1の画素Goldenは残す**。

## 7. 学習/生成時の失敗と予算

`DRAW_EPISODE.max_actions`を受理済みの**モデルAction提案数の上限**とする。Controllerの自動イベントは別で記録してよいが、**提案だけして拒否されたActionもattempt数として別途カウント**し、無限に不正提案を繰り返せないようにする。

- 受理済みActionと拒否attemptを混同しない。どちらも研究・評価時には分母へ含める。
- Action予算またはattempt上限に達したときは**未完成のDraft＋終了理由**。Controllerが残り工程を自動スキップ・描画・finishしてはならない。
- 画質が悪いのに工程を形だけ移動させる場合、Stageの合法性は通っても**画質評価は不合格**になり得る。Controllerは品質を保証しない。
- モデルが故意に一画素だけ描いて進む回避を防ぐには、工程別の画像品質報酬を併用し、筆跡の個数だけを高く評価しない。
- 学習時と推論時で同じ工程順序、制約、見える内部状態、ブラシ設定を用いること。**target画像から生成した中間画像やtarget依存Rewardを推論Policyへ入力する設計は禁止**。

## 8. 実装前に揃えるもの

- **現在できたもの**：機械可読な[Stage JSON](drawing_stage_workflow_v2.json)、この契約、[独立した静的契約テスト](../../tests/test_drawing_stage_workflow_design.py)。
- **まだ必要なもの**：Episode v2 JSON Schema（systemイベント・グループの原子性・stage_id・layer.role・表示フラグ）、v2 StageController/Action validator、v2 Rendererの決定的な筆跡Golden、StyleIntentのversion付き判定、Action予算の実装と回復試験、Policy/Reward/学習器の実体。
- **合格条件**：静的チェックに加え、実際のControllerが受け付けたAction列をReplayし、同じRGBA/レイヤー/表示状態へ戻ること。新しいControllerを実装していないのに静的テストだけで「動作確認済み」と報告しない。

**規範的なID・順序・権限は[Stage JSON](drawing_stage_workflow_v2.json)を優先**。初見の説明は[固定制作工程](fixed_drawing_workflow_v2.md)が正本。不一致を見つけたら黙って片方を解釈せず、仕様とテストを同じPRで修正する。
