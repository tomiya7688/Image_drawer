> **対象読者:** 実装を引き継ぐ開発者、Issueを作る人  
> **目的:** 画像だけで描いて学ぶ新設計（[実装計画](../kadoka/implementation_plan.md)）に照らして、既存コードの再利用範囲・直す契約・未実装を一覧にする。  
> **状態:** S0（Issue #65）の成果物。調査日 2026-11-04、`main` の内容に基づく。

# 既存コードと設計の整合監査

## 1. 結論

- **描画の主経路は未実装**です。`DrawingEngine`、`StageController`、Episode v2とそのReplayer、Policy、Trainer、画像Dataset登録、タグ・prompt抽出器は、コードに存在しません。
- Part Bank、`COMPOSE`、`SELECT`は動作しますが、**補助機能**です。これらが動くことは「AIが絵を描いて学習できる」ことを意味しません。
- Runtime、DSL、Artifact、Trajectory、CIは**再利用できます**。ただし、1回の描画が数百のActionを持つため、Trajectoryに保存する単位を決める必要があります（5節）。

## 2. 既存コードの判定

| 場所 | 役割 | 判定 | 理由・扱い |
| --- | --- | --- | --- |
| `core/models.py`, `serialization.py`, `lineage.py`, `identifiers.py` | `Artifact`、`Score`、`Trajectory`などの記録と系譜 | **再利用** | 描画結果の`Image`と`Episode`のArtifactにも使える。`Artifact.uri`に大きなEpisodeを置く |
| `core/score_set.py` | 評価器ごとの生得点の保存 | **再利用** | Rewardの成分を評価器の名前付きで保存する土台になる |
| `runtime/engine.py`, `validation.py` | Workflowの検証と実行、Trajectory記録 | **再利用** | `StepResult`で追加Artifactを返せる。`DRAW_EPISODE`は`Episode`と`Image`を返す上位Stepにする |
| `steps/base.py`, `registry.py` | Stepの契約とbackendの登録 | **再利用** | 出力型を`top`で変える`output_resolver`がある。描画Stepも同じ仕組みで足せる |
| `dsl/parser.py`, `serializer.py` | Workflowのテキスト表現 | **再利用** | 辞書リテラルを扱える。`DRAW_EPISODE`などの新Stepは登録だけで書ける |
| `evaluation/` (`mock`, `select_images`, `select_best`, `openclip_quality`) | 候補画像の評価と選択 | **補助** | `openclip_quality`は相対得点で、較正された品質値ではない。S3のReward成分の1つにできるが、そのままRewardとは呼ばない |
| `part_bank/` | 画像からのPart抽出・検索・SQLite保存 | **補助** | 描画学習の主経路にしない。Datasetの画像登録（S2）は別に作り、ハッシュ・重複排除の考え方だけを参考にする |
| `steps/compose.py`, `retrieve_parts.py` | Part配置によるPNG生成 | **補助** | 「線を引く行動」の代替にしない。`COMPOSE`に描画機能を足さない |
| `steps/mock.py` | モックStep一式 | **再利用（テスト用）** | RuntimeとDSLのgoldenに使う |
| `gui/` | `image-drawer-gui`の起動点 | **未実装** | placeholder。S4で作る |
| `search/` | 空のpackage | **未実装** | 旧M8（パラメータ探索）用。S3・S5の後に再定義する |
| `scripts/`, `.github/workflows/` | 検査、CI、配布の署名 | **再利用** | pytest、Ruff、build、wheel smoke、attestationを維持する |

## 3. 描画に関する設計と実装の対応

| 設計（正本） | 実装の状態 |
| --- | --- |
| 固定256色、色とブラシは独立（[ブラシv2](brush_presets_v2.md)） | 文書とJSON、静的テストのみ。**コードは未実装**（S1a） |
| ブラシプリセット12種、サイズ19段階・不透明度11段階・筆圧11段階（[段階表](brush_control_levels_v2.md)） | 同上 |
| 7工程の固定順（[固定工程](fixed_drawing_workflow_v2.md)、[Controller契約](stage_controller_contract_v2.md)） | 文書とJSONの静的テストのみ。**Controllerは未実装**（S1b） |
| `DrawingState`・`DrawingAction`・`DrawingEpisode`（[v1](drawing_process_v1.md)） | 文書とgolden fixtureのみ。テスト内のオラクルが再生する。**製品のReplayerは未実装**（S1c） |
| 画像のみのDataset、抽出器の切替（[画像のみ学習](image_only_learning_v1.md)） | **未実装**（S2） |
| 自己描画と報酬による更新（[Policy](drawing_policy_accuracy_v1.md)、[Reward](stage_reward_v2.md)） | 数値は未freeze。**Trainerは未実装**（S3） |
| 文章だけを入力とするtxt2img推論（S4 / #71） | **未実装**（S4） |

## 4. 人間の実演の模倣を前提にした記述

調査したコードに、人間の実演を前提にした実装はありません。文書に次の箇所が残っています。いずれも「必須でない将来の任意機能」として扱います。

| 場所 | 内容 | 扱い |
| --- | --- | --- |
| `brush_control_levels_v2.md` の実演ログ | 人間の実演値を別ログに保存する記述（保存方式の記述で、学習での利用は任意） | 任意機能。学習の必須条件にしない |
| `drawing_data_quality_v1.md` 7節 | 実演データの扱い（オプションのみ） | 維持。Doneに含めない |
| `drawing_process_v1.md`, `stage_controller_contract_v2.md` | `actor=human`、`origin=mixed`を保存できる | 維持。保存形式の互換のために残し、学習には使わない |
| `prior_art_and_research.md` | 先行研究としてのsupervised imitation | 調査記録として維持 |

## 5. 直す契約・決めること

ここは監査で見つけた**変更案**です。確定した契約ではありません。確定は各Issue（S1c、S3、S4）の設計で行います。

1. **Trajectoryの粒度**: 現在のTrajectoryはStep単位で、Artifactを1個ずつ記録します。描画は1回で数百のActionを持つため、Actionの履歴は`Episode` Artifactとして別ファイルに保存し、Trajectoryにはそのpathとハッシュ、系譜だけを残します（S1c、S4）。
2. **`Episode`と`Image`のArtifact型**: 現在の型は`Image`、`ImageSet`、`ScoreSet`などです。`Episode`を追加し、`DRAW_EPISODE`は`StepResult`で`Episode`と`Image`を返します。
3. **評価の契約**: 現在の評価器（`ScoreSet`、`select_*`）は、複数の候補画像を比べて得点を返します。S3では、目標画像と描画結果（または描画の途中の状態）を比べて、成分ごとのRewardを返す別のインターフェースが必要です。既存の候補選択型の契約とはシグネチャを分け、`ScoreSet`をそのまま代用しません。Rewardの履歴を`ScoreSet`で保存できるか、新しい記録型が必要かはS3で決めます。目標画像は評価器だけに渡し、Policyの観測に含めません（[画像のみ学習](image_only_learning_v1.md)）。
4. **v1とv2の併存**: v1の`set_phase`は画素Replayの互換のためだけに残します。v2の工程移行は`advance_stage`で記録します（[工程v2](fixed_drawing_workflow_v2.md)）。
5. **v2のEpisode schema**: v1のschemaのみ（`schemas/drawing_episode_v1.schema.json`）。v2のstroke形式とschemaはS1a・S1cの成果物として作ります。

## 6. 再利用してよい既存の確認

- v1のgolden（`tests/fixtures/drawing_process_v1/`）は、S1aの実装後も画素一致を維持します。
- 既存のRuntime・DSL・Part Bank・評価のテストは、そのまま維持します。

## 7. 実装の順序

[S1a（DrawingEngine, #66）](https://github.com/tomiya7688/Image_drawer/issues/66)→[S1b（StageController, #67）](https://github.com/tomiya7688/Image_drawer/issues/67)→[S1c（Replayerと受け入れテスト, #68）](https://github.com/tomiya7688/Image_drawer/issues/68)の順に進めます。S2（#69）は独立して進められます。S3（#70）はS1の完了後、S4（#71）とS5（#72）はその後です。
