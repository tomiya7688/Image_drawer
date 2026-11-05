> **対象読者:** 実装・レビューを担当する開発者とコーディングエージェント
>
> **目的:** 作業ごとに「どの設計書を読み、どこを実装し、何で確認するか」を1か所にまとめ、実装前の読み込みと実装中の手戻りを減らす。
>
> **状態:** 2026-11-05。設計書は文書、コードは各PRのマージ状況に従う。実装済みと設計のみを、5節で区別する。

# 実装マップ

[AGENTS.md](../../AGENTS.md)から、この文書を開いてください。作業に合う行だけを読みます。

## 1. 経路表：作業 → 読む文書 → 実装先

| 作業（Issue） | 読む設計書（上から順） | 実装先 | 受け入れテスト |
| --- | --- | --- | --- |
| 描画Engine（S1a #66） | [drawing_engine_v2.md](drawing_engine_v2.md)、[ブラシ](brush_presets_v2.md)、[段階表](brush_control_levels_v2.md) | `image_drawer/drawing/` | `tests/test_drawing_engine_v1.py`、`..._v2.py` |
| 工程Controller（S1b #67） | [design_stage_controller.md](design_stage_controller.md)、[契約](stage_controller_contract_v2.md)、[固定工程](fixed_drawing_workflow_v2.md) | `image_drawer/drawing/stage/` | 設計書の受け入れテスト一覧 |
| Episode・Replay（S1c #68） | [design_episode_replay.md](design_episode_replay.md)、[v2 schema](schemas/drawing_episode_v2.schema.json)、[v1仕様](drawing_process_v1.md)、[受け入れ試験](drawing_process_acceptance.md) | `image_drawer/drawing/episode*`、`replay*` | 同上 |
| Dataset登録（S2 #69） | [image_dataset.md](image_dataset.md)、[入力契約](training_input_v1.json)、[データ品質](drawing_data_quality_v1.md) | `image_drawer/dataset/` | `tests/test_dataset.py` |
| タグ・説明文の抽出（#78） | [prompt_extraction_v1.md](prompt_extraction_v1.md)、[統合方針](design_data_pipeline_integration.md) | `image_drawer/prompting/` | `tests/test_prompting_*.py` |
| 学習（S3 #70） | [design_trainer.md](design_trainer.md)、[Policy](drawing_policy_accuracy_v1.md)、[Reward](stage_reward_v2.md)、[画像のみ学習](image_only_learning_v1.md) | `image_drawer/training/` | 設計書の受け入れテスト一覧 |
| 推論・GUI（S4 #71） | [design_inference_gui.md](design_inference_gui.md)、[DSL](workflow_dsl.md)、[GUI](gui.md) | `image_drawer/steps/`、`gui/`、CLI | 同上 |
| 汎化評価（S5 #72） | [design_evaluation_protocol.md](design_evaluation_protocol.md)、[学習と評価](training_and_evaluation.md) | `image_drawer/evaluation/`（報告） | 同上 |
| Runtime・DSL・Artifact | [アーキテクチャ](architecture.md)、[DSL](workflow_dsl.md)、[ソースコードガイド](source_code_guide.md) | `image_drawer/core/`、`runtime/`、`dsl/`、`steps/` | `tests/test_runtime.py` ほか |
| 補助機能（Part Bank・COMPOSE・SELECT） | [part_bank.md](part_bank.md)、[score_sets.md](score_sets.md) | `part_bank/`、`steps/compose.py`、`evaluation/` | 既存テスト |
| 配布・CI・署名 | [verify_release.md](../users/verify_release.md)、[ローカル確認](local_checks.md) | `.github/workflows/`、`scripts/` | CI |
| 方針の確認 | [コンセプト](../kadoka/concept.md)、[実装計画](../kadoka/implementation_plan.md) | - | - |

設計書が無い作業は、実装せず**設計PRを先に出します**（[共同作業ルール 3.2](../kadoka-codex/collaboration_workflow.md)）。

## 2. パッケージの責務と依存方向

```text
core ─┬─ runtime ── steps ── dsl / gui / CLI
      ├─ drawing（Engine → stage → episode / replay）
      ├─ dataset ─ prompting（抽出器）
      └─ training（drawing と dataset を使う）
```

| パッケージ | 責務 | import してよい先 | import してはいけない先 |
| --- | --- | --- | --- |
| `core` | 記録型・シリアライズ・系譜 | 標準ライブラリ | その他すべて |
| `drawing` | 描画の状態・Action・Engine・工程・Episode・Replay | `core` | `dataset`、`training`、`steps`、`gui`、`prompting` |
| `dataset` | 画像の登録・正規化・抽出器の選択・manifest | `core`、`prompting` | `drawing`、`training` |
| `prompting` | Caption・Tagの抽出（モデル呼び出し） | 標準ライブラリ | `drawing`、`training`、`dataset` |
| `training` | Policy・Reward・Trainer | `core`、`drawing`、`dataset` | `steps`、`gui` |
| `steps`・`runtime`・`dsl`・`gui` | Workflowの実行と操作 | 下位すべて | - |

- **target隔離**: Policyの実装（`training/policy*`）は、`RewardTarget` と目標画像のパスを受け取らず、import もしません。契約テストで確認します（[design_trainer.md](design_trainer.md)）。
- 依存方向に反する import が必要になったら、設計PRで責務の置き場所を決め直します。

## 3. 固定契約（名前・署名・データ形式）

次の名前は、設計書どうしで共有する契約です。**変更は設計PRを先にマージしてから**行います。

| 契約 | 内容 | 定義する設計書 |
| --- | --- | --- |
| `DrawingAction(type, payload, actor, seq, time_ms)` | 描画の1操作。v2は `stroke_v2`・`select_color`・`select_brush`・`set_brush_params`・`create_layer`・`select_layer`・`set_layer_visibility`・`advance_stage` | [drawing_engine_v2.md](drawing_engine_v2.md) |
| 7工程 | `rough → sketch → final_drawing → base_color → shading → refine → finish`。`skip_stage` は存在しない | [固定工程](fixed_drawing_workflow_v2.md) |
| `StageController(engine, intent, budget)` | `propose(action) -> ProposalResult(accepted, events, reason, attempt)`、`stage_id`、`finished`、`episode()`、`attempts()`。目標画像を受け取らない | [design_stage_controller.md](design_stage_controller.md) |
| `replay(episode) -> ReplayResult(image, layers, state)` | Controllerを使わず、受理済みイベントを再適用する | [design_episode_replay.md](design_episode_replay.md) |
| `iter_samples(dataset_dir, split)` | `(PolicyInput(image_id, prompt), RewardTarget(image_id, image_path, image_sha256))` を返す | [image_dataset.md](image_dataset.md) |
| `DrawingPolicy`・`RewardEvaluator`・`Trainer` | Policyの観測に目標を含めない | [design_trainer.md](design_trainer.md) |
| Artifact型 `Episode` と Step `DRAW_EPISODE` | `StepResult` で `Episode` と `Image` を返す | [design_inference_gui.md](design_inference_gui.md) |
| 入力契約 | 学習に必要な外部入力は画像のみ。手入力の補助データを要求しない | [training_input_v1.json](training_input_v1.json) |

## 4. 実装順序と並行できる作業

```text
S1a Engine ─→ S1b Controller ─→ S1c Episode/Replay ─┐
                                                      ├─→ S3 Trainer ─→ S5 評価
S2 Dataset ←→ #78 抽出（統合）──────────────────────┘        └─→ S4 推論・GUI
```

- **並行できる**: S1a と S2、S2 と #78 の統合方針、設計書の作成。パッケージが別で、固定契約を共有しないため。
- **順番に進める**: S1a → S1b → S1c。S3 は S1b・S1c・S2 の後。S4・S5 は S3 の後。
- 各設計書の「実装手順」の PR 分割（1PR=1目的、500行以下）に従います。

## 5. 現在の状態

| 項目 | 状態 |
| --- | --- |
| 描画Engine（S1a） | 実装済み（PR #75）。純Pythonで、大きなキャンバスの速度は未計測 |
| 工程Controller・Episode v2・Replay（S1b・S1c） | S1bはPR #84、S1cはブランチ `claude/68-episode-replay` で実装済み（マージ前）。S4の永続化・クラッシュ注入は未実装 |
| Dataset登録（S2） | 実装済み（PR #74）。基本の抽出器は画素の統計だけ |
| 抽出パイプライン（#78） | 実装中（PR #78）。実モデルでの精度は未検証。S2との統合は設計済み（PR #81）で、コードは未実装 |
| 学習（S3） | 設計済み（PR #80）。**コードは未実装** |
| 推論・GUI（S4）・汎化評価（S5） | 設計済み（PR #81）。**コードは未実装** |
| 旧M6〜M14（GUI・実験ランナー・探索など） | 保留（Issue #62）。S4・S5・S3に再定義 |

「設計済み」「コードは未実装」の項目を、実装済みとして報告しないでください。

## 6. この文書の保守

- 経路表・固定契約・状態は、PRで実装や設計を変えた人が同じPRで更新します。
- 更新日を先頭の状態行に書き、5節の「実装済み」には対応するPR番号を残します。
