# 開発者向け文書

**対象読者:** Image Drawerを実装・拡張・レビューする開発者  
**目的:** 内部設計、実装contract、コードの読み方を共有する。

## 初めて設計を読む場合

**まずは [Image Drawer 設計ガイド](start_here.md)**。txt2imgの仕組み、何が決まっているか、ブラシ・筆圧の例、何が未実装かを説明します。 **学習素材は画像のみ**。内部テキストの処理は[学習入力契約](training_input_v1.json)を参照してください。

**固定制作工程を実装する場合**：[7工程の説明](fixed_drawing_workflow_v2.md) → [StageControllerの詳細契約](stage_controller_contract_v2.md) → [各工程の画像評価](stage_reward_v2.md) の順に読みます。

**学習前のCaption/Taggerの実装者**：[CaptionとTagの独立生成・分類/整形・自由登録とN段モデル切替](prompt_extraction_v1.md) → [画像学習仕様](image_only_learning_v1.md) → [ユーザー登録レジストリ](../../image_drawer/prompting/registry.py) の順に読みます。

**特にブラシを実装する場合**：[サイズ・不透明度・筆圧の段階表とパターン](brush_control_levels_v2.md) → [12種のブラシと高精度操作](brush_presets_v2.md) の順に読みます。

## 描画実装前の必読（この順）

1. [最上位コンセプト](../kadoka/concept.md)
2. **[画像だけで描いて学ぶ方式](image_only_learning_v1.md)**
3. **[固定制作工程：ラフ→下書き→本描き](fixed_drawing_workflow_v2.md)**
4. [制作過程の状態・行動・再生仕様 v1（旧画素再生）](drawing_process_v1.md)
5. [描画行動 v1 受け入れ試験とv2学習条件](drawing_process_acceptance.md)（[JSON Schema](schemas/drawing_episode_v1.schema.json)・[再生fixture](../../tests/fixtures/drawing_process_v1/golden_episode.json)・[境界画素fixture](../../tests/fixtures/drawing_process_v1/golden_geometry.json)）
6. [画像の登録とTag/Caption抽出の品質仕様](drawing_data_quality_v1.md)
7. [自己描画Policyの学習と精度評価](drawing_policy_accuracy_v1.md)
8. [アーキテクチャ](architecture.md)
9. [実装計画](../kadoka/implementation_plan.md)

**画像パーツの検索・合成は補助機能。画像だけを教材にAI自身が試し描きして学ぶことと取り違えないこと。**

## コードレビューから入る

- [ソースコードガイド](source_code_guide.md)
- [品質と導線の見直し](design_quality_review.md)：現状の手抜き箇所、学習・推論ユーザーの導線、直す順序（最新）。
- [既存コードと設計の整合監査](design_audit.md)：再利用できる機能、直す契約、未実装の一覧（S0当時の記録）。

## 設計

- [固定制作工程（ラフ→下書き→本描き）の仕様](fixed_drawing_workflow_v2.md)
- [画像だけで描いて学ぶ仕様](image_only_learning_v1.md)
- [画像のみのDataset（実装）](image_dataset.md)：登録、抽出器の契約、target隔離。
- [定番ブラシ12種の独自プリセット・高精度ペンサンプル仕様（v2）](brush_presets_v2.md)
- [ブラシ設定の機械可読な正本](brush_presets_v2.json)
- [固定段階表・筆圧パターンの機械可読な正本](brush_control_levels_v2.json)


- [アーキテクチャ](architecture.md)
- [Workflow DSL](workflow_dsl.md)
- [GUI実装仕様](gui.md)
- [学習と評価](training_and_evaluation.md)
- [評価結果の保存形式](score_sets.md)
- [汎化評価プロトコル（S5）](design_evaluation_protocol.md)と[報告書スキーマ](schemas/eval_report_v1.schema.json)
- [MVP範囲](mvp.md)

## AI / 探索

- [先行研究（2026-09時点の参考資料・非正本）](prior_art_and_research.md)
- [Workflow Search仕様](workflow_search.md)

## Part Bank

- [Part Bank仕様](part_bank.md)
- [Part Bank v0実装仕様](part_bank_v0.md)

## 開発運用

- [言語運用方針](language_policy.md)

プロジェクト全体の方向性や実装優先順位は [Kadoka向け文書](../kadoka/) を参照してください。

## 開発開始

描画・学習に関わる実装では、最初に**画像-only学習・固定制作工程と受け入れ条件**を読んでください。その後 [ソースコードガイド](source_code_guide.md) と対象Issueに対応するtestを確認してください。

基本的なlocal確認:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

実装順・優先順位は [Kadoka向けの実装計画](../kadoka/implementation_plan.md) が正本です。
