# 開発者向け文書

**対象読者:** Image Drawerを実装・拡張・レビューする開発者  
**目的:** 内部設計、実装contract、コードの読み方を共有する。

## 描画実装前の必読（この順）

1. [最上位コンセプト](../kadoka/concept.md)
2. [制作過程の状態・行動・再生仕様 v1](drawing_process_v1.md)
3. [描画行動 v1 受け入れ試験](drawing_process_acceptance.md)（[JSON Schema](schemas/drawing_episode_v1.schema.json)・[再生fixture](../../tests/fixtures/drawing_process_v1/golden_episode.json)・[境界画素fixture](../../tests/fixtures/drawing_process_v1/golden_geometry.json)）
4. [人間の操作を正しく収集・学習データ化する契約](drawing_data_quality_v1.md)
5. [次Actionの学習と精度評価の契約](drawing_policy_accuracy_v1.md)
6. [アーキテクチャ](architecture.md)
7. [実装計画](../kadoka/implementation_plan.md)

**画像パーツの検索・合成は補助機能。人間の制作行動の学習と取り違えないこと。**

## コードレビューから入る

- [ソースコードガイド](source_code_guide.md)

## 設計

- [定番ブラシ12種の独自プリセット・高精度ペンサンプル仕様（v2）](brush_presets_v2.md)
- [ブラシ設定の機械可読な正本](brush_presets_v2.json)


- [アーキテクチャ](architecture.md)
- [Workflow DSL](workflow_dsl.md)
- [GUI実装仕様](gui.md)
- [学習と評価](training_and_evaluation.md)
- [評価結果の保存形式](score_sets.md)
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

描画・学習に関わる実装では、最初に上記の**描画仕様と受け入れ条件**を読んでください。その後 [ソースコードガイド](source_code_guide.md) と対象Issueに対応するtestを確認してください。

基本的なlocal確認:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

実装順・優先順位は [Kadoka向けの実装計画](../kadoka/implementation_plan.md) が正本です。
