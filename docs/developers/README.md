# 開発者向け文書

**対象読者:** Image Drawerを実装・拡張・レビューする開発者  
**目的:** 内部設計、実装contract、コードの読み方を共有する。

## コードレビューから入る

- [ソースコードガイド](source_code_guide.md)

## 設計

- [アーキテクチャ](architecture.md)
- [Workflow DSL](workflow_dsl.md)
- [GUI実装仕様](gui.md)
- [学習と評価](training_and_evaluation.md)
- [MVP範囲](mvp.md)

## AI / 探索

- [先行研究と技術方針](prior_art_and_research.md)
- [Workflow Search仕様](workflow_search.md)

## Part Bank

- [Part Bank仕様](part_bank.md)
- [Part Bank v0実装仕様](part_bank_v0.md)

## 開発運用

- [言語運用方針](language_policy.md)

プロジェクト全体の方向性や実装優先順位は [Kadoka向け文書](../kadoka/) を参照してください。

## 開発開始

最初に [ソースコードガイド](source_code_guide.md) を読み、対象Issueに対応する仕様文書とtestを確認してください。

基本的なlocal確認:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

実装順・優先順位は [Kadoka向けの実装計画](../kadoka/implementation_plan.md) が正本です。
