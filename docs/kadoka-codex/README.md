# KadokaのCodex向け作業規約

**対象読者:** Kadokaの指示でこのrepositoryを実装するCodex  
**目的:** 技術仕様を重複させず、作業の進め方・確認事項・禁止事項を固定する。

## 1. 仕様の正本

このfolderは技術仕様の正本ではありません。

作業前に必要な正本を参照してください。

- 方針・優先順位: [Kadoka向け文書](../kadoka/)
- architecture / API / 実装contract: [開発者向け文書](../developers/)
- 利用者から見える挙動: [利用者向け文書](../users/)

仕様をこのfolderへコピーして独自に更新しないでください。

## 2. Issue / Pull Request運用

- 複数のエージェントが共同作業する手順(毎回のpull、PRの優先処理、CI確認、自己レビュー、マージ)は [共同作業ルールとワークフロー](collaboration_workflow.md) に従う。
- 原則として1 Issue = 1 Pull Requestで進める。
- 実装前に対象Issueと依存Issueを確認する。
- mainへ直接実装しない。作業branchを作る。
- PR本文は日本語で、何を変えたか・何を変えていないか・test結果を書く。
- 設計変更が必要になった場合は、実装へ埋め込まずIssue/文書へ戻して判断できる形にする。

## 3. 実装前に読む順番

**描画・Workflow・評価・学習に関する変更の場合は、Issueを読む前に目的を必ず確認する。**

1. [コンセプト](../kadoka/concept.md)（最上位の目的）
2. [制作過程v1](../developers/drawing_process_v1.md)（状態・行動の契約）
3. [描画受け入れ試験](../developers/drawing_process_acceptance.md)（合否）
4. [実装計画](../kadoka/implementation_plan.md)（優先順位）
5. 対象Issueと依存Issue
6. [ソースコードガイド](../developers/source_code_guide.md)、対象componentの開発者文書と関連test
7. 実装本体

**Part Bankだけの変更でも、それをプロジェクトの主目的と呼ばない。** 説明が古い文書と矛盾する場合は、推測実装せず設計Issueへ報告する。

## 4. 品質要件

変更後は対象に応じて最低限確認する。

```bash
python -m pytest
python -m build
```

entrypointやbuilt artifactへ影響する変更では、生成したwheelをfresh environmentで実行するCI要件も壊さない。

既存testを削除・弱体化して通す対応をしない。

## 5. 文書・comment

- 日本語を正本とする。
- 詳細は [言語運用方針](../developers/language_policy.md) に従う。
- identifier、API名、DSL keyword、外部固有名詞は無理に日本語化しない。

## 6. やってはいけないこと

- 技術仕様をCodex向け文書へ複製し、別の正本を作る。
- Issueにない大きなarchitecture変更を黙って入れる。
- 特定model/backendのためだけにRuntimeやDSLを密結合にする。
- provenance、version、seedなど再現性情報を理由なく落とす。
- test/buildが壊れた状態を完成として扱う。
- dataset、model checkpoint、secretをrepositoryへcommitする。
- 画像部品の配置や完成画像の変換を「人間の筆跡・制作過程の学習」と偽って実装完了にする。
- `COLOR` を意味確認なしにbucket fill・画像の塗りつぶしと同義にする。
- 固定256色パレットの番号対応を独断で変える。

## 7. 完了報告

PRでは最低限次を明示する。

- 対応Issue
- 変更したfile / component
- 主要な設計判断
- test / build結果
- 未対応事項や次Issueへ回す内容
