# Image Drawer

Image Drawer は、**完成画像そのものだけでなく、画像を制作する手順を学習する**ことを目指す実験的な画像生成プロジェクトです。

基本ループ:

```text
状態 -> 制作Step -> 中間成果物 -> 評価 -> 選択 / 次のStep
```

大量の画像データをPart Bankとして再利用し、パーツ選択・合成・評価に加えて、最終的には「どの制作工程を採用するべきか」まで探索・学習対象にします。

## 文書は読者別に分かれています

詳しい文書一覧は [docs/README.md](docs/README.md) にあります。

### Image Drawerを使いたい

→ [利用者向け文書](docs/users/)

現在使える機能、起動方法、CLI/GUIの使い方を確認します。

### コードを読みたい・実装したい

→ [開発者向け文書](docs/developers/)  
→ [ソースコードガイド](docs/developers/source_code_guide.md)

architecture、DSL、Part Bank、学習/評価、コードレビューの入口を確認します。

### Kadokaが方針・実装順を確認したい

→ [Kadoka向け文書](docs/kadoka/)

プロジェクトのコンセプト、優先順位、実装計画を確認します。

### KadokaのCodexが実装作業をする

→ [KadokaのCodex向け作業規約](docs/kadoka-codex/)

Issue/PR運用、参照すべき正本、test/build要件、禁止事項を確認します。

## 現在の状態

現在は基盤実装段階です。Workflow Runtime、DSL、Part BankなどをIssue / Pull Request単位で実装しています。

完成した画像制作GUIはまだ提供していません。利用可能なentrypointは [利用者向けの「はじめに」](docs/users/getting_started.md) を参照してください。

## 言語

プロジェクト独自の説明文は日本語を正本とします。詳細は [言語運用方針](docs/developers/language_policy.md) を参照してください。

Apache-2.0や第三者ライセンスなど、原文自体に法的意味がある文書は公式原文を正本とします。
