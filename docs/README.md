# 文書案内

Image Drawerの文書は、**誰が・何のために読むか**で分けています。

## 利用者向け

[docs/users/](users/)

Image Drawerを使う人向けです。

- 現在使える機能
- install / 起動
- CLI / GUIの使い方
- Workflowの利用方法
- 利用時の注意

設計や内部実装を理解する必要がない利用者は、ここだけ読めばよい構成を目指します。

## 開発者向け

[docs/developers/](developers/)

Image Drawerを実装・拡張・レビューする人向けです。

- architecture
- DSL
- GUI実装仕様
- 学習・評価仕様
- Part Bank
- Workflow Search
- 先行研究
- 実装詳細
- source code guide
- 開発時の言語方針

## Kadoka向け

[docs/kadoka/](kadoka/)

プロジェクトオーナーとして、方針・優先順位・ロードマップを確認するための文書です。

- プロジェクトのコンセプト
- 実装順・ロードマップ
- 今後の意思決定記録

## KadokaのCodex向け

[docs/kadoka-codex/](kadoka-codex/)

KadokaがCodexへ実装を任せる際の作業規約です。

ここには技術仕様を複製しません。Codexは必要な仕様を `developers/` と `kadoka/` の正本文書から読みます。

## 文書追加ルール

新しい文書を追加するときは、先に次を決めます。

1. 誰が読む文書か
2. 何の判断・作業のための文書か
3. 既存文書と正本が重複しないか

読者が決まらない文書を `docs/` 直下へ追加しないでください。
