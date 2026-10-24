---
name: image-drawer-context
description: Image Drawerで実装やテストの場所を探し直すとき、作業を引き継ぐとき、保全ブランチを統合するときに、必要な正本だけへ案内する。
---

# Image Drawerの必要なコンテキストを選ぶ

ファイルが分かる小さな修正では対象を直接読み、このスキルの参照資料を一括で読み込まないでください。

## 作業別の参照先

| 必要な情報 | 開く資料 |
| --- | --- |
| 実装・仕様・テストの最初の確認先 | [作業別の入口](references/routing.md)の該当行 |
| 開発コマンドやWindowsでの確認条件 | [繰り返す操作](references/operations.md)の該当節 |
| 過去のGUI・学習実装とmainの違い | [保全実装の統合](references/integration.md) |
| 別エージェントへ渡す短い記録 | [引き継ぎ](references/handoff.md) |

## 読み進め方

1. 依頼から目的・対象・完了条件を取り出し、対応する参照先を一つ選びます。
2. 候補の実装、呼び出し元、関連テストで担当と現状を確かめます。資料と異なる場合は現在のソースを確認し、入口を修正します。
3. 判断に必要な疑問が残るときだけ、関係する資料へ広げます。材料がそろったら探索を止めます。
4. 実行した確認と未確認の範囲を短く残します。過去ログ全体や大きな出力は貼り直しません。

## 増やしすぎないために

手順の正本は一か所に置き、AGENTS・Claude・Copilot用の同内容の版を作りません。
このファイルに作業履歴や仕様本文を追加せず、条件付きの詳細は該当する参照資料へ置きます。
更新後はリポジトリルートで `python scripts/check_agent_context.py` を実行できます。

## 参考にした手法

[ai-context-reducer](https://github.com/tomiya7688/ai-context-reducer)の「機械で先に絞る・繰り返す手順を再利用する・関係する部分だけ読む」を、このリポジトリの入口と引き継ぎへ適用しています。原文の作業規則を導入するものではありません。

具体的な参照元: [読む情報を絞る3つの方法](https://github.com/tomiya7688/ai-context-reducer/blob/main/docs/jp/手法/AIに読む情報を絞る3つの方法.md)、[作業ごとの入口](https://github.com/tomiya7688/ai-context-reducer/blob/main/docs/jp/手法/作業の種類ごとに、AIに最初に読む場所を伝える.md)。
