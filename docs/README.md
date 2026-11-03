# 文書案内

Image Drawerの文書は、**誰が・何のために読むか**で分けています。

**初めて読む方は [Image Drawer 設計ガイド](developers/start_here.md) から始めてください。** 学習素材の外部入力は画像だけです。内部で生成するテキスト条件は[学習入力契約](developers/training_input_v1.json)にまとめています。 制作工程、定番ブラシ、段階表、学習と未実装範囲を平易な言葉で説明しています。

**実装者向けの詳細な読み順**：[目的](kadoka/concept.md) → **[画像だけで描いて学ぶ](developers/image_only_learning_v1.md)** → **[固定のラフ→下書き→本描き工程](developers/fixed_drawing_workflow_v2.md)** → [受け入れ試験](developers/drawing_process_acceptance.md) → [画像データと自動抽出器](developers/drawing_data_quality_v1.md) → [学習器と精度評価](developers/drawing_policy_accuracy_v1.md) → [旧v1画素再生](developers/drawing_process_v1.md) → [実装計画](kadoka/implementation_plan.md)。**既存のPart Bank中心の説明を最優先仕様として読まないでください。**

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

ブラシは **独自の12プリセット** から選び、サイズ19段階・不透明度11段階・筆圧11段階と筆圧変化5パターンを扱います。初見の方には[段階表と具体例](developers/brush_control_levels_v2.md)、実装者には[ブラシv2仕様](developers/brush_presets_v2.md)を用意しています。従来の単一ブラシは再生契約の検証用であり、イラスト全般向けの完成品ではありません。


[docs/developers/](developers/)

Image Drawerを実装・拡張・レビューする人向けです。

- 制作過程の状態・行動・再生契約とその受け入れテスト
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
