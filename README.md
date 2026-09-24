# Image Drawer

Image Drawer は、**完成画像そのものを直接学習するだけでなく、画像を制作する手順を学習する**ことを目的とした実験的な画像生成プロジェクトです。

基本ループは次の通りです。

    状態 -> 描画/制作Step -> 中間成果物 -> 評価 -> 選択 / 次のStep

画像生成を単一のブラックボックスとして扱わず、スケッチ、パーツ検索、構図、着色、評価、修正、選択などを編集可能なWorkflowとして分解します。

現在の設計では大量の画像データを再利用可能な Part Bank として扱い、どのパーツを選びどう組み合わせるかだけでなく、最終的には「どの制作工程を採用するべきか」まで学習・探索対象にします。

## プロジェクト目標

- 制作工程を可視化・編集可能にする。
- GUIから描画・学習Stepを追加、削除、並べ替えできるようにする。
- GUI上のWorkflowを小さなDSLとして保存・実行する。
- 大量画像データから再利用可能な視覚パーツを検索する。
- 中間成果物と最終画像を評価し、探索・選択・学習に利用する。
- 採用候補だけでなく不採用候補も含め、完全なTrajectoryを保存する。
- パーツ選択、構図判断、Workflow選択・順序を学習対象にする。
- Runtime、Evaluator、GUI、探索アルゴリズム、モデルを疎結合に保つ。
- 既存の画像生成、制御、検索、Reward Model、Preference Learning手法を利用できる箇所では積極的に再利用する。

## 言語方針

仕様・設計文書・README・docstring・コードコメントは**日本語を正本**とします。

識別子、API名、DSLキーワード、CLIコマンド、クラス名、論文名、固有名詞は必要に応じて英語のまま使用します。英訳が存在する場合は参考訳扱いです。

Apache-2.0 本文や第三者ライセンスなど、原文自体に法的意味がある文書は例外として公式原文を正本とします。

詳細は docs/11_language_policy.md を参照してください。

## 文書

- [コンセプト](docs/00_concept.md)
- [アーキテクチャ](docs/01_architecture.md)
- [Workflow DSL](docs/02_workflow_dsl.md)
- [GUI](docs/03_gui.md)
- [学習と評価](docs/04_training_and_evaluation.md)
- [MVP範囲](docs/05_mvp.md)
- [先行研究・技術調査](docs/06_prior_art_and_research.md)
- [Part Bank仕様](docs/07_part_bank.md)
- [Workflow Search仕様](docs/08_workflow_search.md)
- [実装計画](docs/09_implementation_plan.md)
- [Part Bank v0実装仕様](docs/10_part_bank_v0.md)
- [言語運用方針](docs/11_language_policy.md)

## 実装の入口

実装順は docs/09_implementation_plan.md を正本とします。

最初のEnd-to-End目標は次です。

    dataset
      -> Part Bank
      -> RETRIEVE_PARTS
      -> SELECT_PARTS
      -> COMPOSE
      -> EVALUATE
      -> Trajectory
      -> Workflow比較

最初のアーキテクチャ検証では高品質な生成モデルを必須にしません。Workflow、Artifact、provenance、実験基盤が安定するまでは mock / simple backend を使用します。

## 開発

M0のプロジェクト骨格は依存を小さく保ち、Python 3.11+ を前提とします。

    python -m pip install -e ".[dev]"
    python -m pytest
    python -m build
    python -m image_drawer --input hello
    image-drawer-gui --check

CIではPull Requestおよび main へのpushごとに、入力/出力テスト、package build、生成wheelを新規virtual environmentで実行する確認を行います。

実装はGitHub IssueとPull Requestを単位に進めます。
