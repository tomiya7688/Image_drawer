# アーキテクチャ

## 1. 全体構成

    +-------------------+
    |       GUI         |
    | Workflow editor   |
    | parameter/result  |
    +---------+---------+
              |
              v
    +-------------------+
    |   Workflow DSL    |
    | serialized graph  |
    +---------+---------+
              |
              v
    +-------------------+
    |      Runtime      |
    | validation        |
    | scheduling        |
    | state/history     |
    +----+---------+----+
         |         |
         v         v
    +---------+ +-----------+
    | Steps   | | Evaluator |
    | drawing | | / Critic  |
    +----+----+ +-----+-----+
         |            |
         +------+-----+
                v
          +-----------+
          | Selection |
          | / Trainer |
          +-----------+

## 2. GUI

GUIをWorkflow作成の主画面とします。

GUIから次を操作できるようにします。

- Stepの追加・削除
- Step順序の変更
- 入出力接続
- Step parameter編集
- 学習・探索parameter設定
- Workflow実行
- 中間成果物の確認
- 候補score比較
- Workflow保存・読込

DSLはGUI上のWorkflowを永続化する表現です。

## 3. Workflow Runtime

Runtimeの責務:

- Workflow定義のparse
- 型・入出力互換性のvalidation
- 実行graph構築
- Step呼び出し
- Artifact追跡
- score記録
- provenance / history記録
- candidate set管理
- error処理

Runtimeは特定のモデル実装へ依存しないようにします。

## 4. Step interface

Workflowから実行できる操作は共通interfaceを実装します。

    class Step:
        def run(self, inputs, params, context):
            ...

Stepが受け取るもの:

- 名前付きinput
- serialize可能なparameter
- execution context

Stepは1個以上の名前付きoutputを返します。

## 5. Artifact model

中間値は型付きArtifactとして扱います。

初期Artifact候補:

- Text
- Image
- ImageSet
- LineArt
- Mask
- Palette
- Score
- ScoreSet
- Metadata

将来候補:

- Layer
- StrokeSequence
- Region
- Embedding
- ActionSequence

可能な限り各Artifactへ次のprovenanceを保持します。

- 生成したStep
- 親Artifact
- 使用したmodel/config
- random seed
- score
- timestamp / run id

## 6. Model adapter

描画モデルとEvaluatorはadapter経由で接続します。

これによりWorkflow定義を具体的な実装から分離します。

    SKETCH(model="mock")
    SKETCH(model="local_model_a")
    SKETCH(model="remote_model_b")

Workflow上の操作名はSKETCHのまま、backendだけを差し替えます。

## 7. 実行履歴

すべてのrunを追跡可能にします。

最低限の履歴:

    Run
     |- workflow version
     |- global parameters
     |- step execution
     |   |- input artifact ids
     |   |- parameters
     |   |- output artifact ids
     |   |- duration
     |   |- errors
     |- evaluations
     |- selected candidates
     |- final outputs

この履歴を将来の学習・探索データに利用します。
