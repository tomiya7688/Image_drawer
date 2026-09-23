# MVP範囲

## 1. 目標

最初のMVPでは、編集可能な手順型画像Workflowを実行し、中間状態まで観察できることを証明します。

高品質な画像生成は必須ではありません。

## 2. 必須component

### DSL parser

初期Workflow formatをparseします。

### Runtime

線形またはdependency-based Workflowを実行し、Artifactを追跡します。

### Step registry

operationを名前で登録し、schemaを公開します。

### Mock drawing step

決定的または単純な実装を用意します。

- SKETCH
- COLOR

最初はplaceholder画像生成や単純なtest image変換でも構いません。

### Mock evaluator

Selection logicをtestできるよう、再現可能なscoreを返します。

### SELECT

top-k selectionをsupportします。

### GUI

最低限次をsupportします。

- Workflow Step編集
- parameter編集
- run
- intermediate Artifact表示
- score表示
- DSL表示・編集
- validation error表示

## 3. Repository構成案

    Image_drawer/
    ├─ README.md
    ├─ docs/
    ├─ pyproject.toml
    ├─ image_drawer/
    │  ├─ dsl/
    │  │  ├─ parser.py
    │  │  ├─ runtime.py
    │  │  └─ types.py
    │  ├─ steps/
    │  │  ├─ base.py
    │  │  ├─ registry.py
    │  │  ├─ sketch.py
    │  │  ├─ color.py
    │  │  ├─ evaluate.py
    │  │  └─ select.py
    │  ├─ training/
    │  │  ├─ runner.py
    │  │  └─ config.py
    │  └─ gui/
    │     └─ app.py
    ├─ workflows/
    │  └─ basic.idraw
    └─ tests/

実装が進んだ場合は実際の構成を優先し、この図は概念上の目安として扱います。

## 4. 最初のEnd-to-End Workflow

    INPUT prompt: Text

    sketches = SKETCH(prompt, count=4)
    scores = EVALUATE(sketches, evaluator="mock")
    best = SELECT(sketches, scores=scores, top=1)
    colored = COLOR(best, prompt=prompt)

    OUTPUT colored

成功条件:

1. GUIがWorkflowをloadできる。
2. GUIからcountを変更できる。
3. DSLへ変更が反映される。
4. Runtimeが各Stepを実行する。
5. 中間candidateを表示できる。
6. Evaluator scoreを表示できる。
7. SELECTが採用candidateを示す。
8. final outputを確認できる。
9. run historyをlocalへ保存する。

## 5. MVP非目標

初期段階では必須ではありません。

- 高品質なimage foundation model
- distributed training
- reinforcement learning
- automatic workflow discovery
- arbitrary DSL programming
- multi-user / server architecture
- production deployment

MVPの主目的はアーキテクチャと実験loopの検証です。
