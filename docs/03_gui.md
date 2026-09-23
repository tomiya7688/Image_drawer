# GUI

## 1. 目的

GUIは、描画Workflowを構築・実行・観察するための実験workbenchです。

DSLはGUIの下にある実装・永続化層であり、ユーザーが必ず手書きする主interfaceではありません。

## 2. 初期layout

推奨layout:

    +----------------+---------------------------+
    | Parameters     | Workflow                  |
    |                |                           |
    | run settings   | [SKETCH]                  |
    | model settings |    |                      |
    | train settings | [EVALUATE]                |
    |                |    |                      |
    |                | [SELECT]                  |
    |                |    |                      |
    |                | [COLOR]                   |
    +----------------+---------------------------+
    | Results / Intermediate artifacts           |
    | candidate thumbnails / scores / history    |
    +--------------------------------------------+

## 3. Workflow editor

version 1で必要な操作:

- Step追加
- Step削除
- Step並べ替え
- outputからinputへの接続
- parameter編集
- Step有効/無効
- 生成DSLの確認
- Workflowのload/save

最初のMVPでは自由配置Node Editorは必須ではありません。

縦方向のStep list + 接続selectorの方が実装・debugしやすいため、最初はこちらを優先します。

## 4. Parameter editor

parameter UIは各Stepのschemaから生成します。

例:

    SKETCH

    model       [mock        v]
    count       [8            ]
    temperature [0.8          ]
    seed        [random       ]

各Stepは最低限次のmetadataを公開します。

    name
    description
    input schema
    output schema
    parameter schema
    default values

Stepごとに専用formを手書きせず、schema-driven GUIとします。

## 5. Result inspector

各runについて次を確認できるようにします。

- 中間画像
- candidate group
- score
- selected / rejected candidate
- Step parameter
- 実行履歴
- final output

どのcandidateがなぜ選ばれたかを視覚的に追跡できることを重視します。

## 6. Evaluator view

scoreは1個のscalarに限定しません。

例:

    overall        0.84
    composition    0.91
    line_quality   0.79
    color          0.87
    prompt_match   0.82

GUIはaggregate scoreとcomponent scoreの両方を表示できるようにします。

## 7. Training view

初期training panelでは、選択中の学習・探索戦略に必要なparameterだけを表示します。

例:

- candidate count
- top-k
- epochs / iterations
- learning rate
- batch size
- evaluator weights
- checkpoint interval

現在の戦略で使わないparameterを大量に表示しないことを原則とします。

## 8. DSL editor / view

GUIに生成DSLのtext viewを持たせます。

version 1で必要な機能:

- 表示・編集
- parse
- validation error表示
- GUIへの反映

canonical formattingにより、GUI生成WorkflowのGit diffを安定させます。
