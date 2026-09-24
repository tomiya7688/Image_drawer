# Workflow DSL

## 1. 役割

DSLはGUIで操作したWorkflowを保存するためのserialized representationです。

意図的に小さく保ちます。

最初のversionで表現するのは次だけです。

- input
- 順序付きStep
- Step input
- parameter
- output

分岐やloopは、言語レベルの制御構文ではなく専用Stepとして表現する方針です。

## 2. 設計目標

- 人間が読める
- GUI状態から生成しやすい
- parseしやすい
- Git diffが安定する
- data dependencyが明示される
- validationが単純
- hidden stateを持たない

## 3. 構文案

    INPUT prompt: Text

    sketches = SKETCH(
      prompt,
      count=8
    )

    line_scores = EVALUATE(
      sketches,
      evaluator="line"
    )

    lines = SELECT(
      sketches,
      scores=line_scores,
      top=2
    )

    colored = COLOR(
      lines,
      prompt=prompt,
      count=4
    )

    final_scores = EVALUATE(
      colored,
      evaluator="final"
    )

    final = SELECT(
      colored,
      scores=final_scores,
      top=1
    )

    OUTPUT final

## 4. GUIとの対応

1つの代入をGUI上の1 Nodeへ対応させます。

例:

    colored = COLOR(lines, prompt=prompt, count=4)

GUI上では次の意味になります。

    Node type: COLOR
    Node id: colored

    Inputs:
      image <- lines
      prompt <- prompt

    Parameters:
      count = 4

    Output:
      colored

GUIはgraph状態からcanonical DSLを再生成できる必要があります。

## 5. 初期built-in operation

### INPUT
外部Workflow inputを宣言します。

### SKETCH
1個以上のsketch候補を生成します。

### LINE
画像をline artへ変換、またはline artを改善します。

### COLOR
画像/line artと任意のprompt/contextを利用して着色します。

### EVALUATE
1個以上のArtifactに対するscoreを返します。

### SELECT
scoreまたは他のselection strategyでArtifactを選びます。

### OUTPUT
Workflow outputを宣言します。

## 6. Candidate collection

Candidate setを第一級の値として扱います。

    sketches = SKETCH(prompt, count=8)
    scores = EVALUATE(sketches, evaluator="line")
    best = SELECT(sketches, scores=scores, top=2)

Runtime内部ではImageSet + ScoreSetのように表現できます。

## 7. Parameter

Step parameterは単純なserialize可能primitiveを基本とします。

- string
- integer
- float
- boolean
- list
- enum相当のstring

version 1では任意の実行可能expressionを許可しません。

## 8. Control flow

version 1では汎用的な if / for / while を入れません。

代わりにWorkflow operationそのものに意図を持たせます。

    candidates = SKETCH(prompt, count=8)
    best = SELECT(candidates, top=2)

将来候補:

    RETRY
    BRANCH
    MERGE
    STOP_IF
    REFINE

専用operationでは不足すると判明した時点で、言語レベルのcontrol flowを再検討します。

## 9. Validation

実行前に最低限次を検証します。

- 参照variableが存在する
- required inputが接続されている
- parameter名が正しい
- parameter型が正しい
- Artifact型に互換性がある
- output名が重複しない
- required workflow outputが存在する

可能な範囲でGUIはRunを有効化する前にerrorを表示します。
