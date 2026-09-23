# コンセプト

## 1. 目的

Image Drawer は画像生成に対して別のアプローチを試します。

主な学習対象は完成画像の分布そのものではなく、**画像を制作するための手順**です。

制作手順の例:

    prompt
      -> sketch
      -> evaluate
      -> select
      -> line
      -> color
      -> evaluate
      -> correct
      -> final

画像自体も重要ですが、主に観測可能な状態、および評価器への入力として扱います。

## 2. 基本概念

システムは次のループを繰り返します。

    State -> Action -> New State -> Evaluation

Stateには次のような情報を含められます。

- prompt
- canvas / image
- line art
- mask
- palette
- layer
- 過去のscore
- 生成履歴

Actionには次のような操作を含められます。

- sketch
- draw line
- fill
- color
- shade
- erase
- refine
- branch
- evaluate
- select

## 3. 工程を分割する理由

手順を明示化すると、次の利点があります。

- どの中間工程で失敗したかを特定できる。
- 工程ごとに異なるEvaluatorを利用できる。
- 同じ中間状態から複数候補へ分岐できる。
- 個々の完成画像が異なっても、良い制作手順を保存・再利用できる。
- 将来的にStep順序そのものを探索・学習対象にできる。
- GUIから人間が制作工程を確認・変更できる。

## 4. DSLの基本原則

DSLは**汎用プログラミング言語にしない**ことを原則とします。

主目的は、GUIで操作するWorkflowを保存し、検証し、実行することです。

そのため優先するのは次の要素です。

1. 単純なStep定義
2. 明示的な入出力
3. 読みやすいparameter
4. 決定的なserialization
5. GUI <-> DSL の容易な変換
6. 実行前validation

複雑なロジックはDSL構文へ持ち込まず、Runtimeまたは再利用可能なStep実装側へ置きます。

## 5. 長期的な方向性

初期Workflowの順序は人間が定義します。

将来的には次の要素を探索・学習対象にします。

- どのStepを実行するか
- 候補をいくつ生成するか
- どこで評価するか
- どこで分岐するか
- いつ再試行するか
- どの中間成果物を再利用するか
- どの工程順がより良い結果を生むか
