> **対象読者:** 開発者・設計レビュアー
> **目的:** Image Drawerの実装責務を分割し、Part合成と筆による制作・学習を混同させない。

# アーキテクチャ（目的と実装境界）

## 1. 最初に守る不変条件

Image Drawerが学ぶのは**人間の絵師の制作途中の判断・行動の時系列**である。最終画像の生成に成功することは十分条件ではない。

**具体的な型・色番号・描画計算の正本は [制作過程 v1](drawing_process_v1.md)**。同じ数値やスキーマをこの文書で二重定義しない。[受け入れ試験](drawing_process_acceptance.md)を通らない実装は、既存の画像処理が動いても完成ではない。

## 2. 目標システムの構成

```text
              人間の絵師
                  |
             入力操作の収集
                  v
          Recorder -> DrawingEpisode <- Replay
                  |             |           |
                  v             v           v
           学習Dataset       ActionHistory  DrawingEngine
                  |             ^           |
                  v             |           v
          DrawingPolicy --------+------> DrawingState
              （次の行動予測）        （各レイヤー／画素）
                  |                           |
                  +------ DrawingAction ------+
                                              |
                                              v
                                        Canvas preview
                                              |
                  WorkflowRuntime / Artifact / Trajectory
                  （上位Step・評価・系譜・実験結果）
```

**人間入力とモデル予測は同じDrawingAction契約に従う。** 画素に変更を加える権限を持つのは`DrawingEngine`。`DrawingPolicy`は行動を予測するだけで、レイヤー画像を直接上書きしない。

## 3. 基本コンポーネントの担当

| Component | 所有する責務 | やってはいけないこと |
| --- | --- | --- |
| `Palette` | 共通固定256色の`color_id -> RGB`とバージョン | エピソードごとに色番号を入れ替える |
| `DrawingEngine` | 状態、レイヤー、ブラシ、ストローク、消去、再生可能な状態遷移、描画エラー | 学習モデルを内部に持ち色・ストロークを勝手に決める |
| `Recorder` | 人間の操作をActionEventとして記録。時刻と出所の保持 | 最終PNGから人間が描いたことにする |
| `Replay` | 保存された行動から初期状態を再構築し、全イベントを順に適用 | 中間行動を省略して最終画像だけ読む |
| `DrawingPolicy` | Prompt + 状態 + 過去の行動を見て次のActionを予測 | 固定の決め打ちScriptだけを「学習済み」とする |
| `Dataset / Trainer` | 出所・権利・splitを管理し、人間の実演から次行動を学習 | 未来の状態を入力に混ぜる |
| `WorkflowRuntime` | 上位Stepの実行・検証・接続・Errorの記録 | Strokeごとの意味論を独自に作る |
| `Evaluator` | 予測した行動と最終結果を別々に評価 | 画像評価のみを工程学習の成果として扱う |
| `Part Bank` | 必要に応じ画像部品を参照・検索・配置 | Part配置を筆跡や人間の描画実演と称する |

## 4. 三つの異なる粒度を区別する

### 4.1 DrawingAction（小さな1行動）

「色を番号180に変更する」「筆でこの座標列をなぞる」「上レイヤーを選ぶ」「消す」など。学習器が予測する最小単位。状態は1Actionずつ変化する。

### 4.2 DrawingEpisode（一枚を制作する時系列）

1枚を制作する全ActionEvent、prompt、palette/brush/rendererバージョン、実演か推論かという出所を持つ。**時系列に再生して初めて制作工程を復元できる。** `DrawingEpisode`は学習Datasetの単位。

### 4.3 WorkflowStep（大きな制作区間）

`SKETCH`、`LINE`、`COLOR`、`CORRECT`、`EVALUATE`、`SELECT`など。Phase/Workflowの「何をするか」と、内部Actionの「実際に何を描いたか」は別階層。`COLOR` Stepの実行中に複数の色選択・筆跡・消去が起きてよい。描画工程を別モデルやバックエンドに切り替えても、Action契約・再生意味論は変えない。

```text
Workflow: INPUT -> SKETCH -> EVALUATE -> COLOR -> OUTPUT
                         |                 |
                         +------ DrawingEpisode（継続）
                                [create_layer, select_brush, select_color,
                                 stroke, ..., create_layer, select_layer,
                                 select_color, stroke, erase, finish]
```

上記Workflow図は**上位区間の例で、現在動くDSLの証明ではない**。SKETCH/COLORのAdapterを実装したときも、工程を跨いで同じDrawingStateとActionHistoryを維持する。`finish`は制作全体の最後に1回。Step終了とEpisode終了は別。

## 5. 既存コードとの接続（現状と目標の区別）

現在のリポジトリには次が存在する。

- `image_drawer/runtime/engine.py`：`WorkflowRuntime.execute()`がWorkflowSpecを実行し、`Trajectory`を生成。
- `image_drawer/core/models.py`：`Artifact`・`Score`・`Trajectory`・Part関連モデル。
- `image_drawer/steps/base.py`：`Step.run` は主出力`Artifact`または追加Artifact付き`StepResult`を返す。
- `image_drawer/steps/retrieve_parts.py`・`compose.py`：既存Part Bankから素材を検索・合成する補助機能。
- `image_drawer/dsl/`：WorkflowSpecをparse/serializeする仕組み。

**ただし、現在の主要コードにはここで規定したDrawingEngine、DrawingEpisode、DrawingPolicyが実装されているとは限らない。** この仕様PRは機能実装を行ったことを意味しない。

追加の目標インターフェース（型の詳細はv1正本、下記は責務のイメージ）：

```python
class DrawingEngine:
    def initial_state(self, canvas) -> DrawingState: ...
    def apply(self, state: DrawingState, action: DrawingAction) -> DrawingState: ...
    def render(self, state: DrawingState) -> RGBAImage: ...

class DrawingRecorder:
    def record(self, event: ActionEvent) -> None: ...
    def save(self) -> DrawingEpisode: ...

class DrawingPolicy:
    def next_action(self, prompt, state, history) -> DrawingAction: ...
```

新しい`DrawingEpisode`は型付きArtifactのURIとバージョンでWorkflow Runtimeへ紐付ける。`Trajectory.executions`は**上位Stepの実行履歴**として残す。何千回ものstrokeを1個の`StepExecution`へ文字列として押し込まず、`DrawingEpisode.events`に順序通り保存する。

## 6. 学習ループの契約

- **収集**：人間が実行したActionとStateを再生可能な形で保存する。
- **Training**：ActionEventの直前までの状態から次のActionを予測する。色は固定256クラス。道具・工程・線の操作・終了も対象。
- **Inference**：予測したActionを`DrawingEngine`へ適用し、新しいStateを次の入力にする。人間の未来行動や最終画像をpeekしない。
- **Evaluation**：教師あり予測、閉ループで描かれた過程、最終画像の品質を区別して記録する。

単純なReplayはS1〜S2の検証であり、モデルの学習成功ではない。実データでの`DrawingPolicy`の学習・推論は別の完了条件とする。

## 7. 設計優先度と拡張

1. [コンセプト](../kadoka/concept.md)・[v1契約](drawing_process_v1.md)・[受け入れ試験](drawing_process_acceptance.md)を正とする。
2. まずDrawingEngine → Recorder/Replay → 次行動Policy → Workflow/GUI統合の順に成立させる。
3. Part Bank、Workflow Search、品質評価器は**拡張**として再利用してよいが、上位の目的を置き換えない。
4. 変更したくなった場合はAction/Stateのschema version・migration・golden testsと共に設計レビューする。

現在のソースの入口は [source_code_guide.md](source_code_guide.md)。実装順は [Kadoka向け実装計画](../kadoka/implementation_plan.md)。
