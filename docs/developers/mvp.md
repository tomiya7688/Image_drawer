> **対象読者:** 開発者・レビュアー  
> **目的:** 最初の動くImage Drawerの合否を、制作行動の再生と学習で判定する。  
> **状態:** v1目標仕様。現状実装済みの機能一覧ではない。

# MVP範囲：絵師の制作過程を学習する最小系

## 1. 最初に見せるべき動作

「3×2画素のキャンバスに筆で下書きを描き、**その上にレイヤーを追加し、色を選んでストロークを描き、消して修正する**」操作が、**人間の実演でもAIの推論でも同じ描画器**を通る。

```text
人間の入力 -> 保存されたActionEvent -> Replay -> 全中間状態の再現
                                        |
                                   学習Dataset
                                        |
                           DrawingPolicyが次Actionを予測
                                        |
                           Action実行 -> 状態観測 -> 次Action
```

高品質な作品はMVPの合格条件ではない。一方、画像の生成だけ、Partの配置だけ、固定の操作列を流すだけではMVPではない。

## 2. MVPの必須機能と受け入れ基準

| 層 | 必須内容 | 正式な合否 |
| --- | --- | --- |
| DrawingState | キャンバス、レイヤー、色・ブラシ選択、工程、終了状態 | [制作過程 v1](drawing_process_v1.md) |
| DrawingEngine | 固定256色、ストローク、透明レイヤー、消去、source-over | [受け入れ試験 A〜D](drawing_process_acceptance.md) |
| Recorder/Replay | 時系列イベント、出所、途中状態の保存と再生 | [受け入れ試験 B・D・E](drawing_process_acceptance.md) |
| Dataset/Policy | 人間のイベントを次行動教師にし、同じAction形式を生成 | [受け入れ試験 F](drawing_process_acceptance.md) |
| GUI | 色・ブラシ・レイヤーを選び筆を動かして、履歴／途中状態を見られる | [GUI仕様](gui.md) |
| Workflow連携 | Episodeと上位Stepを区別しArtifact/Trajectoryから参照できる | [アーキテクチャ](architecture.md)・[DSL](workflow_dsl.md) |

`DRAW_EPISODE`や`REPLAY_EPISODE`等、未実装Operationを完了済みと記載しない。これらを設計に基づき段階的に追加する。

## 3. 正式な完成条件

- **描画器のDone**：固定パレット、ブラシ、レイヤー、ストローク、消去、検証、完全再生が自動テストで成立。
- **記録のDone**：人間の実入力を、実際に何をしたかというイベント列として保存・再生可能。画像だけを保存していない。
- **学習器のDone**：実演データから未来を見ずに次Actionを学習し、checkpointを再読み込みし、モデルが閉ループでDrawingEngineを動かす。synthetic dataだけの検証は学習器のsmokeに留める。
- **MVP全体のDone**：上記全てが揃い、工程を追って検証できる。最終画像ができただけなら未完成。

## 4. v1で意図的に除外するもの

複数のブラシ、筆圧・傾き、アンチエイリアス、物理ブラシ、bucket fill、複雑なレイヤーブレンド、分散学習、強化学習、GPUによる高品質画像生成、GUIの自由配置Node Graph、PartBankの学習最適化、Workflowの自動構造探索は初期の必須条件ではない。

ただし、除外した機能の代わりに別の画像処理だけを実装しても、それを描画工程の学習成功とみなさない。

## 5. 既存コードとの関係

既存の`WorkflowRuntime`、`Artifact`、`Trajectory`、`Part Bank`、`COMPOSE`は利用できる部分を再利用する。**既存Mock SKETCH/COLOR画像変換やPart合成を、このMVPの描画学習の証拠に使わない**。新機能が未実装の間は「設計済み・実装未完了」と正確に表示する。

実装順とIssueの整理は [Kadoka向け実装計画](../kadoka/implementation_plan.md) を参照。
