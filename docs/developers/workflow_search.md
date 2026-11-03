> **対象読者:** Workflow Search・実験基盤の開発者
> **目的:** 画像制作の**描画行動Policy**を優先しつつ、その上位のWorkflow探索を後から追加するための研究仕様。
> **状態:** 将来の拡張仕様。現在の最優先MVPではない。

# Workflow Search仕様（DrawingPolicyの後段）

## 1. 優先順位

Image Drawerが学習する最小行動は、色・ブラシ・レイヤーの選択、筆跡、消去などの`DrawingAction`。**制作工程の順序は[固定制作工程v2](fixed_drawing_workflow_v2.md)でラフ→下書き→本描き→基本色→陰影→仕上げに固定**する。画像だけを教材とする[自己描画学習](image_only_learning_v1.md)が動いた後、各工程内の候補Actionや計算予算等を探索する。

- **DrawingPolicy**：キャンバスと履歴を見て次の具体的な行動を選ぶ。
- **WorkflowSearch**：どの区間を実行・評価・分岐・再試行するかを探索する。
- **PartSelector**：任意の補助素材を選ぶ。筆跡や工程を学習したことにはならない。

Part Bank中心の`RETRIEVE -> SELECT -> COMPOSE`は、**素材配置の比較実験**には使えるが、プロジェクトの最初のEnd-to-End学習例としては使わない。

## 2. 探索の前提条件

以下を満たさないままWorkflow Searchを「画像制作工程の学習」と呼ばない。

1. `DrawingEpisode`の行動時系列と、各時点の`DrawingState`を再生できる。
2. モデルが`DrawingAction`を逐次予測して`DrawingEngine`へ適用できる。
3. **画像のみの試行学習**で、Evaluatorの報酬→Policy更新→未見画像のclosed-loop描画改善を測定できる。人間の教師Actionは不要。
4. 出所・dataset split、`palette_id`、`brush_set_id`、`renderer_version`、モデルcheckpoint、seed、Workflow versionを追跡できる。

未達ならそれを未実装・未評価と記す。Partの選択精度や画像評価の改善を代用品にしない。

## 3. 探索する対象

### 3.1 Workflow構造

Workflowは`WorkflowSpec`のnodes/edges/inputs/outputsで表す。探索候補：

- 行動Policyを実行する`DRAW_EPISODE`／`REPLAY_EPISODE`の組み合わせ。
- `EVALUATE`の位置、候補`SELECT`、`REFINE`／修正セッションの起動。
- 評価のための複数候補生成と分岐・合流。
- **固定工程の順序は探索しない**。Policyが次の工程へ進むべきかの提案と、工程内の筆跡・道具・時間予算の改善を比較する。

**工程順序だけは固定**し、各段階の筆跡・道具・色・修正はPolicyが学習する。固定の筆跡Scriptを押し付けて学習不能にする構成は許さない。

### 3.2 Parameter／backend

- 候補数、評価器の重み、停止条件、許容する描画Action数、policy checkpoint。
- 同じ論理Stepに複数backendを接続可能にする。backendを変えても`DrawingAction`の意味論は変えない。
- Part Bankを使用する場合は検索深さ、採用素材数なども探索できるが**後続拡張**。

### 3.3 制約

- Step graphの型・接続・依存・cycle・parameterを実行前に検証する。
- Modelやworkflowから不正`DrawingAction`が出た場合は`DrawingEngine`でrejectし、失敗率として記録。
- 色番号を再割当てしたり、レイヤーを潰して過去Actionを消したりして比較指標を改善してはならない。

## 4. ExperimentとObjective

同じprompt set、学習・検証split、checkpoint version、seed policy、パレット・renderer versionで比較する。

```text
Experiment
    workflows[]
    prompt_set / episode_split
    policy_checkpoints
    seed_policy
    palette / brush / renderer_version
    evaluator_versions
  -> results:
       action_prediction_metrics
       action_validation_failures
       closed_loop_completion_rate
       intermediate_state_checks
       image_quality_and_human_preference
       time_and_compute_cost
```

**画像だけからのReward改善・固定制作工程の遵守・閉ループ制作の再現性・画像品質を別指標に保つ。** ある画像が高評価だからといって「人間の制作工程を学習した」と結論しない。aggregate weightを使う場合は個々のcomponentとweightを保存する。

## 5. 実装順（描画学習の後）

- Stage 0：同じDrawingEpisodeデータセットに対する手動Workflow・Policy baselineの比較。
- Stage 1：固定したWorkflowの`max_actions`や評価重みをgrid/random search。
- Stage 2：`EVALUATE`や`SELECT`位置を変えた少数のvalidation済みtemplate比較。
- Stage 3：制約付きWorkflow mutation、必要ならWorkflowSelector。
- Stage 4：次にどの上位工程を行うかのPolicy、必要ならRL。**ストローク行動Policyとの責務を分離**する。

元の`PartSelector`やtemplate searchのアイデアはStage 2以降の選択肢として残る。`DrawingPolicy`を実装しない理由にはしない。

## 6. 保存と再現性

`Trajectory`はWorkflow/Step単位の出力・時間・選択・評価の履歴。`DrawingEpisode`は**すべての実描画Actionの順序列**。Workflow/Searchの記録から必ず関連Episode URIへ辿れるようにする。必要なversion・seedが欠ける比較は再現不可と表記する。

Workflowの一般的なDSL・validationは [Workflow DSL](workflow_dsl.md) に従う。優先順の正本は [実装計画](../kadoka/implementation_plan.md)。Workflow Search自体の合否は実施するStageのIssueで別途定義し、制作過程の[受け入れ試験](drawing_process_acceptance.md)の合格を取り消さない。
