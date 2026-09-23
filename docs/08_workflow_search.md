# Workflow Search仕様

状態: 採用済み設計
日付: 2026-09-20

## 1. 目的

Workflow Searchは「どの画像制作工程を採用するべきか」を探索・学習する層です。

Image Drawerは次だけでなく、

- どのPartを選ぶか
- Partをどうcomposeするか

最終的に次も学習対象にします。

- 次にどのStepを実行するか
- どこで評価するか
- どこでbranchするか
- candidateをいくつ生成するか
- refinementをいつ止めるか
- どのmodel/operatorを使うか

初期段階ではreinforcement learningより先にexperiment/search problemとして扱います。

## 2. Search対象としてのWorkflow

WorkflowはStep nodeのgraphまたはordered DAGとして表現します。

例A:

    INPUT
      -> PLAN_LAYOUT
      -> RETRIEVE_PARTS
      -> SELECT_PARTS
      -> COMPOSE
      -> REFINE
      -> EVALUATE
      -> OUTPUT

例B:

    INPUT
      -> RETRIEVE_PARTS
      -> SELECT_PARTS
      -> COMPOSE
      -> EVALUATE
      -> REFINE
      -> EVALUATE
      -> OUTPUT

どちらも同じRuntimeで実行できる必要があります。

## 3. Search dimension

### Structure

- Step追加/削除
- compatible Stepの並べ替え
- EVALUATE位置変更
- SELECT挿入
- REFINE挿入
- candidate generationのbranch
- candidate merge

### Parameter

- candidate count
- top-k
- evaluator weights
- refinement count
- retrieval depth
- threshold
- temperature / seed等

### Backend choice

同じlogical Stepに複数backendを設定できます。

    COLOR.backend = controlnet_a
    COLOR.backend = adapter_b

将来的にbackend選択もWorkflow Search対象にします。

## 4. Constraint

Workflow Searchから任意の壊れたgraphを生成してはいけません。

各Stepはschemaを公開します。

    inputs
    outputs
    parameter schema
    capabilities

Search engineはstatic validationを通るWorkflowだけを実行します。

必須validation:

- 全inputが利用可能
- Artifact型がcompatible
- OUTPUTへ到達可能
- 現Runtimeでillegal cycleがない
- required parameterがvalid

## 5. Objective model

単一image scoreだけを最適化しません。

runごとに例えば次を持ちます。

    ObjectiveResult
      final_quality
      prompt_alignment
      human_preference
      diversity
      runtime_seconds
      compute_cost
      failure_rate
      editability
      intermediate_consistency

初期MVPではweighted aggregateを使って構いませんが、元componentをすべて保持します。

例:

    objective =
        0.40 * final_quality
      + 0.25 * prompt_alignment
      + 0.20 * human_or_preference_score
      + 0.15 * diversity
      - compute_penalty

weightはexperiment configurationでありhard-codeしません。

## 6. Experiment unit

1 Workflowを1 promptだけで評価しません。

    Experiment
      workflow
      prompt_set
      seeds
      evaluator_config
      dataset/index_version
      model_versions

output:

    ExperimentResult
      runs[]
      aggregate_metrics
      confidence/statistics
      failures[]

Workflow比較では可能な限り同じprompt setとseed policyを再利用します。

## 7. Trajectory要件

各runで保存:

    Trajectory
      workflow_id
      workflow_version
      prompt
      input_metadata
      executions[]
      artifacts[]
      evaluations[]
      selections[]
      final_outputs[]
      human_feedback[]
      timing
      errors

Stepごとの記録:

    StepExecution
      step_id
      step_type
      backend
      parameters
      input_artifact_ids
      output_artifact_ids
      seed
      start_time
      duration
      error

これがtraining/search logになります。

## 8. Search stage

### Stage 0: Manual Workflow

GUIから人間がWorkflowを作ります。

目的:

- baseline確立
- 有用operator発見
- logging検証

automatic searchは行いません。

### Stage 1: Parameter Search

Workflow structureを固定して次を探索します。

- candidate count
- top-k
- evaluator weights
- retrieval depth
- refinement strength/count

algorithm:

- grid search
- random search

最初に実装します。

### Stage 2: Template Search

複数のvalid Workflow templateを定義します。

    template_a:
      RETRIEVE -> SELECT -> COMPOSE -> REFINE

    template_b:
      PLAN -> RETRIEVE -> SELECT -> COMPOSE -> REFINE

    template_c:
      RETRIEVE -> SELECT -> COMPOSE -> EVAL -> REFINE

共通prompt setで比較します。

### Stage 3: Constrained Structural Search

安全なmutationだけを許可します。

- EVALUATE挿入
- SELECT挿入
- REFINE挿入
- candidate count変更
- compatible operator差し替え
- optional Step enable/disable

候補algorithm:

- random search
- evolutionary search
- mixed parameter向けBayesian optimization

### Stage 4: Workflow Policy

十分なTrajectoryが溜まった後、

    choose_workflow(prompt, context)

または

    choose_next_step(state)

を学習します。

ここからlearned orchestrationが始まります。

### Stage 5: RL / Sequential Policy

次を満たした場合のみRLを導入します。

- state/action定義が安定
- Evaluator挙動を理解済み
- reward hackingを監視できる
- search baselineでは不足

RLはMVP dependencyではありません。

## 9. Search space表現

内部表現:

    WorkflowSpec
      nodes[]
      edges[]
      inputs[]
      outputs[]
      metadata

    StepSpec
      id
      type
      backend
      parameters

DSLはWorkflowSpecのserializationです。

Search layerはraw DSL stringを直接書き換えず、WorkflowSpecをmutationします。

## 10. Mutation interface

将来interface:

    class WorkflowMutator:
        def propose(self, workflow, search_space, rng):
            ...

安全なmutation type:

    SetParameter
    ReplaceBackend
    InsertStep
    DisableStep
    ChangeCandidateCount
    ChangeTopK

すべてvalidationしてから実行します。

## 11. Baseline Workflow

repositoryに明示baselineを同梱します。

### Baseline A: retrieval-first

    INPUT prompt
    parts = RETRIEVE_PARTS(prompt, top=20)
    selected = SELECT_PARTS(parts, top=5)
    draft = COMPOSE(selected)
    final = REFINE(draft, prompt)
    OUTPUT final

### Baseline B: refine前に評価

    INPUT prompt
    parts = RETRIEVE_PARTS(prompt, top=20)
    selected = SELECT_PARTS(parts, top=5)
    drafts = COMPOSE(selected, count=4)
    scores = EVALUATE(drafts)
    draft = SELECT(drafts, scores=scores, top=1)
    final = REFINE(draft, prompt)
    OUTPUT final

### Baseline C: layout-first

    INPUT prompt
    layout = PLAN_LAYOUT(prompt)
    parts = RETRIEVE_PARTS(prompt, layout=layout, top=20)
    selected = SELECT_PARTS(parts, context=layout)
    draft = COMPOSE(layout, selected)
    final = REFINE(draft, prompt)
    OUTPUT final

これらを最初のWorkflow比較実験に使います。

## 12. Selection learning

Part selectionとWorkflow selectionを分離します。

### PartSelector

問い:

    どのcandidate Partを使うべきか?

input:

- PartSet
- current composition
- prompt
- layout
- evaluator features

### WorkflowSelector

問い:

    どの制作Workflowを使うべきか?

input:

- prompt
- requested style/task
- resource budget
- optional dataset/context features

分離することで学習・評価がしやすくなります。

## 13. Next-step policy

長期表現:

    policy(state) -> StepDecision

    StepDecision
      step_type
      backend
      parameters
      stop: bool

state候補:

- prompt embedding
- current Artifact summary
- current evaluator scores
- executed Step history
- remaining compute budget

初期versionでは不要ですが、将来学習できるようTrajectoryに必要dataを保持します。

## 14. Stopping policy

良いWorkflowは「何をするか」だけでなく「いつ止めるか」も決める必要があります。

候補:

- quality scoreがthreshold超過
- N回refineしても改善なし
- compute budget消費
- human acceptance
- selector confidenceが十分

初期は明示parameterとして扱い、将来学習対象にできます。

## 15. Experiment reproducibility

各experimentで記録:

- git commit
- workflow version
- dataset version
- Part Bank index version
- model/checkpoint identifiers
- evaluator versions
- random seeds
- search configuration

これらがないresultは信頼できるWorkflow比較として扱いません。

## 16. GUI要件

将来的にWorkflow Search panelを持たせます。

MVP control:

- workflow/template選択
- prompt set選択
- search parameter選択
- experiment開始
- run table表示
- aggregate score比較
- winning/losing trajectory確認

将来:

- mutation visualization
- Pareto front
- cost/quality plot
- prompt-specific workflow recommendation

## 17. Package構成案

    image_drawer/
      search/
        models.py
        objective.py
        experiment.py
        parameter_search.py
        template_search.py
        mutations.py
        optimizer.py
        results.py

      training/
        part_selector.py
        workflow_selector.py
        next_step_policy.py

MVP必須:

- models.py
- objective.py
- experiment.py
- parameter_search.py
- results.py

## 18. MVP受け入れ条件

Workflow Search v0完了条件:

1. 最低2種類のvalid Workflowを定義できる。
2. 同じprompt setで両方を実行できる。
3. 全Trajectoryを永続化する。
4. Workflow単位でEvaluator metricをaggregateする。
5. runtime/failure metricを記録する。
6. simple weighted objectiveでexperiment resultを比較できる。
7. parameter searchで最低1個のStep parameterを変更できる。
8. best configurationを保存metadataから再現できる。
9. GUIから比較resultを表示できる。

RLは不要です。

## 19. 最初の研究実験

問い:

    candidate generation + evaluation + selection は
    追加compute costに見合う改善を生むか?

比較:

    A:
      RETRIEVE -> SELECT -> COMPOSE -> REFINE

    B:
      RETRIEVE -> SELECT -> COMPOSE(x4)
      -> EVALUATE -> SELECT -> REFINE

測定:

- preference / final quality
- prompt alignment
- diversity
- runtime
- failure rate
- 一部promptでhuman A/B preference

アーキテクチャと中心仮説を同時に検証します。

## 20. 実装優先度

1. Trajectory persistence
2. WorkflowSpec data model
3. reusable Experiment runner
4. objective aggregation
5. parameter search
6. baseline workflow comparison
7. template search
8. learned workflow selection
9. structural / evolutionary search
10. 必要な場合のみRL

学習algorithmが変わってもシステム全体を使い続けられる設計にします。
