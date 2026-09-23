# 実装計画

状態: 採用済み
日付: 2026-09-20

## 1. 目標

現在の設計を実装順へ落とします。

最初の目標は高品質画像ではなく、End-to-Endで動く研究workbenchです。

最初の有用loop:

    dataset
      -> Part Bank
      -> GUI/DSL Workflow
      -> retrieval
      -> selection
      -> composition
      -> evaluation
      -> trajectory storage
      -> workflow comparison

## 2. M0: Project skeleton

作成:

    image_drawer/
      core/
      dsl/
      runtime/
      part_bank/
      steps/
      evaluation/
      search/
      gui/

    tests/
    workflows/
    data/

deliverable:

- Python package
- config loading
- logging
- test setup
- CLI entrypoint
- GUI entrypoint

## 3. M1: Core model

最初に実装:

    Artifact
    Score
    SourceImage
    Part
    PartSet
    Layout
    Composition
    StepSpec
    WorkflowSpec
    StepExecution
    Trajectory

規則:

- stable string IDを使う。
- 全recordをserialize可能にする。
- model/backend/version metadataを明示する。
- 大きなimage byteをJSONへ直接入れずURI/path参照にする。

受け入れ条件:

- serialization round-trip testが通る。
- minor schema追加後も可能な範囲で旧recordをloadできる。

## 4. M2: RuntimeとDSL

実装:

- Step base interface
- Step registry
- Workflow validation
- DSL parser
- canonical DSL serializer
- Runtime execution
- Artifact registry
- Trajectory logger

初期Step:

- INPUT
- RETRIEVE_PARTS
- SELECT_PARTS
- COMPOSE
- EVALUATE
- SELECT
- OUTPUT

REFINEは最初はpass-through/mockでも構いません。

受け入れ条件:

- GUIなしでもWorkflow fileをCLIから実行できる。
- 全Stepのinput/outputが記録される。

## 5. M3: Part Bank v0

実装:

- filesystem ingest
- SourceImage record
- simple extractor 1種類
- Part record
- embedding backend 1種類
- vector index backend 1種類
- text/category retrieval

最初のextractorはdatasetに適した最も単純な方法を使います。

完璧なsegmentation model待ちでM3を止めません。

受け入れ条件:

- 小さいdevelopment subsetをingestできる。
- indexをbuildできる。
- text/category queryからtop-k Partをretrieveできる。
- source provenanceを保持する。

## 6. M4: Simple composition

実装:

- PartPlacement
- Layout
- COMPOSE renderer

初期renderer support:

- translation
- scale
- rotation
- z-order
- alpha/mask

受け入れ条件:

- selected Partからdeterministic draft imageを作れる。
- Compositionをrasterized imageとは別Artifactとして保持する。

## 7. M5: Evaluator subsystem

interface:

    class Evaluator:
        def evaluate(self, artifacts, context) -> ScoreSet:
            ...

最低1種類のbaseline Evaluatorを実装します。

support:

- evaluator name/version
- multiple component scores
- aggregate score
- batch evaluation

受け入れ条件:

- EVALUATEでImageSetをscoreできる。
- SELECTでScoreSetからtop-kを選べる。
- raw evaluator outputを保持する。

## 8. M6: GUI v0

完成品Editorではなくworkbenchを作ります。

必須:

- Workflow load/save
- ordered Step list
- Step add/delete/reorder
- Step schema由来parameter editor
- DSL text view
- Run button
- intermediate Artifact viewer
- candidate thumbnails
- score table
- selected/rejected表示
- run history list

free-form node graphより縦型editorを先に実装します。

受け入れ条件:

- GUI parameter変更がserialized DSLへ反映される。
- DSL loadでGUI stateを再構築できる。
- Part retrievalを含むWorkflowをGUIから実行できる。

## 9. M7: Experiment runner

実装:

- prompt-set loader
- repeated run execution
- seed policy
- workflow comparison
- aggregate metrics
- results export

受け入れ条件:

- 同じprompt setで最低2 Workflowを比較できる。
- 全Trajectoryとexperiment metadataを保存する。

## 10. M8: Parameter search

実装:

- grid search
- random search

初期search parameter:

- retrieval top-k
- selected top-k
- composed candidate count
- evaluator weights

受け入れ条件:

- 保存result metadataからbest configurationを再現できる。

## 11. M9: Real model integration

Architectureが動いた後に行います。

候補:

- pretrained image editing/refinement model
- ControlNet / T2I-Adapter系conditioned generation
- preference evaluator
- perceptual / structure evaluator

各integrationは既存Step/Evaluator interfaceの背後にadapterとして実装します。

1 modelのためだけにDSL syntaxを変更しません。

## 12. M10: Learning

保存Trajectoryを使います。

最初の学習target:

1. PartSelector
2. prompt -> WorkflowSelector
3. refinement decision / gating

最初からEnd-to-End RLは行いません。

推奨順:

    logged data
    -> supervised ranking
    -> pairwise preference learning
    -> workflow selector
    -> constrained structural search
    -> 必要な場合のみRL

## 13. Test strategy

### Unit test

- DSL parsing
- serialization
- type validation
- Step schema
- Part repository
- retrieval
- selection
- score aggregation
- workflow mutation

### Golden Workflow test

小さいdeterministic mock Workflowを維持します。

検証内容:

- execution order
- Artifact lineage
- selection result
- Trajectory serialization

### Integration test

小さいfixture datasetを使います。

    ingest
    -> extract
    -> index
    -> retrieve
    -> select
    -> compose
    -> evaluate
    -> output

## 14. Configuration

設定を分離します。

    project config
    dataset config
    model/backend config
    evaluator config
    workflow config
    experiment config

secret/API keyをWorkflow DSLやcommit対象configへ埋め込みません。

## 15. Versioning

記録対象:

- dataset
- Part extraction method
- embedding model
- index
- workflow
- Step implementation/backend
- evaluator
- trained selector/policy

これらのversionがないexperiment resultは再現不能として扱います。

## 16. 初期実装の非目標

- distributed training
- large-scale orchestration cluster
- DSLへの任意programming language機能
- end-to-end RL
- unrestricted automatic graph generation
- polished node-editor UX
- custom foundation image model

## 17. 最初の実行target

    INPUT prompt

    parts = RETRIEVE_PARTS(
      prompt,
      category="generic",
      top=20
    )

    selected = SELECT_PARTS(
      parts,
      top=5
    )

    draft = COMPOSE(
      selected
    )

    scores = EVALUATE(
      draft,
      evaluator="mock"
    )

    OUTPUT draft

次に複数draftへ拡張:

    drafts = COMPOSE(selected, count=4)
    scores = EVALUATE(drafts)
    best = SELECT(drafts, scores=scores, top=1)
    OUTPUT best

これが最初の retrieval-selection-composition-evaluation loopです。

## 18. Architecture phase完了条件

- Runtime coreを変更せず新しいStepを追加できる。
- SELECTを変更せず新しいEvaluatorを追加できる。
- DSLを変えずvector-index backendを差し替えられる。
- GUIとDSLがround-tripする。
- 保存metadataからrunを実験用途として十分に再現できる。
- final imageから使用Part、selection、Stepまで追跡できる。
- 2 Workflowをexperimentとして比較できる。

ここまで完成したら、以後はplumbingよりmodel qualityとlearningへ重点を移します。
