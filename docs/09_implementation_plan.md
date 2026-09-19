# Implementation Plan

Status: adopted
Date: 2026-09-20

## 1. Goal

This document converts the current design into an implementation order.

The first target is an end-to-end research workbench, not image quality.

The first useful loop is:

    dataset
      -> Part Bank
      -> GUI/DSL workflow
      -> retrieval
      -> selection
      -> composition
      -> evaluation
      -> trajectory storage
      -> workflow comparison

## 2. Milestone M0: project skeleton

Create:

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

Deliverables:

- Python package
- config loading
- logging
- test setup
- CLI entrypoint
- GUI entrypoint

## 3. Milestone M1: core models

Implement first:

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

Rules:

- use stable string IDs;
- all records should be serializable;
- model/backend/version metadata must be explicit;
- avoid storing large image bytes directly in JSON records; use URIs/paths.

Acceptance:

- round-trip serialization tests pass;
- old records can be loaded after minor schema additions where practical.

## 4. Milestone M2: runtime and DSL

Implement:

- Step base interface
- Step registry
- Workflow validation
- DSL parser
- canonical DSL serializer
- runtime execution
- artifact registry
- trajectory logger

Initial Step set:

- INPUT
- RETRIEVE_PARTS
- SELECT_PARTS
- COMPOSE
- EVALUATE
- SELECT
- OUTPUT

REFINE can initially be a pass-through/mock operator.

Acceptance:

- GUI is not required yet;
- a workflow file can run from CLI;
- all Step inputs/outputs are recorded.

## 5. Milestone M3: Part Bank v0

Implement:

- filesystem ingest
- SourceImage records
- one simple extractor
- Part records
- one embedding backend
- one vector index backend
- text/category retrieval

For the first extractor, use the simplest dataset-compatible method available.

Do not block M3 on a perfect segmentation model.

Acceptance:

- ingest a small development subset;
- build index;
- retrieve top-k Parts from a text/category query;
- preserve source provenance.

## 6. Milestone M4: simple composition

Implement:

- PartPlacement
- Layout
- COMPOSE renderer

Initial renderer may support only:

- translation
- scale
- rotation
- z-order
- alpha/mask

Acceptance:

- selected Parts can produce a deterministic draft image;
- Composition is stored as an Artifact independently from the rasterized image.

## 7. Milestone M5: evaluator subsystem

Implement evaluator interface:

    class Evaluator:
        def evaluate(self, artifacts, context) -> ScoreSet:
            ...

At least one baseline evaluator must work.

Support:

- evaluator name/version
- multiple component scores
- aggregate score
- batch evaluation

Acceptance:

- EVALUATE can score ImageSet;
- SELECT can choose top-k from ScoreSet;
- raw evaluator outputs are preserved.

## 8. Milestone M6: GUI v0

Build a workbench, not a polished editor.

Required:

- load/save workflow
- ordered Step list
- add/delete/reorder Step
- parameter editor generated from Step schema
- DSL text view
- Run button
- intermediate Artifact viewer
- candidate thumbnails
- score table
- selected/rejected indication
- run history list

A vertical editor is preferred before implementing a free-form node graph.

Acceptance:

- changing a parameter in GUI changes serialized DSL;
- loading DSL reconstructs GUI state;
- a full Part retrieval workflow can run from GUI.

## 9. Milestone M7: experiment runner

Implement:

- prompt-set loader
- repeated run execution
- seed policy
- workflow comparison
- aggregate metrics
- results export

Acceptance:

- compare at least two workflows on the same prompt set;
- persist all Trajectories and experiment metadata.

## 10. Milestone M8: parameter search

Implement:

- grid search
- random search

Initial searchable parameters:

- retrieval top-k
- selected top-k
- number of composed candidates
- evaluator weights

Acceptance:

- search can reproduce best configuration from saved result metadata.

## 11. Milestone M9: real model integration

Only after the architecture works.

Candidate integrations:

- pretrained image editing/refinement model
- ControlNet/T2I-Adapter-like conditioned generation
- preference evaluator
- perceptual/structure evaluator

Each integration must be an adapter behind an existing Step/Evaluator interface.

Do not change DSL syntax just to support one model.

## 12. Milestone M10: learning

Use collected trajectories.

First learning targets:

1. PartSelector
2. prompt -> WorkflowSelector
3. refinement decision/gating

Do not attempt end-to-end RL first.

Recommended progression:

    logged data
    -> supervised ranking
    -> pairwise preference learning
    -> workflow selector
    -> constrained structural search
    -> RL only when necessary

## 13. Testing strategy

### Unit tests

- DSL parsing
- serialization
- type validation
- Step schemas
- Part repository
- retrieval
- selection
- score aggregation
- workflow mutations

### Golden workflow tests

Maintain tiny deterministic workflows using mock backends.

They must verify:

- exact execution order
- artifact lineage
- selection result
- trajectory serialization

### Integration tests

Use a tiny fixture dataset.

Test:

    ingest
    -> extract
    -> index
    -> retrieve
    -> select
    -> compose
    -> evaluate
    -> output

## 14. Configuration

Separate configuration into:

    project config
    dataset config
    model/backend config
    evaluator config
    workflow config
    experiment config

Secrets/API keys must never be embedded in workflow DSL or committed configs.

## 15. Versioning

Track versions for:

- dataset
- Part extraction method
- embedding model
- index
- workflow
- Step implementation/backend
- evaluator
- trained selector/policy

Experiment results without these versions should be treated as non-reproducible.

## 16. Non-goals for the first implementation

Do not initially build:

- distributed training
- large-scale orchestration cluster
- arbitrary programming language features in DSL
- end-to-end RL
- automatic unrestricted graph generation
- polished node-editor UX
- custom foundation image model

## 17. First executable target

The first end-to-end workflow should be:

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

Then expand to multiple drafts:

    drafts = COMPOSE(selected, count=4)
    scores = EVALUATE(drafts)
    best = SELECT(drafts, scores=scores, top=1)
    OUTPUT best

This gives the first complete retrieval-selection-composition-evaluation loop.

## 18. Definition of done for architecture phase

The architecture phase is done when:

- a developer can add a new Step without editing the runtime core;
- a developer can add a new evaluator without editing SELECT;
- a developer can swap vector-index backend without changing DSL;
- GUI and DSL round-trip;
- every run is reproducible from saved metadata to a reasonable experimental standard;
- every final image can be traced back through Parts, selections and Steps;
- two workflows can be compared experimentally.

At that point, further work should focus on model quality and learning rather than plumbing.
