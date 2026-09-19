# Workflow Search Specification

Status: adopted design
Date: 2026-09-20

## 1. Purpose

Workflow Search is the layer that learns or searches for which image-production procedure should be used.

Image Drawer should learn not only:

- which visual Part to choose,
- how to compose Parts,

but eventually:

- which Step should run next,
- when evaluation should occur,
- where branching is useful,
- how many candidates should be generated,
- when refinement should stop,
- which model/operator should be used for a task.

The initial implementation should treat this as an experiment/search problem before introducing reinforcement learning.

## 2. Workflow as a search object

A workflow is represented as a graph or ordered DAG of Step nodes.

Example:

    INPUT
      -> PLAN_LAYOUT
      -> RETRIEVE_PARTS
      -> SELECT_PARTS
      -> COMPOSE
      -> REFINE
      -> EVALUATE
      -> OUTPUT

Alternative:

    INPUT
      -> RETRIEVE_PARTS
      -> SELECT_PARTS
      -> COMPOSE
      -> EVALUATE
      -> REFINE
      -> EVALUATE
      -> OUTPUT

Both should be executable through the same runtime.

## 3. Searchable dimensions

### Structure

- add/remove Step
- reorder compatible Steps
- move EVALUATE
- insert SELECT
- insert REFINE
- branch candidate generation
- merge candidates

### Parameters

- candidate count
- top-k
- evaluator weights
- refinement count
- retrieval depth
- thresholds
- temperatures/seeds where applicable

### Backend choice

A logical Step may have several backends.

Example:

    COLOR.backend = controlnet_a
    COLOR.backend = adapter_b

Workflow Search may eventually include backend selection.

## 4. Constraints

Workflow search must never generate arbitrary invalid graphs.

Every Step exposes a schema:

    inputs
    outputs
    parameter schema
    capabilities

The search engine must operate only on workflows that pass static validation.

Required validation:

- all inputs are available;
- artifact types are compatible;
- OUTPUT is reachable;
- no illegal cycles for the current runtime;
- required parameters are valid.

## 5. Objective model

Do not optimize only one image score.

A run may produce:

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

Initial MVP can use a weighted aggregate, but all components must be preserved.

Example:

    objective =
        0.40 * final_quality
      + 0.25 * prompt_alignment
      + 0.20 * human_or_preference_score
      + 0.15 * diversity
      - compute_penalty

Weights are experiment configuration, not hard-coded policy.

## 6. Experiment unit

A single workflow should never be judged from one prompt.

Define:

    Experiment
      workflow
      prompt_set
      seeds
      evaluator_config
      dataset/index_version
      model_versions

Output:

    ExperimentResult
      runs[]
      aggregate_metrics
      confidence/statistics
      failures[]

The same prompt set and seed policy should be reused when comparing workflows where possible.

## 7. Trajectory requirements

Every run must store:

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

Each Step execution records:

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

This is the training/search log.

## 8. Search stages

### Stage 0: manual workflows

Users build workflows in the GUI.

Purpose:

- establish baselines;
- discover useful operators;
- validate logging.

No automatic search.

### Stage 1: parameter search

Keep workflow structure fixed.

Search:

- candidate count
- top-k
- evaluator weights
- retrieval depth
- refinement strength/count

Algorithms:

- grid search
- random search

This should be implemented first.

### Stage 2: template search

Define several valid workflow templates.

Example:

    template_a:
      RETRIEVE -> SELECT -> COMPOSE -> REFINE

    template_b:
      PLAN -> RETRIEVE -> SELECT -> COMPOSE -> REFINE

    template_c:
      RETRIEVE -> SELECT -> COMPOSE -> EVAL -> REFINE

Compare templates using common prompt sets.

### Stage 3: constrained structural search

Allow safe mutations:

- insert EVALUATE
- insert SELECT
- insert REFINE
- change candidate count
- swap compatible operators
- enable/disable optional Step

Candidate algorithms:

- random search
- evolutionary search
- Bayesian optimization over mixed parameters

### Stage 4: workflow policy

Once enough trajectories exist, train:

    choose_workflow(prompt, context)

or:

    choose_next_step(state)

This is where learned orchestration begins.

### Stage 5: RL / sequential policy

Only introduce RL if:

- state/action definitions are stable;
- evaluator behavior is understood;
- reward hacking is monitored;
- search baselines are insufficient.

RL is not an MVP dependency.

## 9. Search-space representation

Recommended internal structure:

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

The DSL is a serialization of WorkflowSpec.

The search layer should mutate WorkflowSpec, then serialize to DSL if needed.

Do not mutate raw DSL strings.

## 10. Mutation interface

Future interface:

    class WorkflowMutator:
        def propose(self, workflow, search_space, rng):
            ...

Safe mutation types:

    SetParameter
    ReplaceBackend
    InsertStep
    DisableStep
    ChangeCandidateCount
    ChangeTopK

Every mutation must be validated before execution.

## 11. Baseline workflows

The repository should ship with explicit baselines.

### Baseline A: retrieval-first

    INPUT prompt
    parts = RETRIEVE_PARTS(prompt, top=20)
    selected = SELECT_PARTS(parts, top=5)
    draft = COMPOSE(selected)
    final = REFINE(draft, prompt)
    OUTPUT final

### Baseline B: evaluate before refine

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

These form the first workflow-comparison experiment.

## 12. Selection learning

Part selection and workflow selection should be kept separate.

### PartSelector

Question:

    which candidate Part should be used?

Inputs:

- PartSet
- current composition
- prompt
- layout
- evaluator features

### WorkflowSelector

Question:

    which production workflow should be used?

Inputs:

- prompt
- requested style/task
- resource budget
- optional dataset/context features

Keeping these separate makes training and evaluation easier.

## 13. Next-step policy

Long-term representation:

    policy(state) -> StepDecision

Where:

    StepDecision
      step_type
      backend
      parameters
      stop: bool

State can include:

- prompt embedding
- current Artifact summary
- current evaluator scores
- executed Step history
- remaining compute budget

This is not required for early versions, but the Trajectory format should preserve the data needed to train it later.

## 14. Stopping policy

A useful production workflow must learn not only what to do, but when to stop.

Possible stop conditions:

- quality score above threshold
- no meaningful improvement after N refinements
- compute budget exhausted
- human acceptance
- selector confidence high enough

Initially these are explicit Step parameters.

Later they may be learned.

## 15. Experiment reproducibility

Every experiment must record:

- git commit
- workflow version
- dataset version
- Part Bank index version
- model/checkpoint identifiers
- evaluator versions
- random seeds
- search configuration

Without this, workflow comparisons will not be trustworthy.

## 16. GUI requirements

The GUI should eventually expose a Workflow Search panel.

MVP controls:

- choose workflow/template
- choose prompt set
- choose search parameters
- start experiment
- show run table
- compare aggregate scores
- inspect winning and losing trajectories

Later:

- mutation visualization
- Pareto front
- cost/quality plots
- prompt-specific workflow recommendations

## 17. Implementation package structure

Suggested modules:

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

MVP only needs:

- models.py
- objective.py
- experiment.py
- parameter_search.py
- results.py

## 18. MVP acceptance criteria

Workflow Search v0 is complete when:

1. at least two valid workflows can be defined;
2. both run on the same prompt set;
3. all trajectories are persisted;
4. evaluator metrics are aggregated by workflow;
5. runtime/failure metrics are recorded;
6. a simple weighted objective ranks experiment results;
7. parameter search can vary at least one Step parameter;
8. the best configuration can be reproduced from saved metadata;
9. GUI can display comparison results.

No RL is required.

## 19. First research experiment

Recommended first experiment:

Question:

    Does candidate generation + evaluation + selection improve results enough to justify its compute cost?

Compare:

    A:
      RETRIEVE -> SELECT -> COMPOSE -> REFINE

    B:
      RETRIEVE -> SELECT -> COMPOSE(x4)
      -> EVALUATE -> SELECT -> REFINE

Measure:

- preference/final quality
- prompt alignment
- diversity
- runtime
- failure rate
- human A/B preference on a subset

This experiment validates both the architecture and the core project hypothesis.

## 20. Implementation priority

Recommended order:

1. Trajectory persistence
2. WorkflowSpec data model
3. reusable Experiment runner
4. objective aggregation
5. parameter search
6. baseline workflow comparison
7. template search
8. learned workflow selection
9. structural/evolutionary search
10. RL only if justified

The system should remain useful even if the learning algorithm changes.
