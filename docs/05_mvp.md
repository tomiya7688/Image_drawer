# MVP Scope

## 1. Goal

The first MVP should prove that the architecture can execute and inspect an editable procedural image workflow.

It does not need to produce high-quality artwork.

## 2. Required components

### DSL parser

Parse the initial workflow format.

### Runtime

Execute a linear/dependency-based workflow and track artifacts.

### Step registry

Register operations by name and expose their schemas.

### Mock drawing steps

Provide deterministic/simple implementations for:

- SKETCH
- COLOR

These can initially generate placeholder images or transform simple test images.

### Mock evaluator

Return reproducible scores so selection logic can be tested.

### SELECT

Support top-k selection.

### GUI

Support:

- workflow step editing
- parameter editing
- run
- intermediate artifact display
- score display
- DSL view/edit
- validation errors

## 3. Proposed repository structure

```text
Image_drawer/
├─ README.md
├─ docs/
├─ pyproject.toml
├─ image_drawer/
│  ├─ dsl/
│  │  ├─ parser.py
│  │  ├─ runtime.py
│  │  └─ types.py
│  ├─ steps/
│  │  ├─ base.py
│  │  ├─ registry.py
│  │  ├─ sketch.py
│  │  ├─ color.py
│  │  ├─ evaluate.py
│  │  └─ select.py
│  ├─ training/
│  │  ├─ runner.py
│  │  └─ config.py
│  └─ gui/
│     └─ app.py
├─ workflows/
│  └─ basic.idraw
└─ tests/
```

## 4. First end-to-end workflow

```text
INPUT prompt: Text

sketches = SKETCH(prompt, count=4)
scores = EVALUATE(sketches, evaluator="mock")
best = SELECT(sketches, scores=scores, top=1)
colored = COLOR(best, prompt=prompt)

OUTPUT colored
```

Successful MVP behavior:

1. GUI loads this workflow.
2. User changes `count` from GUI.
3. DSL reflects the change.
4. Runtime executes each step.
5. Intermediate candidates appear.
6. Evaluator scores appear.
7. SELECT highlights the chosen candidate.
8. Final output is visible.
9. Run history is stored locally.

## 5. Non-goals for MVP

Not required initially:

- high-quality image models
- distributed training
- reinforcement learning
- automatic workflow discovery
- arbitrary DSL programming
- multi-user/server architecture
- production deployment

The MVP is primarily an architecture and experiment-loop validation.
