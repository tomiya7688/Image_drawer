# Architecture

## 1. High-level components

```text
+-------------------+
|       GUI         |
| workflow editor   |
| params / results  |
+---------+---------+
          |
          v
+-------------------+
|   Workflow DSL    |
| serialized graph  |
+---------+---------+
          |
          v
+-------------------+
|     Runtime       |
| validation        |
| scheduling        |
| state/history     |
+----+---------+----+
     |         |
     v         v
+---------+ +-----------+
| Steps   | | Evaluator |
| drawing | | / Critic  |
+----+----+ +-----+-----+
     |            |
     +------+-----+
            v
      +-----------+
      | Selection |
      | / Trainer |
      +-----------+
```

## 2. GUI

The GUI is the primary workflow authoring surface.

It should allow users to:

- add/remove steps
- reorder steps
- connect inputs and outputs
- edit step parameters
- configure training/search parameters
- run workflows
- inspect intermediate outputs
- compare candidate scores
- save/load workflows

The DSL is the persisted representation of the workflow.

## 3. Workflow runtime

The runtime is responsible for:

- parsing workflow definitions
- validating type/input/output compatibility
- constructing the execution graph
- invoking steps
- tracking artifacts
- recording scores
- recording provenance/history
- handling candidate sets
- handling failures

The runtime should not depend on a specific model implementation.

## 4. Step interface

Every operation exposed to the workflow should implement a common conceptual interface:

```python
class Step:
    def run(self, inputs, params, context):
        ...
```

A step receives:

- named inputs
- serializable parameters
- execution context

A step returns one or more named outputs.

## 5. Artifact model

The runtime should treat intermediate values as typed artifacts.

Initial artifact types:

- Text
- Image
- ImageSet
- LineArt
- Mask
- Palette
- Score
- ScoreSet
- Metadata

Later candidates:

- Layer
- StrokeSequence
- Region
- Embedding
- ActionSequence

Each artifact should carry provenance where possible:

- producing step
- parent artifacts
- model/config used
- random seed
- scores
- timestamp/run id

## 6. Model adapters

Drawing models and evaluators should be connected through adapters.

This keeps workflow definitions independent from concrete implementations.

Examples:

```text
SKETCH(model="mock")
SKETCH(model="local_model_a")
SKETCH(model="remote_model_b")
```

The workflow-level operation remains `SKETCH`; only the adapter changes.

## 7. Execution history

Every run should be inspectable.

Minimum history record:

```text
Run
 |- workflow version
 |- global parameters
 |- step execution
 |   |- input artifact ids
 |   |- parameters
 |   |- output artifact ids
 |   |- duration
 |   |- errors
 |- evaluations
 |- selected candidates
 |- final outputs
```

This history later becomes training/search data.
