# Workflow DSL

## 1. Role

The DSL is the serialized form of a workflow manipulated by the GUI.

It is deliberately small.

The first version only needs to describe:

- inputs
- ordered steps
- step inputs
- parameters
- outputs

Branching and loops may be represented by dedicated steps rather than language-level control flow.

## 2. Design goals

- easy to read
- easy to generate from GUI state
- easy to parse
- stable diffs in Git
- explicit data dependencies
- simple validation
- no hidden state

## 3. Proposed syntax

Example:

```text
INPUT prompt: Text

sketches = SKETCH(
  prompt,
  count=8
)

line_scores = EVALUATE(
  sketches,
  evaluator="line"
)

lines = SELECT(
  sketches,
  scores=line_scores,
  top=2
)

colored = COLOR(
  lines,
  prompt=prompt,
  count=4
)

final_scores = EVALUATE(
  colored,
  evaluator="final"
)

final = SELECT(
  colored,
  scores=final_scores,
  top=1
)

OUTPUT final
```

## 4. GUI mapping

Each assignment corresponds to one GUI node.

For example:

```text
colored = COLOR(lines, prompt=prompt, count=4)
```

maps to:

```text
Node type: COLOR
Node id: colored

Inputs:
  image <- lines
  prompt <- prompt

Parameters:
  count = 4

Output:
  colored
```

The GUI should be able to regenerate canonical DSL text from its graph.

## 5. Initial built-in operations

### INPUT

Declares an external workflow input.

### SKETCH

Produces one or more sketch candidates.

### LINE

Converts or refines an image into line art.

### COLOR

Adds color using an image/line artifact and optional prompt/context.

### EVALUATE

Returns scores for one or more artifacts.

### SELECT

Selects artifacts by score or another selection strategy.

### OUTPUT

Declares workflow output.

## 6. Candidate collections

Candidate sets should be first-class values.

Example:

```text
sketches = SKETCH(prompt, count=8)
scores = EVALUATE(sketches, evaluator="line")
best = SELECT(sketches, scores=scores, top=2)
```

The runtime may internally represent this as `ImageSet` + `ScoreSet`.

## 7. Parameters

Step parameters should use simple serializable primitives:

- string
- integer
- float
- boolean
- list
- enum-like string

Avoid arbitrary executable expressions in the first version.

## 8. Control flow

Version 1 should avoid general `if`, `for`, or `while` syntax.

Instead, workflow operations can express behavior explicitly:

```text
candidates = SKETCH(prompt, count=8)
best = SELECT(candidates, top=2)
```

Later operations could include:

```text
RETRY
BRANCH
MERGE
STOP_IF
REFINE
```

If these prove insufficient, language-level control flow can be reconsidered.

## 9. Validation

Before execution, validate:

- referenced variables exist
- required inputs are connected
- parameter names are valid
- parameter types are valid
- artifact types are compatible
- outputs are unique
- required workflow outputs exist

The GUI should surface these errors before Run is enabled where possible.
