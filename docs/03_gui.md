# GUI

## 1. Purpose

The GUI is an experiment workbench for building, running, and inspecting drawing workflows.

The DSL is an implementation/persistence layer underneath the GUI, not the primary interface users must write manually.

## 2. Initial layout

Suggested layout:

```text
+----------------+---------------------------+
| Parameters     | Workflow                  |
|                |                           |
| run settings   | [SKETCH]                  |
| model settings |    |                      |
| train settings | [EVALUATE]                |
|                |    |                      |
|                | [SELECT]                  |
|                |    |                      |
|                | [COLOR]                   |
+----------------+---------------------------+
| Results / Intermediate artifacts           |
| candidate thumbnails / scores / history    |
+--------------------------------------------+
```

## 3. Workflow editor

Version 1 needs:

- add step
- delete step
- reorder step
- connect step outputs to inputs
- edit parameters
- enable/disable step
- inspect generated DSL
- load/save workflow

A full free-form node editor is optional for the first MVP.

A vertical step list with connection selectors may be significantly easier to implement and debug.

## 4. Parameter editor

Parameters should come from each step's schema.

Example:

```text
SKETCH

model      [mock        v]
count      [8            ]
temperature[0.8          ]
seed       [random       ]
```

This implies every step should expose metadata such as:

```text
name
description
input schema
output schema
parameter schema
default values
```

The GUI should be schema-driven rather than manually implementing a form for every step.

## 5. Result inspector

For each run, users should be able to inspect:

- intermediate images
- candidate groups
- scores
- selected/rejected candidates
- step parameters
- execution history
- final output

Selection decisions should be visually traceable.

## 6. Evaluator view

Scores should not be limited to one scalar.

Example:

```text
overall        0.84
composition    0.91
line_quality   0.79
color          0.87
prompt_match   0.82
```

The GUI should support both:

- aggregate score
- component scores

## 7. Training view

The initial training panel should expose only parameters needed by the chosen training/search strategy.

Examples:

- candidate count
- top-k
- epochs/iterations
- learning rate
- batch size
- evaluator weights
- checkpoint interval

Avoid showing parameters that are irrelevant to the active strategy.

## 8. DSL editor/view

The GUI should provide a text view of the generated DSL.

Version 1 can support:

- read/edit text
- parse
- validation errors
- apply back to GUI

Canonical formatting should keep GUI-generated workflows stable in Git diffs.
