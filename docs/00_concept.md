# Concept

## 1. Purpose

Image Drawer explores a different approach to image generation.

The primary learning target is not a finished image distribution. Instead, the system learns and improves **procedures for producing an image**.

Examples of procedures include:

```text
prompt
  -> sketch
  -> evaluate
  -> select
  -> line
  -> color
  -> evaluate
  -> correct
  -> final
```

The image itself is still important, but mainly as an observable state and as an input to evaluation.

## 2. Core idea

The system repeatedly performs:

```text
State -> Action -> New State -> Evaluation
```

A state may include:

- prompt
- canvas/image
- line art
- masks
- palette
- layers
- previous scores
- generation history

An action may include:

- sketch
- draw line
- fill
- color
- shade
- erase
- refine
- branch
- evaluate
- select

## 3. Why split the process

A procedural representation gives us several useful properties:

- Intermediate failures can be identified.
- Different evaluators can be used for different stages.
- Multiple candidates can branch from the same intermediate result.
- Good procedures can be retained even when individual images differ.
- The generation order itself can eventually become a search or learning target.
- Users can inspect and change the procedure through the GUI.

## 4. First principle

The DSL is **not intended to become a general-purpose programming language**.

Its main role is to serialize and execute workflows that users manipulate in the GUI.

Therefore the DSL should prioritize:

1. simple step definitions
2. explicit inputs and outputs
3. readable parameters
4. deterministic serialization
5. easy GUI <-> DSL conversion
6. validation before execution

Complex logic should live in the runtime or in reusable step implementations, not in DSL syntax.

## 5. Long-term direction

The initial workflow order will be human-defined.

Later, the system may search or learn:

- which steps to run
- how many candidates to generate
- where to evaluate
- when to branch
- when to retry
- which intermediate result to reuse
- which order of operations produces better results
