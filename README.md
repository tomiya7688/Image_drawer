# Image Drawer

Image Drawer is an experimental image-generation project focused on **learning drawing procedures rather than directly learning finished images**.

The core loop is:

```text
state -> drawing step -> intermediate image -> evaluation -> selection / next step
```

Instead of treating image generation as a single black-box operation, Image Drawer decomposes creation into editable workflow steps such as sketching, line drawing, coloring, shading, evaluation, correction, and selection.

## Project goals

- Make the generation procedure visible and editable.
- Let users compose and reorder drawing/training steps from a GUI.
- Represent GUI-editable workflows with a small DSL.
- Evaluate intermediate and final images and use those scores for search and learning.
- Preserve successful action/workflow histories so models can learn procedures.
- Keep the execution engine, evaluators, GUI, and models loosely coupled.

## Initial documentation

- [Concept](docs/00_concept.md)
- [Architecture](docs/01_architecture.md)
- [Workflow DSL](docs/02_workflow_dsl.md)
- [GUI](docs/03_gui.md)
- [Training and evaluation](docs/04_training_and_evaluation.md)
- [MVP scope](docs/05_mvp.md)

This repository is currently in the design/specification phase.
