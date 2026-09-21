# Image Drawer

Image Drawer is an experimental image-generation project focused on **learning drawing procedures rather than directly learning finished images**.

The core loop is:

```text
state -> drawing step -> intermediate image -> evaluation -> selection / next step
```

Instead of treating image generation as a single black-box operation, Image Drawer decomposes creation into editable workflow steps such as sketching, part retrieval, composition, coloring, evaluation, correction, and selection.

The current design assumes a large image dataset and uses it as a reusable **Part Bank**. The system learns not only which parts to select and combine, but eventually which production workflow should be used for a given task.

## Project goals

- Make the generation procedure visible and editable.
- Let users compose and reorder drawing/training steps from a GUI.
- Represent GUI-editable workflows with a small DSL.
- Retrieve reusable visual parts from a large image dataset.
- Evaluate intermediate and final images and use those scores for search and learning.
- Preserve accepted/rejected candidates and complete workflow trajectories.
- Learn part selection, composition decisions, and eventually workflow choice/order.
- Keep the execution engine, evaluators, GUI, search algorithms, and models loosely coupled.
- Reuse existing image-generation, control, retrieval, reward-model and preference-learning methods wherever practical.

## Documentation

- [Concept](docs/00_concept.md)
- [Architecture](docs/01_architecture.md)
- [Workflow DSL](docs/02_workflow_dsl.md)
- [GUI](docs/03_gui.md)
- [Training and evaluation](docs/04_training_and_evaluation.md)
- [MVP scope](docs/05_mvp.md)
- [Prior art and research survey](docs/06_prior_art_and_research.md)
- [Part Bank specification](docs/07_part_bank.md)
- [Workflow Search specification](docs/08_workflow_search.md)
- [Implementation plan](docs/09_implementation_plan.md)

## Recommended implementation entry point

Start with [Implementation plan](docs/09_implementation_plan.md).

The first end-to-end target is:

```text
dataset
  -> Part Bank
  -> RETRIEVE_PARTS
  -> SELECT_PARTS
  -> COMPOSE
  -> EVALUATE
  -> Trajectory
  -> workflow comparison
```

High-quality generation models are intentionally not required for the first architecture milestone. Mock/simple backends should be used until the workflow, artifact, provenance and experiment infrastructure is stable.

## Development

The M0 project skeleton is intentionally dependency-light and requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m image_drawer --input hello
image-drawer-gui --check
```

CI verifies three layers on every pull request and push to `main`: source-level input/output tests, package build, and execution of the generated wheel from a fresh virtual environment.

Implementation now proceeds milestone-by-milestone through GitHub Issues and pull requests.
