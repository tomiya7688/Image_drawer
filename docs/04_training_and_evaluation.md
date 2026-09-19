# Training and Evaluation

## 1. Initial learning strategy

The first implementation should avoid requiring full end-to-end differentiability.

A practical initial loop is:

```text
1. generate N candidates
2. evaluate candidates
3. retain top K candidates
4. save successful procedure/history
5. train or fine-tune from retained examples
6. repeat
```

This is close to rejection sampling + imitation/fine-tuning and is easier to debug than introducing reinforcement learning immediately.

## 2. Evaluation levels

Evaluation can occur at different workflow stages.

### Sketch evaluation

Possible components:

- composition
- silhouette/readability
- anatomy/structure
- prompt alignment
- line economy

### Color evaluation

Possible components:

- color harmony
- foreground/background separation
- lighting consistency
- material consistency
- prompt alignment

### Final evaluation

Possible components:

- overall preference
- prompt alignment
- composition
- anatomy/structure
- line quality
- color quality
- detail/coherence

## 3. Score representation

Prefer structured scores:

```text
Score {
  overall: 0.84
  components: {
    composition: 0.91
    line_quality: 0.79
    color_quality: 0.87
  }
}
```

The aggregate score may be computed from configurable weights.

## 4. Human evaluation

Human preference should be supported as another evaluator source.

Possible GUI actions:

- choose A over B
- approve
- reject
- assign simple rating
- annotate reason/category

Human judgments can be stored alongside automated evaluator scores.

## 5. Selection

Selection should be independent from evaluation.

Examples:

- best score
- top-k
- weighted random among top candidates
- diversity-aware selection
- Pareto selection across multiple score components

The MVP only requires best/top-k.

## 6. What should be saved

For useful future learning, save more than final images.

For each candidate:

- prompt
- parent artifact ids
- workflow version
- step sequence
- step parameters
- model/checkpoint ids
- random seeds
- intermediate artifacts
- evaluator outputs
- selection result
- final acceptance/rejection

This allows learning and analysis at the procedure level.

## 7. Future strategies

Once the basic system works, possible additions include:

- reinforcement learning
- evolutionary search
- beam search
- workflow-order search
- learned evaluators
- preference models
- self-correction loops
- adaptive branching/candidate counts

These are explicitly not required for the first MVP.
