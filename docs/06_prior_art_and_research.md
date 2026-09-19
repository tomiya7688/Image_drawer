# Prior Art and Research Direction

Status: initial survey
Date: 2026-09-20

## 1. Executive summary

Image Drawer should aggressively reuse existing AI methods rather than training every component from scratch.

The parts that already have strong prior art are:

- iterative / process-driven image generation
- stroke-based neural painting
- conditional image generation from line art and structure
- human-preference reward models
- best-of-N ranking
- diffusion preference optimization
- reinforcement learning over generation trajectories
- differentiable or learned renderers

The project-specific core should be the orchestration layer:

1. represent a drawing procedure as an editable workflow,
2. expose it through a GUI,
3. store intermediate visual states and decisions,
4. combine heterogeneous generators and evaluators,
5. learn from successful trajectories,
6. eventually search or optimize workflow structure itself.

## 2. Closest research direction: process-driven image generation

### Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning
Zhang et al., 2026. arXiv:2604.04746

This is currently one of the closest references to the Image Drawer concept.

It decomposes generation into repeated stages:

    textual planning
    -> visual drafting
    -> textual reflection
    -> visual refinement

Key ideas:

- intermediate visual states are explicit;
- later reasoning is conditioned on intermediate images;
- step-wise supervision is used instead of supervising only the final image;
- intermediate-state evaluation is treated as a central problem;
- the generation process becomes an interpretable trajectory.

Image Drawer differs in emphasis. The paper proposes a particular unified multimodal training approach, while Image Drawer aims to build a model-agnostic workflow/runtime where multiple generation and evaluation methods can be swapped, combined and reordered.

Reference:
https://arxiv.org/abs/2604.04746

### Self-Reflective Reinforcement Learning for Diffusion-based Image Reasoning Generation
Pan et al., 2025. arXiv:2505.22407

This work introduces reflective iterations over diffusion-generation trajectories.

Relevant ideas:

- reflection across generation trajectories;
- iterative correction;
- trajectory-level optimization;
- difficulty of evaluating noisy intermediate diffusion states.

Image Drawer can make semantic intermediate states such as sketch, line art and color draft explicit, which may make evaluation easier than scoring noisy latent states.

Reference:
https://arxiv.org/abs/2505.22407

### Fine-grained Multimodal Reasoning
Kim et al., 2026. arXiv:2604.13491

This approach decomposes a prompt into semantic units, verifies them using visual question answering and performs targeted refinements.

This suggests EVALUATE should eventually support structured outputs such as:

    entity_presence
    attribute_binding
    spatial_relation
    counting
    style
    overall

rather than one scalar score.

Reference:
https://arxiv.org/abs/2604.13491

## 3. Stroke-based neural painting

This family is directly relevant if Image Drawer represents drawing as strokes or renderer commands.

### SPIRAL
Ganin et al., 2018. arXiv:1804.01118

SPIRAL trains an agent to emit graphics-program actions that are executed by a renderer.

Conceptually:

    agent -> program/actions -> renderer -> image -> reward

This is very close to the separation between procedure and image proposed for Image Drawer.

SPIRAL also demonstrates that the renderer itself does not need to be differentiable when policy learning is done with reinforcement learning.

Reference:
https://arxiv.org/abs/1804.01118

### Learning to Paint With Model-Based Deep Reinforcement Learning
Huang et al., ICCV 2019. arXiv:1903.04411

The model learns parameters such as stroke position, shape, color and transparency.

Important reusable ideas:

- parameterized stroke action space;
- Bezier curves;
- neural differentiable stroke renderer;
- canvas-as-state representation;
- sequential policy optimization.

Reference:
https://arxiv.org/abs/1903.04411

### Stylized Neural Painting
Zou et al., CVPR 2021

This work formulates painting as vector-stroke optimization rather than pixel-wise generation.

Relevant techniques:

- vectorized strokes;
- differentiable neural renderer;
- separate rasterization and shading;
- coarse-to-fine rendering;
- optimal-transport-inspired optimization.

Reference:
https://openaccess.thecvf.com/content/CVPR2021/html/Zou_Stylized_Neural_Painting_CVPR_2021_paper.html

### Paint Transformer
Liu et al., ICCV 2021. arXiv:2108.03798

Paint Transformer predicts a set of strokes with a Transformer instead of generating every stroke through RL.

Important lesson:

RL should not automatically be the default for every drawing stage.

Some operators may be better expressed as:

    state -> stroke set

rather than:

    state -> stroke -> state -> stroke -> ...

Reference:
https://arxiv.org/abs/2108.03798

### Differentiable Stroke Planning with Dual Parameterization
Liu et al., CVPR 2026

This work combines discrete structural stroke proposals with continuous Bezier optimization.

This suggests a future hybrid approach for Image Drawer:

    discrete structural planning
    +
    continuous parameter optimization

Reference:
https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Differentiable_Stroke_Planning_with_Dual_Parameterization_for_Efficient_and_High-Fidelity_CVPR_2026_paper.html

## 4. Reusing pretrained image generators

The first versions of Image Drawer should not train a complete image generator.

Existing image-generation models can be wrapped as workflow operators.

### ControlNet
Zhang et al., ICCV 2023. arXiv:2302.05543

ControlNet adds spatial conditioning to a pretrained text-to-image diffusion model.

Typical control inputs include:

- edges
- depth
- segmentation
- pose

For Image Drawer this gives a direct path for operations such as:

    line_art -> COLOR
    structure -> DETAIL
    pose -> RENDER

without training the base image model from scratch.

Reference:
https://arxiv.org/abs/2302.05543

### T2I-Adapter
Mou et al., AAAI 2024. arXiv:2302.08453

T2I-Adapter adds lightweight control modules while freezing the large pretrained image model.

It is relevant because it supports structural and color control and is cheaper to experiment with than full-model fine-tuning.

Reference:
https://arxiv.org/abs/2302.08453

## 5. Evaluation and human preference models

The evaluator should be a subsystem, not one fixed metric.

### CLIP / CLIPScore

Useful as a cheap baseline for text-image semantic compatibility.

Good for:

- rough prompt alignment;
- fast filtering.

Not sufficient for:

- aesthetics;
- fine-grained spatial relationships;
- drawing quality.

Reference:
https://arxiv.org/abs/2104.08718

### ImageReward
Xu et al., NeurIPS 2023. arXiv:2304.05977

ImageReward is trained from expert text-to-image preference comparisons.

It also introduces Reward Feedback Learning for optimizing diffusion models against a scorer.

Potential Image Drawer use:

- final-image reward;
- best-of-N candidate ranking;
- baseline preference model.

Reference:
https://arxiv.org/abs/2304.05977

### PickScore / Pick-a-Pic
Kirstain et al., 2023. arXiv:2305.01569

Pick-a-Pic is an open dataset of real user pairwise preferences.
PickScore predicts those preferences.

Potential use:

- candidate ranking;
- pairwise preference training;
- user-facing A/B feedback format.

Reference:
https://arxiv.org/abs/2305.01569

### HPS v2
Wu et al., 2023. arXiv:2306.09341

HPS v2 is trained from a large-scale human preference dataset and is useful as another independent preference model.

It should be treated as an independent signal rather than replacing all other evaluators.

Reference:
https://arxiv.org/abs/2306.09341

### Multi-dimensional Preference Score
Zhang et al., CVPR 2024. arXiv:2405.14705

MPS explicitly models several preference dimensions including:

- aesthetics;
- semantic alignment;
- detail quality;
- overall preference.

This strongly supports making Image Drawer's score representation structured rather than scalar-only.

Reference:
https://arxiv.org/abs/2405.14705

### VisionReward
Xu et al., AAAI 2026

VisionReward extends visual reward modeling toward hierarchical and fine-grained multi-dimensional preference evaluation.

This is especially relevant to stage-specific evaluation.

Reference:
https://ojs.aaai.org/index.php/AAAI/article/view/38107

### DreamSim
Fu et al., NeurIPS 2023. arXiv:2306.09344

DreamSim models human perceptual similarity including layout, pose and semantic content.

Potential use:

- measure structure preservation between stages;
- detect excessive drift during refinement;
- compare alternate intermediate states.

Reference:
https://arxiv.org/abs/2306.09344

## 6. Preference optimization and reinforcement learning

### DDPO
Black et al., 2023. arXiv:2305.13301

DDPO models diffusion denoising as a multi-step decision process and applies policy-gradient optimization against arbitrary rewards.

This validates reward-driven image-generator optimization.

Recommendation for Image Drawer:
do not begin with DDPO. First collect trajectories and validate evaluators, because otherwise model optimization and reward failure become difficult to distinguish.

Reference:
https://arxiv.org/abs/2305.13301

### Diffusion-DPO
Wallace et al., CVPR 2024. arXiv:2311.12908

Diffusion-DPO learns directly from preferred/rejected image pairs without requiring a separately trained reward model.

Once Image Drawer accumulates records like:

    prompt
    preferred candidate
    rejected candidate

individual image operators could be preference-tuned with this class of method.

Reference:
https://arxiv.org/abs/2311.12908

### Dense Reward View
Yang et al., ICML 2024

This work argues that treating preference only as a terminal reward ignores the sequential structure of diffusion generation.

This supports evaluating intermediate Image Drawer states rather than only the final image.

Reference:
https://proceedings.mlr.press/v235/yang24e.html

## 7. Recommended reuse map

### SKETCH

First implementation:

- existing pretrained image generator;
- sketch / line conditioning through ControlNet or T2I-Adapter.

Later:

- learned stroke generator;
- Paint Transformer-style predictor;
- sequential stroke policy.

### LINE

First implementation:

- image-to-line preprocessing/model;
- line-controlled generation or editing.

Later:

- vector-stroke representation.

### COLOR

First implementation:

- ControlNet or T2I-Adapter conditioned on line art;
- pretrained diffusion/flow backbone.

Do not train a full color model initially.

### REFINE

First implementation:

- pretrained image-to-image/editor model.

Later:

- reflective multimodal refinement;
- localized correction agent.

### EVALUATE

Use multiple independent signals:

    semantic_alignment: CLIP or VLM
    preference: ImageReward / PickScore / HPSv2
    multidimensional: MPS / VisionReward
    structure_preservation: DreamSim
    task_specific: custom evaluator
    human: pairwise preference

### SELECT

Initially:

- weighted top-k;
- best-of-N.

Later:

- diversity-aware selection;
- Pareto selection;
- learned selection policy.

### TRAIN

Initially:

- save preferred/rejected trajectories;
- supervised imitation / fine-tuning;
- pairwise preference data.

Later:

- Diffusion-DPO;
- DDPO;
- evolutionary/search-based workflow optimization.

## 8. Reward hacking and evaluator bias

Preference models are proxies, not ground truth.

Optimizing one evaluator too aggressively can exploit its weaknesses.

Design requirements:

1. store raw component scores;
2. keep multiple independent evaluators;
3. support human pairwise evaluation;
4. keep held-out evaluation prompts;
5. separate training rewards from benchmark metrics;
6. monitor diversity;
7. do not define quality through one global aesthetic model.

Aesthetic/preference models can also encode dataset and cultural biases.

Therefore Image Drawer should treat quality as multi-objective and evaluator-specific.

## 9. Research-informed architecture changes

### Score should not be a float

Use something conceptually like:

    Score:
      evaluator
      evaluator_version
      overall
      components
      metadata

A candidate may own many scores from different evaluators.

### Artifact should be first-class

Recommended fields:

    Artifact:
      id
      type
      payload
      parents
      producer_step
      model_info
      seed
      metadata

Useful artifact types:

- Text
- Image
- ImageSet
- LineArt
- Mask
- Depth
- Pose
- Palette
- ScoreSet
- Selection

### Trajectory should be first-class

Recommended structure:

    Trajectory:
      workflow_id
      workflow_version
      input
      executions
      evaluations
      human_feedback
      final_artifacts

Each execution should record:

- step type;
- parameters;
- input artifact IDs;
- output artifact IDs;
- model/checkpoint;
- random seed;
- duration/error information.

This trajectory store is likely to become the project's most valuable training dataset.

## 10. Recommended development stages

### R0: runtime validation

Use mock steps.

Prove:

- GUI <-> DSL round trip;
- artifact tracking;
- candidate branching;
- scores;
- provenance.

### R1: existing-model pipeline

Run a real pipeline such as:

    prompt
    -> sketch/structure
    -> controlled color
    -> evaluate
    -> select

Use pretrained components only.

### R2: evaluator study

Compare several evaluators against human A/B choices.

Study disagreement between:

- ImageReward;
- PickScore;
- HPSv2;
- MPS / VisionReward where practical;
- semantic metrics;
- human preference.

### R3: trajectory dataset

Collect:

- intermediate artifacts;
- rejected and accepted branches;
- evaluator outputs;
- human choices;
- workflow configuration.

### R4: learn one operator

Train only one component first.

Candidates:

- sketch selector;
- refinement policy;
- color operator;
- stroke predictor.

### R5: workflow optimization

Search over:

- operation order;
- branch count;
- candidate count;
- evaluator placement;
- model/operator choice;
- refinement count.

Start with simple search algorithms before RL:

- grid search;
- random search;
- Bayesian optimization;
- evolutionary search.

## 11. Main research hypothesis

A workflow W maps an initial state into a trajectory:

    tau = (s0, a0, s1, a1, ..., sT)

Each action is an operator invocation.

The objective is not only final-image reward.

Possible objectives include:

- final quality;
- prompt alignment;
- controllability;
- editability;
- efficiency;
- diversity;
- intermediate consistency.

This makes Image Drawer naturally a multi-objective procedural-generation system.

## 12. Current recommendation

For implementation after the specification phase:

1. keep the DSL/runtime model-agnostic;
2. make artifacts and trajectories first-class;
3. make scores multi-dimensional and evaluator-specific;
4. use a plugin-style operator registry;
5. integrate existing generation/evaluation methods first;
6. collect trajectories before choosing a training algorithm;
7. use Best-of-N / ranking before RL;
8. treat workflow optimization as a separate research layer.

The AI components will change quickly. The durable value of Image Drawer should be the workflow, provenance, evaluation and experimentation infrastructure around them.
