# 先行研究と技術方針

状態: 初期調査
日付: 2026-09-20

## 1. 要約

Image Drawerでは、すべてのAI componentを独自に学習するのではなく、既存手法を積極的に再利用します。

先行研究が豊富な領域:

- iterative / process-driven image generation
- stroke-based neural painting
- line artやstructureを条件としたimage generation
- human-preference reward model
- best-of-N ranking
- diffusion preference optimization
- generation trajectoryに対するreinforcement learning
- differentiable / learned renderer

Image Drawer固有の中心はorchestration layerです。

1. 制作手順を編集可能なWorkflowとして表現する。
2. GUIから操作できるようにする。
3. 中間視覚状態と判断を保存する。
4. 異なるgeneratorとEvaluatorを組み合わせる。
5. 良いTrajectoryから学習する。
6. 最終的にWorkflow構造そのものを探索・最適化する。

## 2. 最も近い方向: Process-driven image generation

### Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning
Zhang et al., 2026. arXiv:2604.04746

Image Drawerの構想にかなり近い研究です。

生成を次の反復段階へ分解します。

    textual planning
    -> visual drafting
    -> textual reflection
    -> visual refinement

重要な点:

- 中間視覚状態を明示する。
- 後続reasoningが中間画像を条件にする。
- final imageだけでなくstep-wise supervisionを使う。
- 中間状態評価を中心課題として扱う。
- 生成全体を解釈可能なTrajectoryとして扱える。

違いとして、この論文は特定のunified multimodal trainingを提案します。Image Drawerは複数の生成・評価手法を交換、混在、並べ替えできるmodel-agnosticなWorkflow/Runtimeを目指します。

Reference:
https://arxiv.org/abs/2604.04746

### Self-Reflective Reinforcement Learning for Diffusion-based Image Reasoning Generation
Pan et al., 2025. arXiv:2505.22407

Diffusion generation trajectoryにreflectionを挟む研究です。

関連点:

- generation trajectoryを跨いだreflection
- iterative correction
- trajectory-level optimization
- noisy intermediate diffusion stateの評価難易度

Image Drawerではsketch、line art、color draftなど意味のある中間状態をArtifactとして明示することで、noisy latentを直接評価するより扱いやすくできる可能性があります。

Reference:
https://arxiv.org/abs/2505.22407

### Fine-grained Multimodal Reasoning
Kim et al., 2026. arXiv:2604.13491

promptを細かいsemantic unitへ分解し、VQAで検証してtargeted refinementを行う方向です。

EVALUATEは将来的に次のようなstructured outputを持つべきです。

    entity_presence
    attribute_binding
    spatial_relation
    counting
    style
    overall

単一scalarだけにしないことが重要です。

Reference:
https://arxiv.org/abs/2604.13491

## 3. Stroke-based neural painting

Image Drawerで描画をstrokeやrenderer commandとして表現する場合に直接関係します。

### SPIRAL
Ganin et al., 2018. arXiv:1804.01118

Agentがgraphics-program actionを出力し、rendererが画像化します。

    agent -> program/actions -> renderer -> image -> reward

手順と画像を分離するImage Drawerの考えに近い構成です。

policyをreinforcement learningで最適化する場合、rendererそのものがdifferentiableでなくても成立することを示しています。

Reference:
https://arxiv.org/abs/1804.01118

### Learning to Paint With Model-Based Deep Reinforcement Learning
Huang et al., ICCV 2019. arXiv:1903.04411

stroke位置、形状、色、透明度などを学習します。

再利用候補:

- parameterized stroke action space
- Bezier curve
- neural differentiable stroke renderer
- canvas-as-state
- sequential policy optimization

Reference:
https://arxiv.org/abs/1903.04411

### Stylized Neural Painting
Zou et al., CVPR 2021

pixel生成ではなくvector stroke optimizationとしてpaintingを定式化します。

関連技術:

- vectorized stroke
- differentiable neural renderer
- rasterizationとshadingの分離
- coarse-to-fine rendering
- optimal-transport-inspired optimization

Reference:
https://openaccess.thecvf.com/content/CVPR2021/html/Zou_Stylized_Neural_Painting_CVPR_2021_paper.html

### Paint Transformer
Liu et al., ICCV 2021. arXiv:2108.03798

1 strokeずつRLで出す代わりにTransformerでstroke setを予測します。

重要な示唆は、すべての描画StepにRLを使う必要はないことです。

    state -> stroke set

が適切なStepもあれば、

    state -> stroke -> state -> stroke -> ...

が適切なStepもあります。

Reference:
https://arxiv.org/abs/2108.03798

### Differentiable Stroke Planning with Dual Parameterization
Liu et al., CVPR 2026

discreteなstructural stroke proposalとcontinuousなBezier optimizationを組み合わせます。

将来的なImage Drawerでも、

    discrete structural planning
    +
    continuous parameter optimization

のhybrid方式が候補になります。

Reference:
https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Differentiable_Stroke_Planning_with_Dual_Parameterization_for_Efficient_and_High-Fidelity_CVPR_2026_paper.html

## 4. Pretrained image generatorの再利用

初期Image Drawerでは完全な画像生成modelを新規学習しません。

既存modelをWorkflow operatorとしてwrapします。

### ControlNet
Zhang et al., ICCV 2023. arXiv:2302.05543

pretrained text-to-image diffusionへspatial conditionを追加します。

代表的なcondition:

- edge
- depth
- segmentation
- pose

Image Drawerでは例えば次の実装に使えます。

    line_art -> COLOR
    structure -> DETAIL
    pose -> RENDER

base image modelを最初から学習せずに済みます。

Reference:
https://arxiv.org/abs/2302.05543

### T2I-Adapter
Mou et al., AAAI 2024. arXiv:2302.08453

大規模pretrained image modelをfreezeしたまま、軽量control moduleを追加します。

structureやcolor controlに利用でき、full-model fine-tuningより初期実験に向いています。

Reference:
https://arxiv.org/abs/2302.08453

## 5. 評価とHuman Preference Model

Evaluatorは固定metric 1個ではなくsubsystemとして扱います。

### CLIP / CLIPScore

text-image semantic compatibilityの安価なbaselineです。

向いている用途:

- rough prompt alignment
- fast filtering

不十分な用途:

- aesthetics
- 細かいspatial relation
- drawing quality

Reference:
https://arxiv.org/abs/2104.08718

### ImageReward
Xu et al., NeurIPS 2023. arXiv:2304.05977

expertのtext-to-image preference comparisonから学習したreward modelです。Reward Feedback Learningも提案しています。

Image Drawerでの候補用途:

- final-image reward
- best-of-N candidate ranking
- preference baseline

Reference:
https://arxiv.org/abs/2304.05977

### PickScore / Pick-a-Pic
Kirstain et al., 2023. arXiv:2305.01569

Pick-a-Picは実ユーザーのpairwise preference datasetで、PickScoreはその選好を予測します。

候補用途:

- candidate ranking
- pairwise preference training
- GUI上のA/B feedback

Reference:
https://arxiv.org/abs/2305.01569

### HPS v2
Wu et al., 2023. arXiv:2306.09341

大規模human preference datasetから学習した別系統の評価signalです。

他Evaluatorを置き換えるのではなく、独立signalとして扱います。

Reference:
https://arxiv.org/abs/2306.09341

### Multi-dimensional Preference Score
Zhang et al., CVPR 2024. arXiv:2405.14705

MPSは複数のpreference dimensionを明示的に扱います。

- aesthetics
- semantic alignment
- detail quality
- overall preference

Image DrawerのScoreをscalar-onlyではなくstructured dataにする根拠になります。

Reference:
https://arxiv.org/abs/2405.14705

### VisionReward
Xu et al., AAAI 2026

hierarchicalかつfine-grainedなmulti-dimensional visual preference evaluationを扱います。

特にstage-specific evaluationと相性があります。

Reference:
https://ojs.aaai.org/index.php/AAAI/article/view/38107

### DreamSim
Fu et al., NeurIPS 2023. arXiv:2306.09344

layout、pose、semantic contentを含むhuman perceptual similarityを扱います。

候補用途:

- stage間のstructure preservation
- refinementによる過度なdrift検出
- 中間candidate同士の比較

Reference:
https://arxiv.org/abs/2306.09344

## 6. Preference optimizationとRL

### DDPO
Black et al., 2023. arXiv:2305.13301

diffusion denoisingをmulti-step decision processとして扱い、任意rewardに対してpolicy-gradient optimizationを行います。

Image Drawerでは最初から使いません。まずTrajectoryを収集しEvaluatorを検証しないと、model optimization失敗とreward failureを切り分けにくいためです。

Reference:
https://arxiv.org/abs/2305.13301

### Diffusion-DPO
Wallace et al., CVPR 2024. arXiv:2311.12908

別Reward Modelを学習せず、preferred/rejected image pairから直接学習します。

Image Drawerに

    prompt
    preferred candidate
    rejected candidate

が蓄積した段階で、個別image operatorのpreference tuningに利用できる候補です。

Reference:
https://arxiv.org/abs/2311.12908

### Dense Reward View
Yang et al., ICML 2024

terminal rewardだけではdiffusion generationのsequential structureを無視すると論じています。

final imageだけでなく中間Artifactも評価するImage Drawerの方向と一致します。

Reference:
https://proceedings.mlr.press/v235/yang24e.html

## 7. 推奨reuse map

### SKETCH

初期:
- existing pretrained image generator
- ControlNet / T2I-Adapterによるsketch / line conditioning

将来:
- learned stroke generator
- Paint Transformer-style predictor
- sequential stroke policy

### LINE

初期:
- image-to-line preprocessing / model
- line-controlled generation / editing

将来:
- vector-stroke representation

### COLOR

初期:
- line art conditioned ControlNet / T2I-Adapter
- pretrained diffusion / flow backbone

初期段階ではfull color modelを独自学習しません。

### REFINE

初期:
- pretrained image-to-image / editor model

将来:
- reflective multimodal refinement
- localized correction agent

### EVALUATE

複数の独立signalを利用します。

    semantic_alignment: CLIP or VLM
    preference: ImageReward / PickScore / HPSv2
    multidimensional: MPS / VisionReward
    structure_preservation: DreamSim
    task_specific: custom evaluator
    human: pairwise preference

### SELECT

初期:
- weighted top-k
- best-of-N

将来:
- diversity-aware selection
- Pareto selection
- learned selection policy

### TRAIN

初期:
- preferred/rejected trajectory保存
- supervised imitation / fine-tuning
- pairwise preference data

将来:
- Diffusion-DPO
- DDPO
- evolutionary / search-based workflow optimization

## 8. Reward hackingとEvaluator bias

Preference Modelはproxyでありground truthではありません。

1個のEvaluatorへ過剰最適化すると弱点を利用する可能性があります。

設計要件:

1. raw component scoreを保存する。
2. 複数の独立Evaluatorを使えるようにする。
3. human pairwise evaluationをsupportする。
4. held-out evaluation promptを持つ。
5. training rewardとbenchmark metricを分離する。
6. diversityを監視する。
7. 1個のglobal aesthetic scoreを品質定義にしない。

aesthetic / preference modelにはdatasetや文化的biasも含まれ得ます。

そのため品質はmulti-objectiveかつEvaluator-specificとして扱います。

## 9. Researchから反映するArchitecture変更

### Scoreはfloatだけにしない

    Score:
      evaluator
      evaluator_version
      overall
      components
      metadata

1 candidateに複数EvaluatorのScoreを持たせます。

### Artifactを第一級objectにする

    Artifact:
      id
      type
      payload
      parents
      producer_step
      model_info
      seed
      metadata

有用なArtifact type:

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

### Trajectoryを第一級objectにする

    Trajectory:
      workflow_id
      workflow_version
      input
      executions
      evaluations
      human_feedback
      final_artifacts

各executionに記録するもの:

- step type
- parameters
- input artifact IDs
- output artifact IDs
- model/checkpoint
- random seed
- duration / error

長期的には、このTrajectory storeが最も重要な学習datasetになります。

## 10. 推奨研究段階

### R0: Runtime validation
mock stepでGUI <-> DSL round trip、Artifact tracking、branch、score、provenanceを検証します。

### R1: Existing-model pipeline
pretrained componentのみで prompt -> sketch/structure -> controlled color -> evaluate -> select を動かします。

### R2: Evaluator study
ImageReward、PickScore、HPSv2、MPS/VisionReward、semantic metric、人間評価の一致・不一致を比較します。

### R3: Trajectory dataset
中間Artifact、accepted/rejected branch、Evaluator output、人間choice、Workflow configを収集します。

### R4: 1 Operatorだけ学習
例: sketch selector、refinement policy、color operator、stroke predictor。

### R5: Workflow optimization
operation order、branch count、candidate count、Evaluator配置、model/operator choice、refinement countを探索します。

初期はRLより先に次を使います。

- grid search
- random search
- Bayesian optimization
- evolutionary search

## 11. 主研究仮説

Workflow W は初期状態からTrajectoryを生成します。

    tau = (s0, a0, s1, a1, ..., sT)

各actionはoperator invocationです。

目的はfinal-image rewardだけではありません。

- final quality
- prompt alignment
- controllability
- editability
- efficiency
- diversity
- intermediate consistency

したがってImage Drawerはmulti-objective procedural-generation systemとして扱います。

## 12. 現時点の推奨方針

1. DSL / Runtimeをmodel-agnosticに保つ。
2. ArtifactとTrajectoryを第一級objectにする。
3. Scoreをmulti-dimensionalかつEvaluator-specificにする。
4. plugin-style operator registryを使う。
5. 既存generation/evaluation methodを先に統合する。
6. 学習algorithmを固定する前にTrajectoryを収集する。
7. RLより先にBest-of-N / rankingを使う。
8. Workflow optimizationを独立research layerとして扱う。

AI componentは速く変化します。Image Drawerで長く価値を持つ部分は、Workflow、provenance、evaluation、experiment infrastructureです。
