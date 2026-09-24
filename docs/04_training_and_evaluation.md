# 学習と評価

## 1. 初期学習戦略

最初から完全なEnd-to-End differentiable構成を要求しません。

実用的な初期loop:

    1. N個のcandidateを生成
    2. candidateを評価
    3. 上位K個を保持
    4. 良かった制作手順・履歴を保存
    5. 保存例からtrain / fine-tune
    6. 再度生成

これは rejection sampling + imitation / fine-tuning に近く、最初からreinforcement learningを導入するよりdebugしやすい方式です。

## 2. 評価段階

評価はWorkflowの複数段階で行えます。

### Sketch評価

候補component:

- composition
- silhouette / readability
- anatomy / structure
- prompt alignment
- line economy

### Color評価

候補component:

- color harmony
- foreground/background separation
- lighting consistency
- material consistency
- prompt alignment

### Final評価

候補component:

- overall preference
- prompt alignment
- composition
- anatomy / structure
- line quality
- color quality
- detail / coherence

## 3. Score表現

scoreはstructured dataとして保持します。

    Score {
      overall: 0.84
      components: {
        composition: 0.91
        line_quality: 0.79
        color_quality: 0.87
      }
    }

aggregate scoreは設定可能なweightから計算できますが、元componentは失わないようにします。

## 4. Human evaluation

人間の選好もEvaluatorの1つとして扱います。

GUI操作候補:

- A/B選択
- approve
- reject
- 簡易rating
- 理由/category annotation

人間の判断は自動Evaluatorのscoreと並べて保存します。

## 5. Selection

EvaluationとSelectionを分離します。

例:

- best score
- top-k
- 上位candidateからのweighted random
- diversity-aware selection
- 複数componentに対するPareto selection

MVPではbest / top-kのみ必須とします。

## 6. 保存対象

将来の学習に利用するため、final imageだけでなく次を保存します。

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
- final acceptance / rejection

これにより完成画像だけでなく制作手順レベルの学習・分析が可能になります。

## 7. 将来戦略

基盤完成後の候補:

- reinforcement learning
- evolutionary search
- beam search
- workflow-order search
- learned evaluator
- preference model
- self-correction loop
- adaptive branching / candidate count

これらは最初のMVP必須要件ではありません。
