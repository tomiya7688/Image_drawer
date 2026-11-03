> **対象読者:** Image Drawerの各モジュールを実装する人
> **目的:** 学習画像・抽出器・描画Policy・固定工程・Renderer・Evaluatorの責務を分ける。
> **状態:** 目標アーキテクチャ。下記の新コンポーネントが実装済みであることを意味しない。

# Image Drawerのアーキテクチャ

## 1. 全体像

**学習素材は画像だけ、生成時の外部入力は文章だけ**。描く順序はラフ→下書き→本描き→基本色→陰影→仕上げで固定するが、その中の色・ブラシ・筆跡はAI自身が学習する。

```text
学習画像だけ
    │
    ├── ImageDataset（画像の検査/rights/hash/split）
    │       │
    │       └── TagExtractor/CaptionExtractor（ユーザー選択式）
    │                       │
    │                  学習用Prompt
    │                       │
    │                 DrawingPolicy（学習対象）
    │                       │  prompt＋自分のCanvas/履歴のみ
    │                FixedStageController
    │                       │  ラフ→下書き→本描き→…→仕上げ
    │                DrawingEngine/Renderer
    │                       │  実際の一筆を実行
    │                生成Canvas＋DrawingEpisode
    │                       │
    │      元画像──→ Evaluator/Reward ←──生成Canvas
    │                       │
    │                    Trainer（重み更新）
    └───────────────────────┘

生成モード：文章 → DrawingPolicy → Controller → Engine → 完成画像＋Episode
```

**元画像を見られるのは学習用のEvaluatorだけ**。Policyへの観測入力に目標画像、差分画像、目標画像のembeddingを渡したら不合格。

## 2. モジュールの責務

| モジュール | 責務 | 絶対にしないこと |
| --- | --- | --- |
| `ImageDataset` | 画像だけの登録、正規化、split、権利・重複・version管理 | 実演Actionを教材の必須入力にする |
| `TagExtractor/CaptionExtractor` | 選択されたモデルで画像からタグと文章を抽出 | 利用者にprompt/タグを書かせることを必須にする |
| `PromptBuilder` | タグと説明文をversion付きで学習promptへ整形 | 抽出器を勝手に切り替えキャッシュを混ぜる |
| `DrawingPolicy` | prompt＋自分のstateから次の合法Actionを予測 | 元画像やその特徴を観測する |
| `FixedStageController` | ラフ→下書き→本描きの必須順序、stage進行/skip/予算管理 | 筆跡や位置を固定Scriptで決める |
| `DrawingEngine` | Actionを画素・レイヤー・ブラシで実行、拒否を返す | Policyの代わりに色や構図を独断で選ぶ |
| `Recorder / Replay` | **モデルが実際に描いた**Action、stage、RGBA中間状態を記録/再生 | 元画像からActionを捏造する |
| `Evaluator / Reward` | 学習時に元画像と生成画像を比較、形・色・文章一致・コストの報酬を出す | 元画像・差分画像をPolicyに観測させる |
| `Trainer` | Rewardを用いてPolicyのパラメータを更新、checkpointとtrialを保存 | 選択だけ行い学習したと称する |
| `Workflow Runtime` | 上記の実行をArtifact/Trajectoryへ結び付ける | 完成画像だけをActionEpisodeに偽装する |
| `Part Bank` | 任意の補助素材参照 | Part配置を筆跡の代わりにする |

## 3. DrawingEpisode とWorkflowを混同しない

- **DrawingAction**：色変更、ブラシ切替、サイズ変更、筆で描く、消す、Stageを進める、完成するなどの1操作。
- **DrawingState**：自分が今まで描いたキャンバス、透明レイヤー、工具状態、現在のStage。
- **DrawingEpisode**：空白から完成までの**実際のActionの列**。推論・学習のtrialで同じ形式。
- **WorkflowSpec**：学習・生成・評価の上位処理とデータの受け渡し。Action一件ごとの代わりではない。

**v1互換性契約**：[制作過程v1](drawing_process_v1.md)。**新しい固定工程・画像only学習**：[固定工程v2](fixed_drawing_workflow_v2.md)・[画像だけで学ぶ](image_only_learning_v1.md)。v2のRendererとEpisodeはまだ実装されていない。

## 4. 既存コードの入口と再利用

現在の主要な既存資産：

- `image_drawer/runtime/engine.py`：Workflow RuntimeとTrajectory。
- `image_drawer/core/models.py`：ArtifactやScore、Trajectory等の共通型。
- `image_drawer/steps/`：既存StepとPart合成など。
- `image_drawer/dsl/`：Workflow DSLのparser/serializer。

これらは**新しい学習・描画器が既に実装されたことを意味しない**。既存のCI、Artifact URI/lineage、実験ログを利用し、DrawingAction/Stage/Rewardの責務を新しいモジュールに切り出す。

## 5. 学習時と生成時の違い

| | 学習 | 生成（txt2img） |
| --- | --- | --- |
| ユーザーの必須入力 | **画像だけ** | **文章だけ** |
| 文章の用意 | 選択したTag/Caption抽出器 | ユーザー |
| 元画像を読む | **Evaluatorが読み採点** | 不要・存在しない |
| Policyが見る | 文章＋自分の途中状態 | 同じ |
| 固定制作工程 | 同じ | 同じ |
| モデル更新 | **報酬を使って更新** | 更新しない |
| 出力 | Trial・Reward・checkpoint・Episode | 画像・Episode |

学習時だけの元画像を使って推論で直接なぞってはいけない。

## 6. 実装の優先順位と合格条件

1. **Engine/Stage Controller**：12ブラシ、固定256色、ラフ→下書き→本描きの順序と各工程のstroke、Replayが成立する。
2. **ImageDataset/Extractor**：画像だけで学習に登録し、Tagger/Captionerを切り替え、versionとcacheを整合させる。
3. **Policy/Trainer**：実際に描いた画像との比較でパラメータを更新し、学習前後の改善を測る。
4. **txt2img**：文章だけを入力し、画像とEpisodeを出力。元画像なしで動作する。

詳しい実装順は[実装計画](../kadoka/implementation_plan.md)。文書・ビルドだけが成功しても学習モデルや高品質なイラストができたとは呼ばない。
