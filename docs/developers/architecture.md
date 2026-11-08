> **対象読者:** Image Drawerの各モジュールを実装する人
> **目的:** 学習画像・抽出器・描画Policy・固定工程・Renderer・Evaluatorの責務を分ける。
> **状態:** 目標アーキテクチャ。下記の新コンポーネントが実装済みであることを意味しない。

# Image Drawerのアーキテクチャ

## 1. 全体像

**学習素材は画像だけ、生成時の外部入力は文章だけ**。タグ・説明文・prompt・権利情報などの別の入力ファイルは要求しない。学習の内部では文章を自動生成してTextEncoderへ渡す。描く順序はラフ→下書き→本描き→基本色→陰影・仕上げ→修正・確認→完成で固定するが、その中の色・ブラシ・筆跡はAI自身が学習する。

```text
学習画像だけ
    │
    ├── ImageDataset（画像の検査/hash/split）
    │       │
    │       └── TagExtractor/CaptionExtractor（ユーザー選択式）
    │                       │
    │                  学習用Prompt
    │                       │
    │                   TextEncoder
    │                       │  文章の意味を表す内部条件
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
| `ImageDataset` | 画像だけの登録、正規化、重複・split・version管理。内部ID/ハッシュは自動生成 | 画像以外のファイルや権利情報を利用者に要求する |
| `TagExtractor/CaptionExtractor` | 選択されたモデルで画像からタグと文章を抽出 | 利用者にprompt/タグを書かせることを必須にする |
| `ImageTrainingPreprocessor` | CaptionとTaggerを必ず別系統で実行し、任意段数のモデル切替・タグ分類・整形を調停 | Captionを得ただけでTaggerを実行済みとする |
| `PromptBuilder` | 画像からの事実を人間の自然な短文・標準文・詳細文へ整形し、学習分布を制御する | 利用者に別途テキストを書かせる |
| `TextEncoder` | プロンプトを描画Policyが使う**意味特徴**へ変換。学習/生成で同じ条件形式を使用 | 学習時だけ正解画像の特徴を紛れ込ませる |
| `DrawingPolicy` | prompt＋自分のstateから次の合法Actionを予測 | 元画像やその特徴を観測する |
| `FixedStageController` | 全7工程の必須順序、工程別の書込可能レイヤー、レビュー理由、レイヤー表示/上下、終了/予算管理 | 筆跡や位置を固定Scriptで決める／Stageをskippingする |
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

**画像から文章とタグを独立抽出し任意の登録モデルへ切り替える構成**：[高精度Caption＋タグ分類・自由登録](prompt_extraction_v1.md)。`image_drawer/prompting/`に登録・CLI・アダプタ・タグ整形の基盤があり、実モデル導入・Dataset/Trainer/GUI接続は未完成。

**新しい固定工程Controllerの動作**：[工程移行・レイヤー・atomic記録](stage_controller_contract_v2.md)。**画像による工程別採点**：[Stage Reward](stage_reward_v2.md)。Controllerは対象画像を観測せず、Trainerは試行後の報酬からPolicyを更新する。

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
| 文章の用意 | システムが選択済みTag/Caption抽出器とPromptBuilderで自動生成 | ユーザー |
| TextEncoderの入力 | システム内部の生成プロンプト | ユーザー入力プロンプト |
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

## 7. 「モデル」の範囲と入出力

Image Drawerの**モデル**は、絵を描くPolicyだけではありません。**読み出し部分から描画までの一連の流れ全体**を指します。

```text
[読み出し]  画像 → Captioner / Tagger → 整形(PromptBuilder) → TextEncoder
                                                                ↓
[描画]                         Policy → 固定7工程 → Renderer → 画像 + Episode
```

| | 学習 | 推論（今回の対象） | 推論（将来。最初から対応してもよい） |
| --- | --- | --- | --- |
| 必須の入力 | 画像 | 文章 | 画像 + 文章 |
| 任意の入力 | プロンプト（あれば読み込める） | なし | なし |
| 出力 | checkpoint・Trial・Reward | 画像・Episode | 画像・Episode |

| 項目 | 内容 |
| --- | --- |
| 目標 | 入出力の形はSDXLと同じ（txt2imgと、画像と文章を組み合わせる入力）です。同じ入出力でそれ以上の強さを出すことが目標で、現時点では達成を意味しません。達成の主張は、`ClaimGate`と汎化評価（[評価プロトコル](design_evaluation_protocol.md)）の判定の範囲に限ります。 |
| 学習の入力 | ユーザーが用意するのは画像だけです。プロンプトが付いていれば読み込めますが、必須ではありません。プロンプトと抽出結果の優先順位・併用の規則は未決事項です。 |
| 評価の単位 | 読み出し（Captioner・Tagger・整形・TextEncoder）もモデルの一部です。部品ごとの診断（[学習設計 12章](design_trainer.md)）は原因の切り分けに使い、評価の単位は端から端（画像または文章から、描画された画像まで）です。強いCaptioner・Taggerを入れてもPolicyが文章条件を使わなければ精度は上がらず、逆も同じです。 |
| 実行方法 | 複数の部品（モデル）は順番に1つずつ動かします（ロード、実行、解放）。同時に動かす必要はありません。メモリはオフロードを許容します。複数の抽出結果の組み合わせとフォールバックも許容します（[プロンプト抽出](prompt_extraction_v1.md)）。 |
| 対象 | イラスト（人物・風景）です。写真などイラスト以外は、設計の対象外です。 |
| 将来の画像+文章入力 | 利用者が与える入力です。学習時に目標画像をPolicyへ見せない規則（5節）とは別で、Policyの観測へ入力画像の条件を足す設計が必要です（未設計）。 |
