> **対象読者:** Image Drawerの開発優先順位を決める人、Issueを作る人
> **目的:** 画像だけで「描いて学ぶ」txt2imgを完成させるまでの作業順序・合格条件を決める。
> **状態:** 設計改訂案。実装済みと未実装を混同しない。

# 実装計画：画像だけで学び、固定工程で描く

## 1. 最初に完成させるもの

> 学習に必要な外部入力は画像のみとする。
> ユーザーおよび開発者は、画像以外の補助データを用意する必要はない。
> タグ・説明文・プロンプトは、必要に応じてシステム側で自動抽出する。

内部ではタグ・説明文から学習用プロンプトを生成し、**TextEncoderでDrawingPolicyへの文章条件を作る**。抽出器はユーザーが切り替えられる。人間の筆跡は不要。

**生成時は文章だけ**。AIが白紙から**ラフ→下書き→本描き→基本色→陰影・仕上げ→修正・確認→完成**の全7工程で実際に描く。各工程内でどのブラシ・色・筆圧・座標を使うかはAIが学習する。

```text
学習：画像を登録 → 自動Tag/Caption → AIが段階ごとに描く
      → 評価器だけが元画像と比較 → 報酬でPolicy更新

生成：文章だけ → 固定工程の順序でAIがActionを実行
      → 完成画像＋ReplayできるAction履歴
```

人間の実演を用いた模倣学習が先にないと始められない構成は禁止。Part Bankや直接画像変換の代用も禁止。

## 2. 正本と責務

| 何を知りたいか | 開く文書 |
| --- | --- |
| 製品の目的 | [コンセプト](concept.md) |
| **画像だけで描いて学ぶ方式** | [画像-only学習仕様](../developers/image_only_learning_v1.md) |
| **ラフ→下書き→本描きの固定順序** | [固定制作工程](../developers/fixed_drawing_workflow_v2.md) |
| タグ・prompt抽出と画像Dataset | [データ品質](../developers/drawing_data_quality_v1.md) |
| AIが選ぶAction・学習評価 | [DrawingPolicy](../developers/drawing_policy_accuracy_v1.md) |
| 色・ブラシ・筆圧 | [ブラシv2](../developers/brush_presets_v2.md)と[段階表](../developers/brush_control_levels_v2.md) |
| 旧v1のReplayと画素テスト | [制作過程v1](../developers/drawing_process_v1.md)と[受け入れ試験](../developers/drawing_process_acceptance.md) |

## 3. 実装ステージ

### S0：既存コードと設計の整合監査

**成果物**：旧human模倣学習前提の箇所、既存Part/Workflowの機能、描画Engineの未実装箇所を洗い出した表。親Issueは#62。

**Done**：どの機能を再利用でき、どの契約を直すか明記。v1ゴールデン互換を維持できる設計。

### S1：決定的描画器と固定工程Controller

**成果物**：ModelなしでActionを適用できるDrawingEngine、v2の12ブラシ、固定256色、レイヤー、サブピクセル・筆圧・不透明度、stage controller、Episode保存/Replay。

- 旧v1の整数ブラシは**テスト用互換経路**で維持する。
- 新v2のStageは**rough→sketch→final_drawing→base_color→shading→refine**。必須工程を飛ばせない。
- 本描きは必ず黒の線画ではなく、画風に応じた本番描画。**全7工程はスキップ不可**。モノクロ・フラット作品の色/影と、修正不要の場合は、対応工程へ入場し**理由付きレビュー**で通過できる。
- 工程ごとに描ける**レイヤー役割**をControllerで制限。本描きの線を基本色が覆わない固定の重ね順、ラフ/下書きの非表示、修正ステージからの可視レイヤー編集を[工程仕様](../developers/fixed_drawing_workflow_v2.md)どおりに実装する。
- 筆の動かし方は固定Scriptにしない。

**StageControllerの実装必読**：[操作許可・工程移行の原子性・レイヤー・レビュー](../developers/stage_controller_contract_v2.md)。進行・レイヤーリセット・表示変更・Controllerが発行するfinishが、Replay可能な一まとまりのイベントとして永続化されること。

**Done**：v1のgoldenが全合格し、v2の全12ブラシの画素golden、工程順序、必須Stageでの有効stroke、保存/Replayが実装されている。現時点でv2の数値カーネルとGoldenは**未完成**。

### S2：画像だけのDataset登録と抽出器選択

**精度面の正本**：[画像から高精度なプロンプトを生成する設計](../developers/prompt_extraction_v1.md)。高精度VLMを最優先し、画像内容の説明が拒否・空文・検証不合格となったら、設定したQwen-VLなどへ自動フォールバックする。**人間の短い文章と詳細キャプションを同じTextEncoderで学習する**ため、短文・標準文・詳細文を混合する。`image_drawer/prompting/`の単体実装とテストを先行させ、Dataset/GUIへの接続は後続。

**成果物**：画像だけの登録→自動タグ・説明文/プロンプト生成→TextEncoder→内部ハッシュ・分割・キャッシュ管理。

- `TagExtractor`と`CaptionExtractor`を利用者が**別々に切り替えられる**。ユーザーには画像以外のデータ提供を求めない。
- `PromptBuilder`が非空の内部学習プロンプトを作り、`TextEncoder`が意味特徴へ変換する。どちらもシステム内部で動かし、モデルの選択・バージョン・生成結果を記録する。
- GUI/CLI/設定ファイルから同じregistryとversionを使用する。
- 画像の最終画素は**学習時のEvaluatorのみに**提供し、Policyの観測を汚染しない。
- 抽出器やモデルversionを変更したらcacheを無効化し、新実験として追跡する。

**Done**：画像だけで登録と内部プロンプト生成・TextEncoder入力が成功し、抽出器切替・高精度モデルの拒否時Qwen切替・構造化出力の検証・空文の失敗処理・train/test漏洩防止がテストできる。実モデルを使った画像内容の正確さと未見の人間プロンプトへの追従改善は別途計測する。人間の実演・補助ファイルは不要。

### S3：AI自身の描画試行による学習

**成果物**：Policyが画像だけのDatasetに対して、文章＋白紙からActionを生成し、**実際に描画した結果**からRewardを受けて更新するTrainer。

- Stage順序はS1のControllerで固定。各Stageの内部ActionをPolicyが学習する。
- Rewardは形・輪郭・色・知覚差・文章一致・進捗・Actionコストなどを別成分で記録。
- **Policy-gradientなどの報酬最適化**をベースラインとして比較。人間の教師Actionは存在しない。
- 更新前後のcheckpoint、Trial、Reward、失敗・予算超過を記録する。

**Done**：小さな画像セットで学習前より生成画像の品質が改善し、未見画像へも少なくとも測定可能な改善を示す。元画像をPolicyに見せず、固定Script・コピー・img2imgに逃げない。**成功は保証できないので、達しなければ未完成と表示**。

### S4：txt2img推論・GUI・Workflow連携

**成果物**：ユーザーが文章だけを入力し、AIがラフ→下書き→本描き→…→完成まで自律描画するUI/CLI。

- モデルの途中レイヤー/筆跡、段階の進行、最後の画像を見られる。
- Datasetの画像登録と抽出器の選択UIを提供する（教師用筆跡のアップロードUIは必須ではない）。
- WorkflowのStepは**工程Controllerを呼ぶ上位単位**。Stepに置き換えて筆跡が消えたら不合格。

**Done**：文章だけのtxt2imgと全Action Replayが成立する。モデルが参照画像を必須としない。

### S5：未見画像・未見文章・画風への汎化評価

**成果物**：train/test分割、複数seed、画像品質と固定工程の遵守率、計算コスト、失敗集計、独立の人間評価。

**Done**：同じ条件で学習前後を再現し、未見画像・未見promptで改善を確認できる。安定した精度向上が見えなければ**高精度と公表しない**。

## 4. 旧M0〜M14の位置づけ

Workflow Runtime、DSL、Artifact、Trajectory、CI、Part Bankなどの既存資産は**補助機能として再利用**可能。ただし、以前のPart Bankの採用・COMPOSEの動作を「描画学習の主経路が実装済み」とみなさない。旧M0〜M14のIssue・PR履歴は残し、#62を設計修正の親Issueとする。

人間の入力をRecordする制作GUIは**任意の将来拡張**。開発順序のS2・S3へ戻して**必須**にしてはいけない。

## 5. すべてのIssueのDone

- 入力・出力・状態遷移・Actionの出所・失敗・テスト・未実装範囲を明確にする。
- 固定256色とブラシ分離、段階ID、固定工程、学習時のtarget隔離を壊さない。
- 日本語が仕様文書の正本。初見の人が例から読める入口を置く。
- PR運用・CI（pytest、Ruff、build、wheel、実動作スモーク）を維持。
- 設計書が書けただけ、CIが通っただけでは**モデルが絵を描いて学習できたことにしない**。
