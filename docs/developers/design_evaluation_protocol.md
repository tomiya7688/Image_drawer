> **対象読者:** 評価基盤（分割、指標、報告書）を実装する開発者、評価結果を公表する人、レビュー担当
>
> **目的:** Issue #72（S5）の実装設計を定める。同じ条件で学習前後を再現し、未見画像・未見promptで改善を確認する手順と、「高精度」と公表してよい条件・してはいけない条件を、検証できる形で決める。
>
> **状態:** 設計書。`image-drawer-evaluate`、指標、報告書、ゲートは`image_drawer/generalization/`に実装済みです（`useful_stroke_rate`の実行配線、OpenCLIP系の独立指標を除く。`prompt_shuffle_delta`は`run --shuffle-prompts`、`probe_hit_rate`は`run --probe`、`nearest_train_hit_rate`は`memorization-tau`と`run --memorization-tau`、`generalization_gap`は`run --split train --max-items N`と`report --train-run`で配線済み）。実checkpointでの評価は未実測です。Policy、Trainer、Reward、Controllerも未実装のため、現時点では学習の効果を測定した結果は存在しない。文書が揃ったこと、CIが通ることは、モデルが絵を描けること、学習で上達したことを意味しない。

# 汎化評価プロトコルの実装設計（S5）

## 1. 範囲と原則

### 1.1 この文書が決めること

- train/validation/testの使い分けと、固定したtestを調整に使わないための仕組み
- 複数seedの種類と、集計・信頼区間の計算
- 指標（画像品質、固定工程の遵守率、有効stroke率、拒否率、計算コスト、失敗集計）の定義と計算式
- 独立した人間評価の手順
- 報告書の形式、公表してよい条件・してはいけない条件、再現コマンド
- 受け入れテストと、PRの分け方

### 1.2 この文書が決めないこと

| 項目 | 決める文書 |
| --- | --- |
| Rewardの成分と重み | [工程別Reward](stage_reward_v2.md)、S3（数値は未freeze） |
| `DrawingPolicy`、`Trainer`、checkpoint | `design_trainer.md` |
| 推論のStep・CLI | [推論・GUI設計](design_inference_gui.md) |
| Datasetの登録とsplitの割り当て | S2（`image_drawer/dataset/`、`image_dataset.md`） |

### 1.3 固定する共通契約

| 契約 | 形 |
| --- | --- |
| 工程Controller | `StageController(engine, intent: StyleIntent, budget: Budget)`、`propose(action) -> ProposalResult(accepted, events, reason, attempt)`、`episode() -> DrawingEpisode` |
| 再生 | `replay(episode) -> ReplayResult(image, layers, state)` |
| Dataset | `iter_samples(dataset_dir, split) -> Iterator[(PolicyInput(image_id, prompt), RewardTarget(image_id, image_path, image_sha256))]` |
| 学習側 | `DrawingPolicy`（目標画像を受け取らない）、`RewardEvaluator`、`Trainer` |
| 7工程 | `rough, sketch, final_drawing, base_color, shading, refine, finish` |

### 1.4 原則

1. **評価はPolicyの外側で行います。** 目標画像は評価器だけが読みます。生成の経路（[推論設計](design_inference_gui.md)）は評価の有無で変わりません。
2. **失敗を分母から除かない。** 未完成、拒否、例外、抽出失敗はすべて集計に残します。
3. **事前に決める。** 指標の定義、閾値、分析方法、使うseedとtest集合は、結果を見る前に固定して記録します。結果を見てからの変更は、その評価を無効にします。
4. **主張の範囲を測定した範囲に限る。** 結果は、使ったデータの種類・件数・解像度・seed数の範囲だけの主張になります。
5. **単一の「精度」を作らない。** Rewardの成分ごと、工程の遵守、コスト、人間評価を別々に報告します。

## 2. 比較する条件

| 条件ID | 内容 | 目的 |
| --- | --- | --- |
| `before` | 学習前のcheckpoint。`after`と同じアーキテクチャ、同じ初期化seed、学習0ステップ | 学習の効果の基準 |
| `after` | 学習後のcheckpoint | 評価の対象 |
| `random_valid` | 合法なActionを一様に選ぶPolicy（Controllerを通す） | 「工程を守るだけ」「でたらめに描くだけ」との比較 |
| `blank` | 何も描かない白紙の画像（参照用の値） | 失敗時の得点の基準（6.7節）と、指標の下限の確認 |

学習の効果は`after`と`before`の差で測ります。`random_valid`は、Controllerが規則を強制するだけでは品質が上がらないことを確認するための対照です。

## 3. データの分け方

### 3.1 split

S2のDatasetは、近重複のグループ単位で`train`（約80%）、`validation`（約10%）、`test`（約10%）に決定的に割り当てます。Issue #72は「train/test」と書いていますが、本書は`validation`を含む3分割を使います（11節の差分1）。

| split | 使ってよい用途 | 使ってはいけない用途 |
| --- | --- | --- |
| `train` | 学習、抽出器の動作確認 | 汎化の主張の根拠 |
| `validation` | ハイパーパラメータ、Reward重み、抽出器の選択、checkpointの選択、停止条件の決定 | 公表する汎化結果 |
| `test` | 最終評価（3.4節の手順で1回） | 上記の調整のすべて |

[Policy精度仕様](drawing_policy_accuracy_v1.md)の「比較に使う評価画像はPolicyの学習やReward重みの調整に使用しない」に従い、Reward重みの調整は`validation`だけで行います。

### 3.2 未見画像と未見prompt

| 用語 | 定義 |
| --- | --- |
| 未見画像 | `test`の画像。その`group_id`が`train`にも`validation`にも存在しない（S2のグループ分割が保証。評価開始時に再検査する） |
| 未見prompt（厳密） | 正規化後の文字列が、`train`と`validation`のすべての学習用promptに一致しないprompt。`test`の画像から、**学習と同じ固定の抽出器・PromptBuilder**で作る |
| 構成的な未見prompt | 単語は`train`に現れるが、評価で変える属性の組（対象と色、対象と個数など）が`train`に現れないprompt。開発者が書く評価用テンプレート`prompt_probes_v1.json`から作る（色・個数・位置の入れ替え） |
| 既見prompt | `train`の画像から作ったprompt。記憶の確認（8節）にだけ使う |

評価用テンプレートは評価資産であり、学習の入力ではありません。ユーザーに追加の入力を求める根拠にもしません。

### 3.3 凍結した評価集合（`eval_set`）

S2の制限として、新しい画像の追加で既存画像の`group_id`と`split`が変わることがあります。評価の途中でtestが変わらないように、評価を始める前にtestを**スナップショット**します。

`eval_set_<id>.json`の内容は次のとおりです。

| キー | 内容 |
| --- | --- |
| `eval_set_id`、`created_at` | 識別子と作成日時 |
| `dataset_hash`、`split_seed` | S2のmanifestの値 |
| `items[]` | `image_id`、`image_sha256`、`group_id`、`prompt`（標準の1文）、`prompt_variants`（短・標準・詳細。#78統合後）、`extractor_ids`、`prompt_builder_version` |
| `probes[]` | 構成的な未見promptの`probe_id`、基準のitem、変更する属性 |
| `train_group_ids_sha256` | 分離の確認用 |
| `eval_set_sha256` | 上記全体のハッシュ |

`items`はimage_id順、`probes`はテンプレートの記載順で保存し、この順序も`eval_set_sha256`の対象です（順序を変えるとハッシュが変わります）。`prompt_seen_in_train`は`prompt`と同様に空白と大文字小文字を揃えた文字列（`normalize_prompt`）で判定し、`items[].prompt`自体は元の文字列を保持します。

構成的な未見promptのテンプレートは、`probe_id`、`base_image_id`、`attribute`、`original`、`replacement`を持ちます。`original`が英数字と下線だけの語なら、前後が英数字・下線でない位置だけに一致させ（`cat`は`certificate`に一致しない）、日本語など空白で語を区切らない文字列は大文字小文字を区別する完全一致の部分文字列とします。どちらも基準itemのpromptにちょうど1回だけ現れる必要があり、そうでなければ拒否します。

Datasetが後で変わっても、`eval_set`は変更しません。新しい画像を加えた評価は、新しい`eval_set_id`で行います。

### 3.4 固定したtestを調整に使わない規則

1. **禁止**: testの目標画像、抽出文、得点、人間評価を、学習、抽出器の選択、Reward重み、ハイパーパラメータ、checkpointの選択、停止条件、評価用テンプレートの修正に使うこと。
2. **凍結宣言**: test評価の前に`freeze_manifest.json`を保存します。内容は、`after`と`before`のcheckpointのSHA-256、コードのgit commit、Rewardと指標のversionと設定のSHA-256、`eval_set_sha256`、推論seed、`max_actions`、分析計画のSHA-256です。
3. **1回だけ実行**: `image-drawer-evaluate run --split test`は`--freeze-manifest`を必須とし、無ければ拒否します。実行は`test_ledger.jsonl`（追記専用）に、`eval_set_id`、凍結宣言のSHA-256、日時を記録します。凍結宣言の`eval_set_id`と`eval_set_sha256`が評価集合と一致しなければ拒否します。帳簿の読込から追記までは、ロックファイル（`<帳簿>.lock`）で排他にします。壊れた行や必須キーの欠けた行は、`LedgerError`（終了コード2）にします。
4. **再実行の扱い**: 同じ凍結宣言の再実行は、決定性の確認だけに使い、結果の選択に使いません。**同じ`eval_set_id`に対して別の凍結宣言でtestを実行すると、帳簿が`retest`を付け、その`eval_set`の結果は公表条件（10節）を満たしません**。新しいtest集合が必要です。
5. **汚染の扱い**: 規則に反してtestの結果を調整に使った場合、その`eval_set`は`contaminated`として記録し、以後の公表に使いません。

`validation`の実行には凍結宣言を要求しません。調整に使えるためです。

## 4. 複数seed

seedを3種類に分け、混ぜません。

| seed | 意味 | 値 |
| --- | --- | --- |
| 学習seed | 初期化と学習の乱数。独立に学習し直す | 公表条件は3個以上（暫定）。`after`ごとに記録 |
| 推論seed | `DRAW_EPISODE`の`seed`（grainなどの乱数） | 1つの画像・条件につき`K`個。`K=5`（暫定）。`seed(image_id, k) = SHA-256("{eval_set_id}:{image_id}:{k}")`の先頭32ビット。`before`と`after`で同じ値を使う（対応づけ） |
| 評価seed | ブートストラップの乱数 | 固定値を報告書に記録 |

推論の決定性は、同じハードウェア・同じ実装でのみ要求します（[受け入れ試験](drawing_process_acceptance.md)の0節6）。ハードウェアが異なる実行どうしの`before`/`after`比較は行いません。`hardware_id`を報告書に記録します。

## 5. 学習前後の比較手順

1. `validation`で学習・調整を終え、`after`のcheckpointを決める。
2. 凍結宣言を保存する（3.4節）。
3. 同じ`eval_set`、同じ推論seed、同じ`max_actions`、同じcanvasで、`before`、`after`、`random_valid`の各条件について`DRAW_EPISODE`を実行する。学習seedが複数ある場合は、学習seedごとに`before`/`after`を作る。
4. 各生成について、Episode、Image、`attempts.json`、計算コストを保存する。
5. 指標（6節）を計算する。目標画像を読むのは評価器だけにする。
6. 差と信頼区間（7節）を計算し、報告書（9節）を作る。

同じ手順を`train`の部分集合（`test`と同数）にも適用し、`train`と`test`の改善の差（8節）を測ります。

## 6. 指標

記法: 項目`i`（画像と、その学習用prompt）、推論seed`k`、条件`c`の生成結果を`R(c,i,k)`とします。`T_i`は目標画像、`I(c,i,k)`は完成画像（RGBAを白背景へ合成したRGB）です。生成に失敗した場合の扱いは6.7節に従います。

### 6.1 画像品質（Rewardの成分別）

- `RewardEvaluator`（S3）が定義する成分`m_1..m_n`（例: 粗い輪郭、中程度の輪郭、低解像度RGB、色分布、明るさ分布）を、**成分ごとに別の指標**として報告します。重み付き合計を見出しの数値にしません。
- 計算式: `score_j(c,i,k) = m_j(I(c,i,k), T_i)`（0から1、大きいほど良い）。入力は完成画像と目標画像、出力はスカラー。目標画像のcanvasへの合わせ方は`RewardEvaluator`の正規化に従う。
- 工程ごとの途中の得点は、Replayの工程退出時のフレームから`stage_score_s(c,i,k)`を計算して、工程別の改善（`stage_score_after − stage_score_before`）を診断として報告します。
- **学習に使っていない独立指標**（`heldout_*`）を必ず併記します。Rewardを直接最適化した結果は、Rewardが上がっても絵が良くなった根拠にならないためです。候補は、(a)学習のRewardと異なる画像特徴・解像度で計算する指標（例: 64×64でのSSIM）、(b)OpenCLIPの画像間コサイン類似度（既存の`openclip` extraを利用）、(c)画像と文章のCLIP一致度です。採用する指標は実装PRで決め、`metrics_v1`として凍結します（未決事項E1）。
- `reward_gain_without_heldout_gain`（Rewardの成分は有意に改善したが、独立指標がどれも改善しない）を警告として報告します。

### 6.2 固定工程の遵守率

Controllerは規則を満たさないActionを受理しないため、**受理済みActionの遵守率は構造上100%**になります。この数値を成果として報告しません。代わりに次を測ります。

| 指標 | 式 | 意味 |
| --- | --- | --- |
| `completion_rate` | 完成した生成数 ÷ 全生成数 | 最後まで`finish`に達した割合。Draftは失敗として分母に残す |
| `proposal_legal_rate` | 受理された提案数 ÷ 全提案数（受理＋拒否） | Policyが合法なActionを出す率 |
| `stage_violation_rate` | 工程規則による拒否（`skip_stage`、後戻り、書込不可レイヤー、必要な筆跡のない移行など）の提案数 ÷ 全提案数 | 工程の理解 |
| `audit_pass_rate` | `StageAuditor`が合格とした生成数 ÷ 完成した生成数 | 独立した検証の結果 |
| `auditor_controller_disagreements` | 監査器とControllerの判定が食い違った件数 | 0でなければバグ |
| `review_only_rate[stage]` | 工程`s`をレビューのみで通過した生成数 ÷ 完成数 | `base_color`/`shading`/`refine`の「描かない」の頻度。多すぎる場合は早すぎる完成の疑い |

`StageAuditor`は、保存された`DrawingEpisode`だけから、次を**Controllerのログを使わずに**再計算します。(a)7工程の入場・退出が順に1回ずつあること、(b)`rough`/`sketch`/`final_drawing`の各専用roleに有効な筆跡が1つ以上あること、(c)`finish`が1回で最後であること、(d)許可されないレイヤーへの受理済みの描画がないこと、(e)レビュー通過した工程の理由コードが許可されたものであること、(f)`skip_stage`と`v2`での`set_phase`がないこと。

### 6.3 有効stroke率

| 指標 | 式 |
| --- | --- |
| `accepted_stroke_rate` | 受理された`stroke_v2` ÷ 提案された`stroke_v2` |
| `effective_stroke_rate` | 受理時に許可レイヤーの1画素以上のRGBAを変えた`stroke_v2` ÷ 受理された`stroke_v2` |
| `useful_stroke_rate` | 工程得点を上げた有効`stroke_v2` ÷ 有効`stroke_v2`（目標画像に依存するため評価専用。Policyには見せない） |
| `dot_stroke_rate` | 経路長が1画素未満の`stroke_v2` ÷ 受理された`stroke_v2` |
| `pixels_changed_per_required_stage` | 必須工程（`rough`/`sketch`/`final_drawing`）ごとの、有効strokeが変えた画素数の中央値 |

「1画素だけ描いて工程を通過する」回避を、`dot_stroke_rate`と`pixels_changed_per_required_stage`で検出します。有効の判定には、Replayで各Action前後のレイヤー画素を比べます。

### 6.4 拒否率

`rejection_rate = 拒否された提案数 ÷ 全提案数`を、理由コード別、工程別に分けて報告します。あわせて、`attempt_limit_hit_rate`（試行上限`max(100, 5×max_actions)`に達した生成の割合）と、連続拒否の最大長を報告します。

### 6.5 計算コスト

1生成ごとに次を記録し、中央値とp90を報告します。

| 項目 | 内容 |
| --- | --- |
| `accepted_actions`、`attempts`、`system_events` | 件数 |
| `renderer_points_used`、`renderer_area_used` | Engineの`DrawingState.points_used`/`area_used` |
| `wall_policy_seconds`、`wall_engine_seconds`、`wall_total_seconds` | 実時間（Policyの推論、Controller＋Engine、合計） |
| `peak_memory_mb` | 取得できる場合のみ。取得できなければ`null`（推定値を入れない） |
| `policy_parameter_count`、`hardware_id` | モデル規模と実行環境 |

コストの閾値は設けず、すべて開示します。品質の改善とコストの増加を並べて示し、「改善がコストに見合う」とは報告書に書きません。

### 6.6 prompt忠実度と記憶の検査

| 指標 | 式・手順 |
| --- | --- |
| `prompt_shuffle_delta` | `test`の項目`i`のpromptを、別の項目`j`（固定乱数で選ぶ）のpromptに入れ替えて生成し、`score(正しいprompt) − score(入れ替えたprompt)`の平均。0以下ならpromptを使っていない疑い |
| `probe_hit_rate` | 構成的なprobe（例: 「赤い」と「青い」）で、指定した色の面積が背景を除いて最大となる割合。色名は固定256色の最近傍で判定する。個数・位置は後続のprobeで追加。実装のprobeは`controls.PROBES`の4つ（赤い・青い・緑の・黄色い）で、目標画像は使わずpromptだけで同じ推論seedの生成を行う。完成しなかった生成は命中に数えず分母に残す |
| `nearest_train_hit_rate` | 生成画像`I`について、`train`の目標画像のうち最も近いもの（独立指標の距離で距離`d`）が閾値`τ`未満となる割合。記憶・コピーの疑いの検出（`τ`は`validation`で設定して凍結）。実装では、距離を`1 − heldout_ssim64`とし、`τ`を`image-drawer-evaluate memorization-tau`が決める（`validation`の各目標画像から`train`の目標画像までの最小距離の下側5%点。「未見の実画像でも普通は届かない近さ」までの接近をコピーの疑いと数える）。`run --memorization-tau tau.json`は各生成の最小距離を`nearest_train_distance`に記録し、`τ`と`train`の件数を`RunOutput.memorization`へ保存する。命中は距離が`τ`未満で、完成しなかった生成は命中に数えず分母に残す。`test`の結果を見て`τ`を変えない |
| `generalization_gap` | `gain_train − gain_test`（`gain`は`after − before`の平均）。実装では、`run --split train --max-items N`（画像IDの昇順で先頭N件。Nは`test`と同数にする。台帳・凍結宣言は不要で、公表の根拠にしない比較用）の結果を`report --train-run`に渡し、ITT（失敗は白紙得点）の画像ごとの平均の平均で`gain`を求める。信頼区間は未実装 |

### 6.7 失敗の集計

失敗の種類ごとに件数と割合を報告します。

`draft_budget_exhausted`、`draft_attempt_limit`、`draft_event_limit`、`draft_cancelled`、`policy_error`、`engine_error`、`timeout`、`reward_eval_error`、`replay_mismatch`、`dataset_exclusion`（抽出失敗・空prompt。評価の対象外となったtestの項目も件数を示す）。

品質指標は2通りで報告します。

- **条件付き**: 完成した生成だけの平均。
- **意図を損なわない集計（ITT）**: 失敗した生成に`blank`の得点を割り当てた、全生成の平均。**見出しの数値はITT**です。

## 7. 集計と信頼区間

### 7.1 差の定義

項目`i`の改善を、推論seedの平均で定義します。

```text
gain(i) = (1/K) Σ_k [ metric(after, i, k) − metric(before, i, k) ]
```

学習seed`r`ごとに`gain_r(i)`を計算します。

### 7.2 クラスタ・ブートストラップ

近重複の画像は独立ではないため、リサンプリングの単位を**`group_id`**にします。

1. `test`の`group_id`を復元抽出し、同じ大きさの標本を作る（10000回）。
2. 各標本で`gain`の平均を計算する。
3. 2.5%と97.5%のパーセンタイルを、95%信頼区間とする。
4. 評価seedで乱数を固定する。

見出しの結果は、学習seedごとの区間と、学習seedをプールした区間の両方を示します。あわせて次を報告します。

- `win_rate = (1/N) Σ_i [ 1(gain(i)>0) + 0.5·1(gain(i)=0) ]`
- 効果量（`gain`の平均を、`before`の標準偏差で割った値）
- 学習seedごとの方向の一致（全学習seedで`gain`の中央値が正か）

複数の指標・比較にまたがる検定は、Holm法で調整します。信頼区間が0を含む結果は「改善を確認できない」と書きます。

## 8. 独立した人間評価

人間評価は自動指標の代わりではなく、補完です。**独立**とは、評価者が(a)学習・設計・実装に関与しておらず、(b)どの画像がどの条件のものか知らされず、(c)対象の画像の作者でないことを指します。LLMによる自動判定は、人間評価として数えません（補助として別に表示する）。

### 8.1 手順

1. **計画の事前登録**: `human_eval_plan.json`に、対象のitem、比較の組、課題、除外規則、分析方法、評価者への指示文（SHA-256）を、結果を見る前に保存して凍結する。
2. **抽出**: `eval_set`から、固定乱数で`N_h`項目を選ぶ（`N_h=min(100, 項目数)`、暫定）。`before`/`after`の結果を見て選ばない。
3. **ペアの作成**: 各項目について、`after`対`before`、`after`対`random_valid`の画像対を作る。左右の位置は乱数で決める。
4. **盲検化**: ファイル名を`pair_<番号>_<L|R>.png`にする。PNGにメタデータを含めない（`image-drawer-draw`のPNGは標準ライブラリで符号化し、テキストチャンクを持たない）。条件との対応は`blind_key.json`に保存し、評価者に渡さず、集計時にだけ使う。
5. **課題**:
   - T1 prompt一致: 文章を見せ、どちらが文章に合うかを選ぶ（目標画像は見せない）。
   - T2 完成度: どちらがイラストとして完成しているかを選ぶ（文章は見せない）。
   - 選択肢は「左」「右」「差がない」。
6. **確認項目**: (a)同一画像のペア（期待は「差がない」）、(b)明らかに差のあるペア（白紙と完成した画像）を、全体の約10%混ぜる。(b)の正答が90%未満の評価者は、事前登録した規則で除外し、除外数を報告する。
7. **評価者**: 項目ごとに独立した評価者3人以上、評価者全体で5人以上（暫定）。1セッションは30分以内とし、項目の順序は評価者ごとに乱数で並べる。
8. **集計**: 項目ごとに多数決（同数は「差がない」）。`after`対`before`の勝率を、クラスタ・ブートストラップで区間推定し、符号検定（両側の正確二項検定）で補う。T1とT2の2つにHolm法を適用する。
9. **評価者間一致**: Fleissのκを報告する。κが0.2未満の課題は結論に使わず「判定不能」と書く。

### 8.2 記録項目

`human_responses.csv`の1行に、`rater_id`（匿名のハッシュ）、`session_id`、`item_id`、`task`、`choice`、`response_time_ms`、`display_order`、`is_catch_trial`、`catch_expected`、`timestamp`、`display_resolution`を記録します。評価者の氏名・連絡先は保存しません。利益相反の有無、指示文のSHA-256、同意の取得を報告書に記載します。

評価に使うのが`test`の個人画像である場合は、データの持ち出しと閲覧の許可を運用者が確認してください。このツールは画像を外部へ送信しません。

### 8.3 ツール

評価者用のパックは、画像、回答用のCSVの雛形、指示文から成ります。GUIは必須としません（`image-drawer-evaluate human-pack`と`human-collect`、12節）。

## 9. 報告書の形式

### 9.1 JSON（`idraw-eval-report/v1`、設計案）

スキーマは実装PRで`docs/developers/schemas/eval_report_v1.schema.json`として追加します。

```json
{
  "schema_version": "idraw-eval-report/v1",
  "report_id": "string",
  "created_at": "ISO8601",
  "status": "draft | frozen | published",
  "claim_level": "none | pipeline_verified | improvement_observed | high_accuracy_claim_allowed",
  "code": {
    "git_commit": "string", "worktree_clean": true, "package_version": "string",
    "renderer_version": "string", "palette_id": "string", "brush_set_id": "string",
    "control_set_id": "string", "stage_workflow_sha256": "string",
    "reward_version": "string", "reward_weights_sha256": "string",
    "metrics_version": "string", "policy_architecture_id": "string"
  },
  "dataset": {
    "dataset_hash": "string", "split_seed": 0, "eval_set_id": "string", "eval_set_sha256": "string",
    "counts": {"train": {"images": 0, "groups": 0}, "validation": {"images": 0, "groups": 0},
               "test": {"images": 0, "groups": 0}},
    "extractors": {"tagger": {"id": "", "version": "", "model_hash": "", "config_hash": ""},
                   "captioner": {"id": "", "version": "", "model_hash": "", "config_hash": ""},
                   "prompt_builder_version": ""},
    "exclusions": [{"image_id": "string", "reason": "string"}]
  },
  "conditions": [
    {"condition_id": "before | after | random_valid | blank", "checkpoint_sha256": "string|null",
     "training_seed": 0, "training_steps": 0}
  ],
  "protocol": {
    "inference_seeds_per_item": 5, "seed_derivation": "sha256(eval_set_id:image_id:k)",
    "max_actions": 1000, "canvas": {"width": 256, "height": 256},
    "bootstrap": {"unit": "group_id", "resamples": 10000, "ci": 0.95, "rng_seed": 0},
    "freeze_manifest_sha256": "string", "test_ledger": {"run_index": 1, "retest": false},
    "hardware_id": "string"
  },
  "results": {
    "by_condition": {"<condition_id>": {"n_items": 0, "n_generations": 0,
        "metrics": {"<metric>": {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0, "denominator": 0,
                                  "failures_included": true, "reporting": "itt | conditional"}},
        "failure_breakdown": {"<kind>": {"count": 0, "rate": 0.0}},
        "cost": {"<field>": {"median": 0.0, "p90": 0.0}}}},
    "comparisons": [{"a": "after", "b": "before", "metric": "string",
        "subset": "test_unseen_image | test_unseen_prompt | train",
        "delta": 0.0, "ci95": [0.0, 0.0], "win_rate": 0.0, "effect_size": 0.0, "n_groups": 0,
        "per_training_seed": [{"seed": 0, "delta": 0.0, "ci95": [0.0, 0.0]}]}],
    "controls": {"prompt_shuffle_delta": {"delta": 0.0, "ci95": [0.0, 0.0]},
                 "probe_hit_rate": {"before": 0.0, "after": 0.0},
                 "nearest_train_hit_rate": {"before": 0.0, "after": 0.0}},
    "process": {"completion_rate": 0.0, "audit_pass_rate": 0.0,
                "auditor_controller_disagreements": 0, "replay_exact_rate": 1.0}
  },
  "human_eval": null,
  "gates": [{"gate_id": "G1", "passed": false, "evidence": "string"}],
  "claims": {"allowed": ["string"], "disallowed": ["string"]},
  "limitations": ["string"],
  "unverified": ["string"],
  "reproduction": {"commands": ["string"], "environment": {"python": "", "os": "", "pillow": ""},
                   "input_hashes": {"freeze_manifest": "", "eval_set": "", "checkpoints": []}}
}
```

`human_eval`は実施したときだけ、`plan_sha256`、`raters`、`items`、課題ごとの`win_rate`・`ci95`・`p_holm`・`kappa`・`catch_pass_rate`・`excluded_raters`を持つオブジェクトにします。

### 9.2 Markdown（`report.md`）の見出し

1. 結論と主張の水準（`claim_level`と、許す表現・許さない表現）
2. 条件（checkpoint、学習seed、コードのcommit）
3. データと分割（件数、グループ数、除外、抽出器、`eval_set_id`）
4. 結果（成分別の表、独立指標、`before`/`after`/`random_valid`の比較）
5. 工程の遵守と有効stroke、拒否
6. 失敗の集計
7. prompt忠実度と記憶の検査
8. 人間評価
9. 計算コスト
10. 制限と未確認の範囲
11. 再現手順
12. 付録: 出力例（ランダムに選んだ12件と、得点が最も低い12件。**良い例だけを載せない**）

## 10. 公表の条件

### 10.1 主張の水準

| 水準 | `claim_level` | 満たすべきゲート | 使える表現 |
| --- | --- | --- | --- |
| L0 | `none` | なし | 「評価していない」 |
| L1 | `pipeline_verified` | G0 | 「配線とReplayの一致を確認した（テスト用Policy）」。モデルの品質には触れない |
| L2 | `improvement_observed` | G0〜G4、G6 | 「（条件）で学習前より改善が観測された」 |
| L3 | `high_accuracy_claim_allowed` | G0〜G10のすべて | 「（データ・解像度・件数の範囲で）高精度」 |

### 10.2 ゲート

数値は、最初の実験より前に`metrics_v1`として凍結する**暫定値**です。結果を見てからの変更は無効です。

| ID | 条件 |
| --- | --- |
| G0 | 再現可能。作業ツリーがクリーンで、報告書の再現コマンドを新しい環境で実行して、凍結した入力から同じ指標（浮動小数の許容誤差内）が得られる。`replay_exact_rate = 1.0`、`auditor_controller_disagreements = 0`、Policyの観測にtargetが入らないテストが合格 |
| G1 | データ。`test`のグループ数が100以上。`train`との`group_id`の共有が0。`dataset_exclusion`を含む除外を全件開示。`test_ledger`に`retest`がない |
| G2 | 学習の実体。`before`と`after`でPolicyの重みが異なる（Trainerのテスト）。固定スクリプトの再生ではない |
| G3 | seed。学習seedが3個以上、推論seedが5個以上 |
| G4 | 改善。見出しのRewardの**各成分**について、未見画像と未見promptの両方で、プールした`gain`の信頼区間の下限が0を超える。かつ全学習seedで`gain`の中央値が正 |
| G5 | 独立指標。`heldout_*`の少なくとも1つが同様に改善し、どの成分・指標にも有意な悪化（信頼区間の上限が0未満）がない。`reward_gain_without_heldout_gain`が出ていない |
| G6 | 対照。`after`が`before`と`random_valid`の両方を、見出しの指標で上回る（`gain`の信頼区間の下限が0超） |
| G7 | prompt。`prompt_shuffle_delta`の信頼区間の下限が0超。`probe_hit_rate`が`before`より改善 |
| G8 | 失敗。`completion_rate`が0.95以上（ITTの数値を併記）。`audit_pass_rate = 1.0` |
| G9 | 人間評価。T1とT2の両方で、`after`対`before`の勝率の信頼区間の下限が0.5超。κが0.4以上。評価者5人以上、項目あたり3人以上。確認項目に合格 |
| G10 | 記憶。`nearest_train_hit_rate`が閾値以下（暫定0.05）で、`before`より悪化していない。`generalization_gap`を開示 |

### 10.3 してはいけない公表

次の1つでも当てはまれば、L3（および該当する水準）を主張しません。

- testを調整に使った、または`retest`/`contaminated`である。
- 良いseed、良いcheckpoint、良い画像を選んで報告した。
- Best-of-Nの結果を単発の生成として報告した（Best-of-Nは別の系列で、Nを明記する）。
- 失敗した生成や除外した項目を分母から外した。
- Rewardの成分の改善だけを品質の改善として報告した。
- テスト用Policy（`policy_trained=false`）、モックの評価器、小さな合成データだけの結果を、モデルの精度として報告した。
- 人間評価なしでL3を主張した。
- ハードウェアが異なる実行を`before`/`after`として比較した。
- 結果を見てから指標・閾値・分析方法を変えた。
- 一部のデータで測った結果を「イラスト全般」「あらゆるprompt」に一般化した。
- 「高精度」「必ず収束」「プロ品質」などの語を、L3の報告書への参照なしに使った。
- 人間の絵師の筆跡を学習した、元画像から描画行動を推定した、と表現した（[Policy精度仕様](drawing_policy_accuracy_v1.md)の8節）。

### 10.4 公表文の書き方

公表する文には、次を必ず併記します: データの種類と件数（`test`のグループ数）、解像度、seed数、`claim_level`、報告書の`report_id`、制限、未確認の範囲。

## 11. 正本との差分と判断

既存の正本は変更しません。

| # | 正本 | 本書の判断 | 理由 |
| ---: | --- | --- | --- |
| 1 | Issue #72と[実装計画](../kadoka/implementation_plan.md)は「train/test分割」と書く。S2とデータ品質仕様は`validation`を含む | train/validation/testの3分割。調整は`validation`、公表は`test` | 調整用の分割がないと、testが調整に使われる |
| 2 | [画像のみ学習](image_only_learning_v1.md)の8節は「95%信頼区間」とだけ書く | 区間の計算を、`group_id`単位のクラスタ・ブートストラップ（10000回、パーセンタイル）に具体化 | 近重複の画像は独立でない |
| 3 | [S2のDataset文書](image_dataset.md)は、画像の追加で`split`が変わり得ると書き、「固定のtestを何度も学習調整に使わない」とする | testを`eval_set`としてスナップショットし、凍結宣言・帳簿で調整への使用を機械的に検出する | 規則を文書の注意にとどめず、実行時に強制する |
| 4 | [工程別Reward](stage_reward_v2.md)は、`finish`の報酬に品質スコアの2倍を加える | 評価の見出しにこの報酬を使わず、成分ごとの得点を報告する | 学習用の報酬と評価の指標を分け、Goodhartを避ける |
| 5 | [Controller契約](stage_controller_contract_v2.md)は、Controllerが規則違反のActionを受理しないとする | 「受理済みActionの遵守率」を成果にせず、提案レベルの率と独立した`StageAuditor`で測る | 構造上100%になる値は、モデルの能力を示さない |
| 6 | [Policy精度仕様](drawing_policy_accuracy_v1.md)の5節は未見のprompt（赤／青、一つ／二つ、左／右）の検査を求める | 構成的な未見promptを`prompt_probes_v1.json`の評価資産として定義し、`probe_hit_rate`と人間評価で測る | 評価の入力をユーザーに求めない |

### 11.1 未決事項E1の決定（S5-5）

独立指標`heldout_*`は次のとおり決めます。`metrics_v1`として凍結します。

| 指標 | 扱い | 内容 |
| --- | --- | --- |
| `heldout_ssim64` | 必須 | 64×64の輝度で、8×8の重ならないブロックのSSIMを平均した値。学習のRewardと解像度・特徴が異なる。追加の依存はない（NumPyとPillow） |
| `heldout_openclip_image` | 任意 | OpenCLIPの画像間コサイン類似度。`openclip` extraと重みがある場合だけ測る |
| `heldout_openclip_text` | 任意 | 画像と文章のCLIP一致度。同上 |

任意の指標を測れない場合、報告書の`unverified`に書き、測ったことにしません。G5の「`heldout_*`の少なくとも1つが改善」は、測定した指標について判定します。`nearest_train_hit_rate`の距離にも`heldout_ssim64`の差（1−SSIM）を使います。

## 12. 再現コマンド（実装後の仕様）

以下は設計案です。実装前には実行できません。実行ディレクトリはリポジトリのルート、`python -m pip install -e ".[dev]"`済みを前提とします。

```powershell
# 1. 評価集合を凍結する（学習・調整の前に1回）
image-drawer-evaluate plan --dataset datasets/main --out reports/eval/eval_set_main_v1.json

# 2. validationで調整する（凍結不要。何回でも可）
image-drawer-evaluate run --dataset datasets/main --eval-set reports/eval/eval_set_main_v1.json `
  --split validation --condition before=checkpoints/init.ckpt --condition after=checkpoints/run1.ckpt `
  --out reports/eval/validation_run1

# 3. 凍結宣言を作る（調整の終了後に1回）
image-drawer-evaluate freeze --eval-set reports/eval/eval_set_main_v1.json `
  --before checkpoints/init.ckpt --after checkpoints/run1.ckpt --analysis-plan docs/developers/eval_analysis_plan_v1.json `
  --out reports/eval/freeze_manifest.json

# 4. testで1回だけ評価する
image-drawer-evaluate run --dataset datasets/main --eval-set reports/eval/eval_set_main_v1.json `
  --split test --freeze-manifest reports/eval/freeze_manifest.json --out reports/eval/test_run

# 5. 比較と報告書
image-drawer-evaluate compare --run reports/eval/test_run --out reports/eval/compare.json
image-drawer-evaluate report --compare reports/eval/compare.json --out reports/eval/report.json --markdown reports/eval/report.md

# 6. 人間評価
image-drawer-evaluate human-pack --run reports/eval/test_run --plan reports/eval/human_eval_plan.json --out reports/eval/human_pack
image-drawer-evaluate human-collect --pack reports/eval/human_pack --responses responses.csv --out reports/eval/human_result.json
```

| コマンド | 入力 | 出力 | 終了コード |
| --- | --- | --- | --- |
| `plan` | Dataset | `eval_set` | 0 成功、2 実行不能 |
| `freeze` | `eval_set`、checkpoint、分析計画 | 凍結宣言 | 0、2 |
| `run` | Dataset、`eval_set`、条件 | 生成結果、指標、失敗集計、コスト | 0 成功、1 失敗の生成が含まれる、2 実行不能（testで凍結宣言がない、`retest`など）、3 Replay不一致・監査器の不一致 |
| `compare` | 実行結果 | 差、区間、勝率 | 0、2 |
| `report` | 比較、人間評価 | 報告書のJSONとMarkdown | 0、2（ゲート未達でも報告書は作る） |
| `human-pack`、`human-collect` | 実行結果、計画、回答CSV | パック、集計 | 0、2 |

`run`の終了コード1は、失敗した生成があっても結果を保存したことを示します。失敗を隠して0を返しません。

実験の乱数と生成結果、`test_ledger.jsonl`、`blind_key.json`は`reports/`や`data/`に置き、Gitへ登録しません（[保存場所](../../data/README.md)）。

## 13. 受け入れテスト一覧

### `tests/test_eval_set.py`

- `test_eval_set_snapshot_is_unchanged_when_dataset_grows`
- `test_eval_set_rejects_group_shared_with_train`
- `test_eval_set_hash_changes_when_any_item_changes`
- `test_unseen_prompt_excludes_prompts_seen_in_train_and_validation`
- `test_probe_templates_change_only_the_declared_attribute`
- `test_test_split_run_requires_freeze_manifest`
- `test_second_freeze_on_same_eval_set_marks_retest_in_ledger`
- `test_ledger_is_append_only_and_records_manifest_hash`
- `test_validation_run_does_not_touch_the_ledger`

### `tests/test_eval_stats.py`

- `test_inference_seed_derivation_is_stable_and_paired_across_conditions`
- `test_cluster_bootstrap_resamples_by_group_id_not_by_image`
- `test_bootstrap_is_deterministic_for_fixed_rng_seed`
- `test_bootstrap_interval_contains_zero_for_identical_conditions`
- `test_win_rate_counts_ties_as_half`
- `test_holm_adjustment_matches_reference_values`
- `test_fleiss_kappa_matches_reference_values`
- `test_itt_assigns_blank_score_to_failed_generations`
- `test_conditional_and_itt_aggregates_differ_when_failures_exist`

### `tests/test_stage_auditor.py`

- `test_auditor_accepts_complete_seven_stage_episode`
- `test_auditor_rejects_missing_stage_enter_or_exit`
- `test_auditor_rejects_out_of_order_stage_events`
- `test_auditor_rejects_required_stage_without_effective_stroke`
- `test_auditor_rejects_write_to_forbidden_layer`
- `test_auditor_rejects_skip_stage_and_v2_set_phase`
- `test_auditor_rejects_review_reason_not_allowed_for_stage`
- `test_auditor_does_not_read_controller_logs`
- `test_auditor_and_controller_agree_on_generated_episodes`

### `tests/test_eval_metrics.py`

- `test_effective_stroke_rate_ignores_strokes_that_change_no_pixel`
- `test_dot_stroke_rate_detects_sub_pixel_paths`
- `test_rejection_rate_breaks_down_by_reason_and_stage`
- `test_proposal_legal_rate_uses_all_attempts_as_denominator`
- `test_cost_fields_use_null_when_memory_is_unavailable`
- `test_prompt_shuffle_delta_is_positive_for_prompt_dependent_toy_policy`
- `test_prompt_shuffle_delta_is_near_zero_for_prompt_ignoring_policy`
- `test_nearest_train_hit_rate_flags_copied_train_target`
- `test_quality_metrics_report_each_reward_component_separately`
- `test_reward_gain_without_heldout_gain_warning_is_raised`
- `test_metrics_never_load_target_inside_policy_process_boundary`

### `tests/test_eval_runner.py`

- `test_runner_uses_same_seeds_budget_and_canvas_for_all_conditions`
- `test_runner_keeps_failed_generations_in_denominators`
- `test_runner_exit_code_is_1_when_failures_exist_and_results_are_saved`
- `test_runner_blocks_comparison_across_different_hardware_ids`
- `test_runner_before_after_with_toy_policy_shows_expected_direction`（配線の確認。モデルの精度の根拠にしない）

### `tests/test_human_eval.py`

- `test_human_pack_hides_condition_names_and_metadata`
- `test_human_pack_randomizes_left_right_by_seed`
- `test_human_pack_includes_catch_trials_identical_and_obvious`
- `test_human_collect_excludes_raters_below_catch_threshold_and_reports_count`
- `test_human_collect_marks_task_inconclusive_when_kappa_is_below_threshold`
- `test_human_plan_hash_must_match_before_collecting`
- `test_responses_csv_has_no_personal_fields`

### `tests/test_eval_report.py`

- `test_report_json_validates_against_schema`
- `test_report_marks_unverified_layers_explicitly`
- `test_claim_level_is_none_without_trained_policy`
- `test_claim_level_l1_for_scripted_policy_runs`
- `test_claim_level_l3_requires_every_gate_g0_to_g10`
- `test_gate_g4_fails_when_any_component_interval_includes_zero`
- `test_gate_g1_fails_when_ledger_has_retest`
- `test_gate_g9_fails_without_human_evaluation`
- `test_report_markdown_has_required_headings_and_random_and_worst_examples`
- `test_report_rejects_disallowed_phrases_without_l3`
- `test_reproduction_commands_reproduce_metrics_from_frozen_inputs`

## 14. 実装手順（PRの分割）

1PRは1つの目的で、変更は500行以下を目安にします。パッケージは`image_drawer/generalization/`とし、既存の`image_drawer/evaluation/`（Workflowの候補選択用）と混ぜません。実装済みのPRは S5-1〜S5-8 です（S5-9は本書の再現手順と[ユーザー向け手順](../users/evaluation.md)）。

| PR | 目的 | 主な内容 | 前提 |
| --- | --- | --- | --- |
| S5-1 | 評価集合と帳簿 | `eval_set.py`、`freeze`、`test_ledger.jsonl`、`test_eval_set.py` | S2 |
| S5-2 | 統計 | `stats.py`（クラスタ・ブートストラップ、勝率、Holm、κ、ITT）。外部依存なし | なし |
| S5-3 | 工程の監査 | `auditor.py`、`test_stage_auditor.py` | S1c |
| S5-4 | 工程・stroke・拒否・コスト指標 | `metrics_process.py`、`test_eval_metrics.py`の該当部分 | S5-3、S4-1 |
| S5-5 | 画像品質・対照 | `metrics_quality.py`（Reward成分、独立指標、prompt入れ替え、記憶検査） | S3、E1の決定 |
| S5-6 | ランナーとCLI | `runner.py`、`cli.py`（plan/freeze/run/compare）、`pyproject.toml`のscript | S5-1〜5、S4-4 |
| S5-7 | 人間評価 | `human.py`（pack/collect）、`test_human_eval.py` | S5-2 |
| S5-8 | 報告書とゲート | `report.py`、`docs/developers/schemas/eval_report_v1.schema.json`、`test_eval_report.py` | S5-6、S5-7 |
| S5-9 | 文書 | `docs/users/`の再現手順、`docs/developers/README.md`の索引 | S5-8 |

S5-1、S5-2、S5-3はS3のPolicyやTrainerを待たずに進められます。S5-5以降は、実際の評価にはS3が必要です。完了報告には、実行した層（テスト用Policyによる配線か、実checkpointによる評価か）を区別して書きます。

## 15. 未決事項

| ID | 内容 | 決める場所 |
| --- | --- | --- |
| E1 | 決定済み（11.1節）。`heldout_ssim64`を必須、OpenCLIP系を任意とする | S5-5 |
| E2 | 暫定値の根拠（test 100グループ、seed 3/5個、人間評価N_h=100など）。検出力の計算で見直す | 最初の実験の前 |
| E3 | Rewardの成分名・範囲・画像の正規化（S3が未freeze） | S3 |
| E4 | 色・個数・位置のprobeのうち、個数と位置の自動判定手段 | S5-5。判定できない間は人間評価のみで測る |
| E5 | `StageAuditor`が使う有効strokeの判定がEngineの内部状態に依存する範囲 | S1c、S5-3 |
| E6 | `retest`となった`eval_set`の後始末（新しいtestの作り方） | S5-1 |
