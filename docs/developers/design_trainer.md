> **対象読者:** `image_drawer/training/` を実装する人、Trainerの受け入れテストを書く人、レビュー担当
>
> **目的:** 文章と白紙からAI自身が描き、描画結果のRewardでPolicyを更新するTrainer（S3、Issue #70）の実装設計を定める。
>
> **状態:** 実装済み（Trainer、Policy、RewardEvaluator、TextEncoder、ClaimGate、学習CLI）。合成画像での実測は1回だけで、判定は`not_learned`でした（12節）。この文書は「モデルが絵を描いて学習できた」ことを示しません。

# Trainer実装設計（S3）

親Issueは #62、全体計画は[実装計画](../kadoka/implementation_plan.md)です。入力の正本は[画像だけで学ぶ仕様](image_only_learning_v1.md)、[DrawingPolicy学習仕様](drawing_policy_accuracy_v1.md)、[工程別Reward](stage_reward_v2.md)（[JSON](stage_reward_v2.json)）、[工程Controller契約](stage_controller_contract_v2.md)です。これらを変更せず、本書で具体化します。食い違いは10節に記録します。

## 1. 目的・範囲・成功と失敗の基準

### 1.1 目的

文章（training_prompt）と白紙から、`StageController`が固定する7工程の中で、Policyが自分のActionを選んで描きます。描画結果を`RewardEvaluator`が目標画像と比べて採点し、`Trainer`が採点結果でPolicyのパラメータを更新します。更新前後で品質が上がるかを、未見画像で測ります。

### 1.2 範囲

- CPUだけで動く最小のPolicy、TextEncoder、RewardEvaluator、Trainer、評価と記録を実装します。
- 工程順はS1bのControllerが固定します。Policyは各工程の中のAction（レイヤー選択、色、ブラシ、サイズ、ストローク、工程を進める提案）を学びます。
- 人間の教師Actionは使いません。ベースラインとして、ランダム合法Policyと未学習Policyを同条件で比較します。
- 各trialの`actor=model`のAction、拒否attempt、成分別Reward、失敗・予算超過、更新前後のcheckpointを保存します。

### 1.3 範囲外

- GPU、大規模なニューラルネットワーク、画像encoder、拡散・Transformer型Policyは範囲外です。5.5節の拡張点だけを用意します。
- 抽出器の実装（S2）、txt2img推論の入口とGUI（S4）、Part Bank、`COMPOSE`の描画経路は範囲外です。
- RewardのGolden値をv2の正本として確定することは範囲外です（10節）。
- 高画質、収束、人間の絵師との同等性は保証しません。

### 1.4 成功と失敗の基準

「改善が見えなければ未完成と表示する」を、次の測定と判定器`ClaimGate`で扱います。閾値は**この文書で先に固定**し、実験結果を見て変更しません。変更する場合は理由を残し、別のPRで正本とテストを同時に直します。

測定の対象は、学習に使わないheld-out画像（`split=test`）です。8枚以上の画像と、画像ごとに5個以上の評価seedを使います。学習前（未学習Policy）と学習後を、同じ画像・同じ文章・同じseed・同じBudgetで比べます。

| ID | 測定 | 合格条件 |
| --- | --- | --- |
| M1 | パラメータ更新 | `‖θ_after − θ_before‖₂ > 0`。同じ記録済みActionに対する対数確率が更新前後で変わる |
| M2 | 完走率 | 7工程すべてを通って`finish`した割合（M2a）が学習後に+0.30以上増える。ただし余地（1−学習前の完走率）が0.30より小さいときは余地まで。学習前を下回らない。予算超過率（M2b）が学習前より増えない |
| M3 | 最終品質 | `final_quality`（9節の定義）の平均差が+0.05以上。画像単位の対応ブートストラップ95%信頼区間の下限が0より大きい |
| M4 | ベースライン比較 | M3の学習後が、ランダム合法Policyの`final_quality`も同じ基準で上回る |
| M5 | 文章の使用 | 同じseedで文章だけを入れ替えた出力が変わる（出力画素距離の平均が0より大きい）。かつ`prompt_swap_margin`（自分の文章での品質−入れ替えた文章での品質）の平均が正で、信頼区間の下限が0より大きい |
| M6 | 固定Script化の検査 | 異なる文章で、学習後の成功trialのAction列が全て同一ではない（8節のT-FS） |

判定は3値です。

- `learned`: M1〜M6のすべてを満たし、8.4節の条件を満たす。「小規模設定で、測定した範囲で改善した」と表示できます。
- `not_learned`（未完成）: M1が満たされない、またはM2〜M6のいずれかが満たされない。報告・READMEの表示を「未完成」にします。
- `inconclusive`: 画像数・seed数・失敗率の開示が足りず、判定できない。成功として扱いません。

失敗の基準も同じ表で扱います。`finish`に達しない、拒否attemptが多い、全文章で同じ絵を出す、は成功ではありません。

## 2. モジュール構成

新規パッケージ `image_drawer/training/` を作ります。1ファイル300行以下を目標にし、責務ごとに分けます。既存の`image_drawer.drawing`（S1a、S1b）と`image_drawer.dataset`（S2）を読み込む側で、Trainerからそれらを書き換えません。継承による再利用はせず、合成とProtocolを使います（[規約](../kadoka-codex/coding_conventions.md)の継承の節）。

| ファイル | 責務 |
| --- | --- |
| `__init__.py` | 公開APIの再エクスポート。numpyが無い場合に案内付きのエラーを出す |
| `config.py` | `TrainConfig`、`CurriculumConfig`、`RewardConfig`。検証とJSON化 |
| `text_encoder.py` | `TextEncoder` Protocol、`HashedNgramTextEncoder`、`register_text_encoder` |
| `observation.py` | `Observation`、`OBSERVATION_KEYS`、`ObservationBuilder`。targetを知らない |
| `action_codec.py` | Policyの離散・連続出力を`DrawingAction`へ変換。工程別の候補Action種別 |
| `policy.py` | `DrawingPolicy` Protocol、`PolicyDecision`、`ParametricPolicy`（線形softmaxの各head。ストロークは離散） |
| `policy_params.py` | パラメータ配列の初期化、勾配の蓄積、Adam状態 |
| `rollout.py` | `TrialStep`、`Trial`、`TrainingRollout`、`RolloutRunner`（Policy×Controllerのループ。評価器を持たない） |
| `reward_features.py` | numpyだけの画像特徴（縮小、Sobel、Dice、ヒストグラム）。純関数 |
| `reward.py` | `RewardEvaluator`、`RewardBreakdown`、目標特徴のキャッシュ |
| `optimizer.py` | REINFORCEの勾配計算、工程別baseline、Adam更新 |
| `checkpoint.py` | `.npz`＋sidecar JSONの保存・読込・SHA-256検証 |
| `trainer.py` | `Trainer`（rolloutの収集→採点→更新→保存の統括） |
| `experiment.py` | `ExperimentRecord`と書き出し |
| `evaluation.py` | ベースライン、対応ブートストラップ、`ClaimGate` |
| `integrity.py` | trialの整合検査（Replay一致、許可Action型、禁止importの静的検査） |
| `__main__.py` | `python -m image_drawer.training train / evaluate` |

出力先は、Git管理外の `reports/training/<experiment_id>/`（[保存場所](../../data/README.md)）とします。checkpointは `checkpoints/<experiment_id>/` に置きます。

## 3. 公開API

型は`numpy.typing.NDArray`を使います。以下は契約を示す署名で、本体は実装PRで書きます。実装では各クラス・関数の直前に[規約](../kadoka-codex/coding_conventions.md)のJSON-like Comment Outsを付けます。

### 3.1 Observation と DrawingPolicy

Policyが受け取れるものだけを`Observation`に集めます。目標画像、目標特徴、目標との差分、Rewardを持つフィールドは作りません。

```python
OBSERVATION_KEYS: frozenset[str] = frozenset({
    "text_condition",      # TextEncoder.encode(training_prompt) のfloat32ベクトル
    "stage_id",            # 7工程のいずれか
    "canvas_summary",      # 自分の合成キャンバスを8x8へ縮小したRGB (float32, 3x8x8)
    "layer_summary",       # 自分のレイヤー: role, visible, has_pixels, is_active
    "tool_state",          # 色・ブラシ・サイズ・不透明度・レイヤーが選択済みか
    "budget",              # remaining_actions_ratio, remaining_attempts_ratio
    "action_tail",         # 自分の直近L個のActionの種別と accepted の真偽
})

@dataclass(frozen=True)
class Observation:
    text_condition: NDArray[np.float32]
    stage_id: str
    canvas_summary: NDArray[np.float32]
    layer_summary: tuple[LayerBrief, ...]
    tool_state: ToolState
    budget: BudgetView
    action_tail: tuple[ActionBrief, ...]

class ObservationBuilder:
    def __init__(self, tail_length: int = 8, summary_side: int = 8) -> None: ...
    def build(
        self,
        text_condition: NDArray[np.float32],
        controller: StageController,
        engine: Engine,
        recent: Sequence[ProposalResult],
    ) -> Observation: ...
```

`ProposalResult`から`ActionBrief`に取り出すのは種別と`accepted`だけです。拒否理由は、Controllerの理由コードを粗い列挙（`illegal_stage`、`no_layer`、`invalid_value`、`budget`など）へ写します。これは目標画像に依存しません。

```python
@dataclass(frozen=True)
class PolicyDecision:
    action: DrawingAction            # actor="model"。seq・time_msはEngineが補う
    sample: PolicySample             # 勾配計算に必要な特徴量・選択結果・乱数

class DrawingPolicy(Protocol):
    policy_version: str
    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision: ...
    def accumulate_log_prob_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float: ...                  # weight × ∇logπ を grads に足し、logπ を返す
    def new_grads(self) -> PolicyGrads: ...
    def apply_update(self, deltas: Mapping[str, NDArray[np.float64]]) -> None: ...
    def state_dict(self) -> dict[str, NDArray[np.float64]]: ...
    def load_state_dict(self, state: Mapping[str, NDArray[np.float64]]) -> None: ...
```

`act`の引数は`Observation`と乱数だけです。`RewardTarget`、`PolicyInput`の`image_id`、画像パス、Dataset型は`policy.py`、`observation.py`、`action_codec.py`、`text_encoder.py`のimportにも現れません（5節）。

`ParametricPolicy`は`DrawingPolicy`を実装するクラスです。他の具象クラスを継承しません。

### 3.2 TextEncoder

```python
class TextEncoder(Protocol):
    encoder_version: str             # 例: "hashed-ngram-v1"
    dim: int
    def encode(self, training_prompt: str) -> NDArray[np.float32]: ...
        # 空文字・空白のみは ValueError。出力は形状 (dim,)、有限値、L2ノルム1

class HashedNgramTextEncoder:
    def __init__(self, dim: int = 64, ngram_range: tuple[int, int] = (1, 3)) -> None: ...

def register_text_encoder(encoder_id: str, factory: Callable[[], TextEncoder]) -> None: ...
def get_text_encoder(encoder_id: str) -> TextEncoder: ...
```

`HashedNgramTextEncoder`は、文字のn-gram（1〜3）を`hashlib.sha256`で`dim`個のバケットへ符号付きで加え、L2正規化します。Pythonの`hash()`は起動ごとに値が変わるため使いません。学習パラメータを持たず、日本語の分かち書き器も要りません。差し替えるときは、同じProtocolを満たすクラスを`register_text_encoder`で登録し、`TrainConfig.text_encoder_id`で選びます。checkpointは`encoder_version`と`dim`を保存し、読込時に一致しなければ`CheckpointMismatchError`にします。画像encoderやsentence encoderは、この登録口に後から足します（5.5節）。

### 3.3 RewardEvaluator

```python
class RewardEvaluator:
    def __init__(self, config: RewardConfig, canvas_side: int) -> None: ...
    def prepare_target(self, target: RewardTarget) -> TargetFeatures: ...
        # 目標画像を読み、canvas_sideへ縮小し、特徴を一度だけ計算して返す。ファイルは最初の1回だけ読む
    def evaluate_trial(
        self, trial: Trial, target: TargetFeatures, training_prompt: str
    ) -> TrialReward: ...

@dataclass(frozen=True)
class RewardBreakdown:
    stage_id: str
    kind: Literal["step", "invalid", "final", "failure", "boundary"]
    features: Mapping[str, float]        # edge_32 など stage_reward_v2.json の特徴名
    effective_resolution: Mapping[str, int]
    potential_before: float | None
    potential_after: float | None
    components: Mapping[str, float]      # shape, outline, color, perceptual, text_match,
                                         # progress, action_cost, final_bonus, failure_penalty
    total: float                         # 学習に使う値（下表の学習重みで合成）
    reward_profile_id: str
    reward_impl_version: str

@dataclass(frozen=True)
class TrialReward:
    steps: tuple[RewardBreakdown, ...]   # 受理Action1件、または拒否attempt1件につき1件
    final_quality: float | None          # finishしなかった場合はNone
    total_return: float
```

Rewardはtrialが終わってから`evaluate_trial`で一括計算します。`RolloutRunner`は`RewardEvaluator`も`TargetFeatures`も受け取りません。`Trial`が持つ各受理Action後の合成画像（スナップショット）だけを、Evaluatorが事後に読みます。これで「同一trial内でtarget依存Rewardを観測に入れない」を、実行時の約束ではなく構造として保証します（5.3節）。

成分とJSONの特徴名の対応は次です。「学習重み」が0の成分は記録・評価だけに使い、更新には使いません。

| 成分 | 内容 | JSONの特徴 | 学習重み |
| --- | --- | --- | --- |
| `outline`（輪郭） | 工程別の輪郭類似度 | `edge_32/64/128` | 工程別表どおり |
| `color`（色） | RGB類似度と色ヒストグラム | `rgb_32/128`、`rgb_hist_16` | 工程別表どおり |
| `perceptual`（知覚差） | 輝度分布類似度 | `luma_64` | 工程別表どおり |
| `shape`（形） | 前景マスクのIoU（白から一定以上離れた画素） | JSONに無い（暫定） | 0（記録のみ） |
| `text_match`（文章一致） | 出力画像にS2の`CaptionExtractor`を適用した文と`training_prompt`の語の一致率 | JSONに無い（暫定） | 0（記録と評価のみ） |
| `progress`（進捗） | `potential_after − potential_before` | 工程別重みの合計 | 1 |
| `action_cost`（Actionコスト） | `−0.05 / max_actions`（全attemptに適用） | JSONの式 | 1 |
| `final_bonus` / `failure_penalty` | `2.0 × final_quality`／`−1` | JSONの式 | 1 |

`step_reward`は、受理Actionについて`potential_after − potential_before − 0.05/max_actions`です。拒否attemptは`invalid_action_penalty`（`−0.05/max_actions`）だけを与えます。工程を進めるActionの報酬は`action_cost`だけで、**新しい工程の入場時点のpotentialを新しい基準点**にします（工程境界で異なる重み同士の差を報酬にしません）。予算超過・attempt上限・不正終了では`episode_failure_penalty`（−1）を最後に加えます。

特徴量の数値化（縮小、Sobel、Dice、ヒストグラム）は、v2の正本が未freezeです。本書の暫定実装を`reward_impl_version = "stage-reward-impl-v0"`として`reward_features.py`に置き、Goldenテストを同じPRに入れます（4.4節、10節）。

### 3.4 Trial と TrainingRollout

```python
@dataclass(frozen=True)
class TrialStep:
    attempt_index: int                    # 受理・拒否を合わせた通し番号
    stage_id: str
    sample: PolicySample                  # 勾配計算用。Policyの特徴ベクトルを含む
    proposal: DrawingAction               # actor="model"
    accepted: bool
    reject_reason: str | None
    events: tuple[DrawingAction, ...]     # Controllerが返した受理イベント（systemを含む）
    composite_after: bytes | None         # 受理時のみ。canvas_side×canvas_side×4のRGBA

@dataclass(frozen=True)
class Trial:
    trial_id: str
    image_id: str                         # 記録用。Policyの入力ではない
    training_prompt: str
    actor: Literal["model"]               # 固定。human・syntheticを受け付けない
    seed: int
    policy_version: str
    policy_sha256: str                    # このtrialを実行したときのcheckpoint
    steps: tuple[TrialStep, ...]
    end_reason: Literal["finish", "action_budget", "attempt_budget",
                        "curriculum_stop", "policy_error", "engine_error"]
    final_composite: bytes | None
    episode: Episode                      # controller.episode()。未完成Draftを含む
    stop_after_stage: str | None          # カリキュラムで途中打切りした工程。完成扱いにしない

@dataclass(frozen=True)
class TrainingRollout:
    rollout_id: str
    iteration: int
    trials: tuple[Trial, ...]
    rewards: tuple[TrialReward, ...]      # trialsと同順。採点後に結合する

class RolloutRunner:
    def __init__(
        self,
        policy: DrawingPolicy,
        text_encoder: TextEncoder,
        observation_builder: ObservationBuilder,
        config: TrainConfig,
    ) -> None: ...
    def run_trial(
        self, policy_input: PolicyInput, seed: int, stop_after_stage: str | None
    ) -> Trial: ...
```

`run_trial`が受け取る`PolicyInput`は`prompt`だけを使い、`image_id`は`Trial`に記録するためだけに保持します。`RewardTarget`は引数にありません。`stop_after_stage`が`None`でないtrialは、`end_reason="curriculum_stop"`の部分学習用で、`final_quality`と完成率の分母から除きます（6節、8.4節）。

### 3.5 Trainer と ExperimentRecord

```python
class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        policy: DrawingPolicy,
        text_encoder: TextEncoder,
        evaluator: RewardEvaluator,
        output_dir: Path,
    ) -> None: ...
    def train(self, dataset_dir: Path) -> ExperimentRecord: ...
    def collect_rollout(self, iteration: int, samples: Sequence[tuple[PolicyInput, RewardTarget]]) -> TrainingRollout: ...
    def update_policy(self, rollout: TrainingRollout) -> UpdateStats: ...
    def save_checkpoint(self, iteration: int) -> CheckpointRef: ...

@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    git_commit: str
    dataset_hash: str
    extractors: Mapping[str, str]          # tagger/captionerのid・version・hash、prompt_builder_version
    text_encoder: Mapping[str, str]        # id, version, dim
    palette_id: str
    brush_set_id: str                      # 例: brush-v2-core12
    control_set_id: str
    renderer_profile: str                  # 例: brush-raster-v2
    stage_workflow_sha256: str
    reward: Mapping[str, str]              # reward_profile_id, reward_impl_version, stage_reward_v2.json のSHA-256
    config: Mapping[str, Any]              # TrainConfigの全項目
    seeds: Mapping[str, int]
    policy_before: CheckpointRef           # path, sha256, policy_version
    policy_after: CheckpointRef
    counts: Mapping[str, int]              # trial数、受理数、拒否数、各end_reasonの数
    metrics_path: str
    claim: ClaimReport | None
    environment: Mapping[str, str]         # python, numpy, platform
```

## 4. 最小の学習アルゴリズム

### 4.1 選択

**線形softmaxの確率的Policyを、REINFORCE（方策勾配）＋工程別のbaselineで更新します。**

選んだ理由です。

1. Issue #70とDrawingPolicy学習仕様が、v0のベースラインとして方策勾配を挙げています。
2. 勾配が解析式で書け、numpyだけで実装できます。有限差分と照合して正しさを検査できます。
3. 進化戦略やクロスエントロピー法は、勾配を使わずtrial全体の合計Rewardだけで更新します。パラメータが約7万個あるため、1回のtrialが長い（約100 Action）この問題では必要なtrial数が増えます。REINFORCEはActionごとの対数確率勾配を使うので、同じ試行数での情報量が多くなります。
4. 進化戦略は、比較用のOptimizerとして後から足せます（5.5節）。

Actor-critic、PPO、画像encoderを使うPolicyは、v0で比較する価値が出てから足します。

### 4.2 Policyの形

`ParametricPolicy`は、観測を固定長の特徴ベクトル`x`に変換し、複数のheadで次のActionを選びます。

- 特徴ベクトル`x`（次元`D`）: `text_condition`（64）、`stage_id`のone-hot（7）、`canvas_summary`の平坦化（8×8×3=192）、`tool_state`（5）、`budget`（2）、直近`action_tail`（直近1件の種別のone-hot 8と、直近8件のacceptedの比率1で計9）、バイアス（1）。合計はおよそ280です。
- `kind` head: Action種別（`select_layer`、`select_color`、`select_brush`、`set_brush_params`、`stroke_v2`、`advance_stage`）のsoftmax。工程別の候補（`action_codec.legal_kinds(stage_id)`）にマスクをかけます。マスクは固定工程表だけで決まり、目標画像に依存しません。
- `color` head: 固定256色の`color_id`へのsoftmax。
- `brush` head: 12プリセットのsoftmax。色とは別のheadです（色×ブラシの直積分類をしません）。
- `size` head（19段階）、`opacity` head（11段階）: それぞれsoftmax。
- `layer` head: 存在するレイヤー上限`MAX_LAYERS`（16）のうち`layer_summary`で書込可能なものへのsoftmax。
- ストローク head（実装済み。直線を描けない連続座標版を置き換えました。仕様は[design_stroke_action.md](design_stroke_action.md)）: 始点の格子マス（`stroke_start`、G×G）、`stroke_mode`（new／continue）、線分数（`stroke_segments`、1〜K）、線分ごとの方向（`stroke_dir_i`、16方向。2本目以降は直前に対する相対）と長さ（`stroke_len_i`、段階 {1,2,3,5,8,13} マス）のsoftmaxです。端点が盤面外になる選択は逐次にマスし、使わない線分・`start`は対数確率・エントロピー・勾配に入れません。探索は温度`softmax_temperature`（既定1.0、logitsをTで割る）とエントロピー項で制御し、ガウスの`σ`設定は廃止しました。
- `advance_stage`は種別の1つとして選びます。進めてよいかはControllerが決め、飛び先はPolicyが選べません。
- 観測から合法と分かるActionだけを選ぶマスク（`action_codec.kind_mask`・`brush_mask`）: 拒否の大半は、道具未選択のstrokeと、筆跡のない工程での`advance_stage`でした（合成画像100枚・seed1の実測は下記）。Controllerの受理条件のうち、`Observation`（`tool_state`・`layer_summary`・`action_tail`）だけで判定できるものをマスクにします。目標画像・`RewardTarget`・`canvas_summary`・`text_condition`は使いません。
  - `stroke_v2`: レイヤー・色・ブラシがすべて選択済み（`tool_state`）で、始点を置けるマスがある。
  - `set_brush_params`: ブラシが選択済み（Engineは未選択を拒否）。
  - `advance_stage`: 筆跡が必須の工程（ラフ・下書き・本描き・基本色・陰影）では、その工程の役割レイヤーに可視の画素がある（工程入場時のレイヤーは空なので、画素があれば有効な筆跡が受理済み）。必須描画のない`refine`では、直近の受理済み`advance_stage`以降に受理済み`stroke_v2`が`action_tail`にある。`review_reason`付きの退出はPolicyが出さないので対象外です。
  - `select_layer`: 書込可能な可視レイヤーがある（従来どおり）。`layer` headも同じ。
  - `brush` head: 消去ブラシ（`eraser_*`）を除く。`stroke_v2`は消去ブラシを必ず拒否し、`erase_v2`はv0の対象外です。定数のマスクです。
  - `select_color`・`select_brush`は常に選べるので、全候補が消えることはありません（`advance_stage`は条件を満たした後に必ず選べ、逃げ道になります）。マスクは保守的でよく、Controllerが受理する提案を禁じる誤りは契約テスト`test_legal_mask.py`が完走台本で検査します。`action_tail`は直近`tail_length`件だけなので、`refine`で長く他のActionを続けるとstrokeを見失い、`advance_stage`が一時的に選べなくなります（strokeを選び直せば戻る）。
  - 実測（合成画像100枚・seed1・20 iteration、`scripts/run_training_experiment.py`）: マスク前は27,442 attemptのうち5,476件（20.0%）を拒否しました。内訳は、`stroke_v2`が3,831件（レイヤー未選択1,636、消去ブラシ1,356、色未選択635、ブラシ未選択204）、`advance_stage`が1,433件（筆跡なし）、`set_brush_params`が212件（ブラシ未選択）です。マスク後は23,051 attemptの拒否が0件で、受理された`stroke_v2`は3,584から7,778になりました。学習の成否はこの数値から主張しません。

可変長の座標列と`POINT_END`、筆圧パターン、`erase_v2`はv0に入れません（10節）。ストロークは格子点の経路から`stroke_samples`（既定12）個の点を線分ごとに等間隔に補間して`stroke_v2`の`samples`を作ります（`K+1 ≤ stroke_samples`を設定検証）。筆圧は設定の固定値（既定は中央のlevel）です。`samples`の厳密なキー集合は`image_drawer.drawing`の実装に合わせ、`action_codec.py`のラウンドトリップテストで確認します。

パラメータ数は、`D≈280`、全headの出力数の合計が`6+256+12+19+11+16+6=326`なので、約9.1万個（float64で約0.7MB）です。Adamの`m`・`v`と勾配バッファを含めても3MB未満です。

### 4.3 更新

1回の更新（iteration）の手順です。

1. 学習画像から`B`枚（既定4）を選びます。各画像に`K`個（既定8）のseedでtrialを実行します（`B×K`trial）。
2. 各trialが終わってから`RewardEvaluator.evaluate_trial`で採点します。
3. 各stepの収益を`G_t = Σ_{u≥t} γ^{u−t} r_u`（`γ=0.99`）とします。`r_u`は受理Actionでも拒否attemptでも、そのstepの`RewardBreakdown.total`です。
4. 工程ごとのbaseline `b_s`（指数移動平均、係数0.1）を引き、`A_t = G_t − b_{stage(t)}`とします。batch全体で`A`を平均0・標準偏差1に正規化します。標準偏差が0のbatchは更新しません（ログに記録）。
5. 勾配は`∇θ = (1/N) Σ_t A_t ∇logπ(a_t|x_t) + β ∇H`（エントロピー係数`β=0.01`）。要素ごとの勾配をノルム5.0でクリップし、Adam（学習率0.001、`β₁=0.9`、`β₂=0.999`）で更新します。
   学習率の根拠（実測）: 特徴ベクトルは連続値が約190成分（L1ノルム約185）あり、Adamの初回は全パラメータを約±lr動かすため、1回の更新でlogitは最大約`lr×185`変わります。学習率0.01では種別logitが約1.8動き、1回の更新でentropyが6.85から1.96へ落ち、「strokeを描かずに受理されないadvance_stageを繰り返す」方策へ崩れてreturnが+0.12から-1.25へ下がりました（合成画像100枚・seed1の実測）。0.001では1回の更新が約0.18に収まり、同条件の6 iterationでentropyは6.1〜8.0、returnは+0.12〜+0.19を保ちました。テスト`test_default_update_moves_logits_only_slightly`が1回の更新幅の上限を守ります。
6. `trial.sample`にある特徴ベクトルと選択結果から対数確率の勾配を計算します。Policyは再実行しません。

`outline`、`color`などの成分は`RewardBreakdown`に別々に保存し、更新に使う`total`は`config.reward.training_weights`で合成します。重みはexperiment configに固定し、理由なく途中で変えません。

カリキュラム（`CurriculumConfig`）として、`[rough] → [rough, sketch] → … → 全7工程`の順に、工程を増やしながら学習できます。先頭の工程だけを描いて`stop_after_stage`で打ち切ったtrialは、その工程のpotentialだけをRewardに使い、`final_bonus`を与えません。最終段階では全7工程で学習します。カリキュラムは学習を安定させる設定で、完成した絵の主張には使いません（8.4節）。

### 4.4 計算量・メモリの見積り

下の数値は設計時の**見積り**で、実測値ではありません。S1aのEngineは純Pythonの画素バッファです。実装PR（PR6）で`scripts`の計測コマンドにより実測し、見積りと異なる場合は設定の既定値を直します。

| 項目 | 設定 | 見積り |
| --- | --- | --- |
| キャンバス | 32×32（`canvas_side=32`）。64×64は任意 | 画素1,024。レイヤー最大16 |
| 予算 | `max_actions=96`、attempt上限は契約どおり`max(100, 5×max_actions)=480` | 工程あたり最低6 Action（`select_layer`、`select_color`、`select_brush`、`set_brush_params`、`stroke_v2`、`advance_stage`）×7＝42で、余裕を持たせた値 |
| 1 Actionのコスト | Policy推論は行列積のみ（約9万積和）で1ms未満。Engine適用と再合成が支配的 | 数ms程度と見積り。要実測 |
| 1 trial | 約100 attempt | 1秒前後と見積り |
| 1 iteration | `B=4`、`K=8` | 32 trial。30秒前後 |
| 学習全体 | 8枚の学習画像、200 iteration | 約1.5〜2時間と見積り（CIでは実行しない） |
| メモリ | スナップショット4KB×100×32 trial | 約13MB。パラメータ・Adam・勾配で3MB未満 |

目標画像の特徴（縮小画像、Sobel、ヒストグラム）は`prepare_target`で1画像1回だけ計算し、`TargetFeatures`に保存して再利用します。trialごとのファイル読込・デコードはしません。

### 4.5 依存ライブラリ

- **numpyを使います。** 理由は、行列積、乱数（`numpy.random.Generator`/`PCG64`）、Sobel・ヒストグラムのベクトル化です。純Pythonでは評価と更新が遅くなります。
- `pyproject.toml`に追加するextrasの名前は`train`です: `train = ["numpy>=1.26,<3", "Pillow>=12.3.0,<13"]`。目標画像の読込にPillowを使います。`dev`にも`numpy>=1.26,<3`を追加し、CIでテストを実行します。`dependencies`は空のままにし、numpyを必須にしません。
- `image_drawer.training`は`image_drawer`の他のモジュールからimportしません。wheelのsmokeは`train`なしでも通ります。`import image_drawer.training`でnumpyが無い場合は、`pip install "image-drawer[train]"`を案内する`ImportError`にします。

### 4.6 固定ScriptやコピーでRewardを稼ぐ経路の禁止

学習中の絵は、`actor=model`のActionだけから作られていなければなりません。次の経路を、テストで検査します（テスト名は8節）。

| 経路 | 検査 |
| --- | --- |
| 固定Script | 異なる文章・seedでのAction列が全て同一でないこと（T-FS）。Action列の正規化後ハッシュの種類数が1なら`scripted_suspect`として`ClaimGate`が`learned`にしない |
| 正解画像のコピー | `integrity.py`が、各trialの`episode`を空のEngineでReplayして`final_composite`と画素一致することを確認する。Engineを経由しない画素が混ざれば不一致になる |
| img2img・Part配置 | `image_drawer.training`が`part_bank`、`steps.compose`、`evaluation.select_*`、`retrieve_parts`をimportしないことを`ast`で検査する。許可Action型は`stroke_v2`などの描画v2の型と`advance_stage`だけで、`paste`や`compose`に相当する型を`ObservationBuilder`と`action_codec`が出力できない |
| 実行時のファイル読込 | Policyの`act`中に`open`・`Path.open`が呼ばれないことを、読込を禁止するfixtureで確認する（T-NOFILE） |
| 不正なactor | `Trial.actor`は`"model"`のみ。`human`・`synthetic`を含むEpisodeは学習に使わない |
| 同一画素の水増し | 画素が変わらない重ね描きは有効筆跡に数えない（Controller契約）。さらに`progress`が負または0の連続を、Actionコストが相殺する |

## 5. target隔離の検査

### 5.1 型で隔離する

- `policy.py`、`observation.py`、`action_codec.py`、`text_encoder.py`、`policy_params.py`は、`image_drawer.dataset`、`reward`、`rollout`のうち`Trial`以外を**importしません**。静的テスト（T-IMPORT）が`ast`でimport先を調べます。
- `DrawingPolicy.act`の引数と戻り値の型注釈を`typing.get_type_hints`で再帰的に展開し、`RewardTarget`、`TargetFeatures`、`RewardBreakdown`が現れないことを検査します（T-SIG）。
- `RolloutRunner.__init__`と`run_trial`の型注釈にも、`RewardTarget`、`TargetFeatures`、`RewardEvaluator`が現れません。

### 5.2 観測キー集合の契約テスト

`Observation`のフィールド名の集合が`OBSERVATION_KEYS`と**完全に一致**することを検査します。キーを足すときは、このテストの期待値と本書の同時変更が必要です。フィールドに`target`、`reward`、`score`、`diff`、`goal`を含む名前があれば失敗させます（T-KEYS、T-NAMES）。

### 5.3 同一trial内の構造的な分離

- Rewardはtrial終了後にだけ計算します。`RolloutRunner`は`RewardEvaluator`を持たないため、ループ内で目標を参照できません。
- `ObservationBuilder`は`text_condition`、`controller`、`engine`、直近の`ProposalResult`だけから作ります。`ProposalResult.reason`は粗い列挙に写します。
- 学習中の更新はtrial単位のbatchが終わってからです。1 iteration内の全trialは同じ`policy_sha256`を使います。

### 5.4 目標のすり替え試験

同じ文章・seed・checkpointで、`RewardTarget`を別の画像に差し替えます。`RolloutRunner`はtargetを受け取らないため、全attemptの`proposal`が一致します。この試験は、Trainer全体（`collect_rollout`）に別画像の`RewardTarget`を渡しても、`Trial.steps[*].proposal`が一致し、`RewardBreakdown`だけが変わることとして書きます（T-SWAP）。パラメータ更新前に実行します。

### 5.5 拡張点

| 差し替える部分 | 方法 | 守る契約 |
| --- | --- | --- |
| TextEncoder | `register_text_encoder`で登録 | `encoder_version`・`dim`をcheckpointと`ExperimentRecord`へ保存 |
| Policy | `DrawingPolicy` Protocolを実装（torch等） | `act`は`Observation`と乱数だけを受け取る。torchは別extras |
| Optimizer | `optimizer.py`に`EvolutionStrategyUpdater`などを追加 | Rewardはtrial後にだけ使う |
| 特徴 | `reward_features.py`に特徴を足し、JSONの特徴名と同時に更新 | 未知の特徴名は拒否 |
| 画像encoder | `Observation.canvas_summary`の算出を差し替える | 自分のキャンバスだけから計算し、学習時と推論時で同じ規則 |

GPUと大規模モデルは本書の範囲外です。上の拡張点から後で追加します。

## 6. 乱数・再現性

- `TrainConfig.master_seed`から`numpy.random.SeedSequence`を作り、`spawn`で `init`（パラメータ初期化）、`iteration_i/trial_j`（Policyのサンプル）、`engine`（`DrawingState.seed`、grain用）、`split`（画像の選択順）に分けます。乱数は`PCG64`を使い、グローバルな`numpy.random`・`random`は使いません。
- 同じ環境（Python、numpy、OS、CPU）で、同じ`master_seed`、dataset、設定なら、Action列とcheckpointが一致することをテストで確認します。環境が異なる場合の浮動小数点の完全一致は保証しません。`ExperimentRecord.environment`に版を保存します。
- checkpointは`checkpoints/<experiment_id>/iter_<n>.npz`と `iter_<n>.json` です。`.npz`は`allow_pickle=False`で保存・読込し、Policyの重み、Adamの`m`・`v`・step、工程別baseline、乱数状態（`bit_generator.state`をJSON化）を含みます。sidecar JSONにnpzのSHA-256、`policy_version`、`encoder_version`、`dim`、`config_hash`、`stroke_head_hash`（ストローク head の名前・出力数・G・K・方向数・長さ段階のSHA-256）、`iteration`を保存します。読込時にSHA-256、版、形状を検査し、不一致は`CheckpointMismatchError`にします。
- `ExperimentRecord`は、3.5節の項目（`dataset_hash`、extractorの`id`・`version`・`hash`、`prompt_builder_version`、`text_encoder`、palette、brush set、control set、renderer、stage workflowのSHA-256、`reward_profile_id`・`reward_impl_version`・`stage_reward_v2.json`のSHA-256、config、seed、`policy_before`・`policy_after`、各`end_reason`の件数、環境）を`experiment.json`に書きます。`dataset_hash`は`DatasetManifest.compute_hash()`の値で、`iter_samples`の対象splitと一緒に保存します。
- 学習の途中再開は、checkpointのiterationから、保存した乱数状態で再開します。再開前後で結果が一致することをテストします（T-RESUME）。

## 7. 失敗・予算超過・拒否の扱い

[Controller契約の7節](stage_controller_contract_v2.md)の`Budget`をそのまま使います。

- **拒否attempt**: Controllerが返す`ProposalResult(accepted=False)`をTrialに残します。報酬は`invalid_action_penalty`、`seq`は進みません。Policyの観測には「直近の種別とaccepted」だけが戻ります。拒否もPolicy勾配に含めます（避けるべき選択として学ぶため）。
- **予算超過・attempt上限**: Controllerは未完成のDraftと終了理由を返し、残り工程を自動で進めません。`Trial.end_reason`を`action_budget`/`attempt_budget`にし、`episode_failure_penalty`を最後のstepに加えます。成功trialの統計から除かず、失敗として分母に入れます。
- **Policy例外・Engine例外**: `policy_error`/`engine_error`としてtrialを記録します。そのtrialは勾配更新に使わず、件数を`ExperimentRecord.counts`に入れます。同じiterationで半数を超えた場合は学習を中断し、`ExperimentRecord`に中断理由を残します。
- **除外の禁止**: 失敗した画像・trialをDatasetから除外しません。Datasetの抽出失敗はS2の`IngestReport`にあり、Trainerはそれを`counts`に引き継ぎます。
- **カリキュラム打切り**: `curriculum_stop`は失敗ではなく、完成扱いにもしません。完成率・`final_quality`の分母に入れません。
- **早すぎる`finish`**: Controllerが拒否します。受理された場合でも`final_quality`が低ければ`final_bonus`が小さくなります。

## 8. 受け入れテスト一覧

置き場所は `tests/training/`（実装PRで作成）です。numpyが無い環境では`pytest.importorskip("numpy")`でskipします。Engine/Controller/Datasetが未マージの段階では、各PRはProtocolを満たす小さなfakeでテストし、統合テストは依存PRのマージ後に有効にします。

### 8.1 target隔離・契約

| テスト関数 | 内容 |
| --- | --- |
| `test_observation_keys_match_contract` | `Observation`のフィールド集合が`OBSERVATION_KEYS`と一致 |
| `test_observation_has_no_target_like_field_names` | `target`/`reward`/`score`/`diff`/`goal`を含む名前が無い |
| `test_policy_act_signature_excludes_reward_target` | `get_type_hints`の再帰展開に`RewardTarget`等が無い |
| `test_runner_signature_excludes_evaluator_and_target` | `RolloutRunner`の型注釈に`RewardEvaluator`/`TargetFeatures`が無い |
| `test_policy_modules_do_not_import_dataset_or_reward` | `ast`でimport先を検査（T-IMPORT） |
| `test_policy_act_performs_no_file_io` | `open`/`Path.open`を禁止して`act`を実行（T-NOFILE） |
| `test_swapping_reward_target_keeps_all_proposals_identical` | 5.4節（T-SWAP）。同じ文章・seed・checkpointで`proposal`が一致し、Rewardだけが変わる |
| `test_reward_is_computed_only_after_trial_ends` | `RolloutRunner`実行中に`RewardEvaluator`が呼ばれない（呼出し回数0） |
| `test_trial_actor_is_model_only` | `human`・`synthetic`のEpisodeを`Trial`が拒否 |

### 8.2 TextEncoder・Policy・Reward

| テスト関数 | 内容 |
| --- | --- |
| `test_text_encoder_is_deterministic_and_unit_norm` | 同じ文は同じベクトル、L2ノルム1、`hash()`に依存しない |
| `test_text_encoder_rejects_empty_prompt` | 空文字・空白のみで`ValueError` |
| `test_text_encoder_registry_rejects_dim_mismatch_checkpoint` | 差し替えて`dim`が違えば読込失敗 |
| `test_policy_gradient_matches_finite_difference` | 各headで解析勾配と有限差分が一致 |
| `test_policy_masks_kinds_by_stage` | 工程外の種別の確率が0 |
| `test_policy_same_seed_same_action` | 同じ`Observation`・rngで同じ`PolicyDecision` |
| `test_action_codec_roundtrips_through_engine` | 生成した`DrawingAction`が実Engineで受理される（境界: 座標0と1、最大レイヤー） |
| `test_action_codec_rejects_out_of_range_indices` | 範囲外の`color_id`/level等で`ValueError` |
| `test_reward_features_golden_values` | 縮小、Sobel、Dice、ヒストグラムの小画像Golden（暫定版の固定入力と期待値） |
| `test_stage_weights_sum_to_one_and_unknown_feature_rejected` | `stage_reward_v2.json`の読込検証 |
| `test_reward_blank_vs_blank_target_is_not_rewarded_for_blank_when_target_has_edges` | 白紙は輪郭を持つ目標に対し低い |
| `test_reward_progress_improves_when_stroke_matches_target` | 目標の位置に引いた筆跡のpotentialが上がる |
| `test_reward_stage_boundary_does_not_mix_potentials` | 工程を進めるstepに前後のpotentialの差が入らない |
| `test_reward_invalid_attempt_penalty_and_no_pixel_change` | 拒否attemptは`−0.05/max_actions`のみ |
| `test_reward_failure_penalty_on_budget_exceeded` | 予算超過でペナルティ−1 |
| `test_reward_breakdown_contains_all_components` | 全成分名と`reward_impl_version`がある |
| `test_reward_resolution_is_capped_to_canvas_side` | 32×32で`edge_128`が32にcapされ、`effective_resolution`に記録 |

### 8.3 Rollout・Trainer・記録

| テスト関数 | 内容 |
| --- | --- |
| `test_rollout_records_rejected_attempts` | 拒否が`Trial.steps`に残り、`seq`が進まない |
| `test_rollout_budget_exhaustion_returns_unfinished_draft` | 予算超過で`end_reason`と未完成Draft |
| `test_rollout_curriculum_stop_is_not_counted_as_completed` | 8.4節の分母 |
| `test_trial_replay_matches_final_composite` | Replayで画素一致（integrity） |
| `test_integrity_rejects_trial_with_foreign_pixels` | 外部画素を混ぜたtrialを検出 |
| `test_training_package_does_not_import_part_bank_or_compose` | `ast`検査 |
| `test_trainer_updates_parameters` | 1 iteration後に`‖Δθ‖>0`、同じ記録済みActionの対数確率が変わる |
| `test_trainer_skips_update_when_advantage_std_is_zero` | 全Rewardが同じbatchは更新せずログに記録 |
| `test_trainer_clips_gradients` | 巨大なAdvantageでもノルムが上限以下 |
| `test_trainer_aborts_when_most_trials_error` | 半数超が例外で中断し理由を記録 |
| `test_checkpoint_roundtrip_and_sha_verification` | 保存→読込で一致、改ざんで`CheckpointMismatchError` |
| `test_checkpoint_rejects_pickle` | `allow_pickle=False` |
| `test_resume_matches_uninterrupted_training` | T-RESUME |
| `test_training_is_reproducible_with_same_seed` | 同じseed・環境でAction列一致 |
| `test_experiment_record_contains_required_fields` | 6節の全項目が非空 |
| `test_experiment_record_counts_failures_in_denominator` | 失敗trialが分母に入る |

### 8.4 学習前後の改善を測る小規模な統合試験

**データ**: テスト補助関数が、32×32の合成PNGを一時ディレクトリに作ります（白地に1色の長方形を左右・色違いで配置、学習用8枚、held-out用8枚）。S2の`register_images`で**画像だけ**を登録し、`iter_samples`で`(PolicyInput, RewardTarget)`を得ます。Action列や手書きの文章は渡しません。held-outは学習と色・位置の組合せが異なる画像にします。

**手順**:

1. 未学習Policy（固定seedの初期値）をcheckpoint保存し、held-out画像を各5 seedで描かせて、M2〜M5の指標を測ります。
2. 同条件でランダム合法Policy（工程別の候補Actionから一様に選ぶ）を測ります。
3. カリキュラム込みの学習を実行します。
4. 学習後Policyを手順1と同じ画像・文章・seed・Budgetで測ります。
5. `ClaimGate.evaluate`でM1〜M6を判定し、結果を`ExperimentRecord.claim`に保存します。

テスト関数:

| テスト関数 | 内容 | 既定で実行 |
| --- | --- | --- |
| `test_claim_gate_marks_not_learned_when_no_improvement` | 合成の指標表を渡し、改善が無ければ`not_learned`（Done「達しなければ未完成と表示する」の検査） | する |
| `test_claim_gate_requires_confidence_interval_lower_bound_above_zero` | 平均差が+0.05でも下限が0以下なら`not_learned` | する |
| `test_claim_gate_returns_inconclusive_with_too_few_images_or_seeds` | 8画像・5 seed未満は`inconclusive` | する |
| `test_claim_gate_flags_scripted_suspect` | Action列の種類数が1なら`learned`にしない | する |
| `test_paired_bootstrap_ci_is_deterministic_for_seed` | 対応ブートストラップの再現性 | する |
| `test_training_smoke_updates_parameters_and_writes_records` | 2 iteration・`B=2`・`K=2`・`max_actions=24`の短縮設定で、更新、checkpoint、`experiment.json`、`metrics.jsonl`が出力される。**改善は主張しない** | する |
| `test_small_training_improves_heldout_quality_vs_untrained` | 上の手順1〜5の実測。`training_experiment`マーカー付き | しない（手動または専用workflow） |

`training_experiment`の実験は、数十分かかる見込みです。通常のCIでは実行しません。実行結果は `reports/training/<experiment_id>/` に残し、`learned`/`not_learned`/`inconclusive`をそのまま報告します。この実験は閾値を事前登録したテストで、実験結果を見て閾値を調整しません。

**「モデルが絵を描いて学習できた」と主張してよい条件**（すべて必要）:

- 1.4節のM1〜M6がすべて合格し、`ClaimGate`が`learned`を返し、`ExperimentRecord`が保存されている。
- 評価がheld-outの8画像以上・5 seed以上で、失敗・予算超過・拒否の件数を分母に入れて報告している。
- 完成率と`final_quality`が、`curriculum_stop`のtrialを含めずに、全7工程の`finish`したtrialで計算されている。
- 8.1節のtarget隔離テストとT-FS、`integrity`のReplay一致が通っている。
- 報告が「合成した単純な画像セットで、測定した範囲で改善した」という範囲を明記している。

**主張してはいけない場合**:

- 設計書の完成、unit testとCIの成功、スモークテストの成功だけの段階。
- 学習画像に対する改善だけで、held-outの改善が無い、または信頼区間が0を含む。
- 文章を入れ替えても出力が変わらない、またはAction列が全文章で同一。
- Reward成分の一部だけが上がり、`final_quality`が上がらない。
- `curriculum_stop`したtrialだけの改善。
- 実画像（写真・イラスト一般）での品質。この設計の検証対象は合成画像だけです。

## 9. 品質指標の定義

`final_quality`は、`stage_reward_v2.json`の`final_quality_weights`（`rgb_128: 0.6`、`edge_128: 0.3`、`rgb_hist_16: 0.1`）を使った、完成画像の目標との重み付き類似度（0〜1、大きいほど良い）です。32×32では`rgb_128`・`edge_128`が32の解像度にcapされます（`effective_resolution`に記録）。`prompt_swap_margin`は、held-outの同じ出力画像について「自分の目標との`final_quality`」から「別の画像の目標との`final_quality`」を引いた平均です。入れ替え先は、文章の属性（色または位置）が異なる画像から固定の規則で選びます。

## 10. 正本との差分と判断

正本は変更しません。本書が決めたことと、判断の理由です。

| 項目 | 正本 | 本書の判断 |
| --- | --- | --- |
| Rewardを計算する時点 | [画像のみ学習](image_only_learning_v1.md)の擬似コードは、各Action後に`evaluator.compare`を呼ぶ | 学習ループ中ではなく、trial終了後に事後計算する。正本の「同一trial内でtarget依存Rewardを観測に入れない」をより強く守る。報酬の値は同じ式で、更新はtrial後 |
| `stage_reward_v2.json`の`stage_ids` | 6工程（`finish`なし） | `finish`はpotentialを持たない。`finish`のstepは`final_bonus`だけ。JSONを変更しない |
| 特徴量の数値化 | Sobel、縮小、正規化は未freeze（`pending-exact-v2-feature-normalization-and-goldens`） | `stage-reward-impl-v0`として暫定実装を置く。縮小は箱フィルタ、輪郭は輝度のSobel大きさ（4で割って1へ飽和）を3×3平均した上でのsoft Dice（`2Σab/(Σa²+Σb²)`、同一画像で1）、RGB類似度は`1 − 平均絶対差`、`luma_64`は縮小輝度の`1 − 平均絶対差`、ヒストグラムは16bin/チャンネルの交差（縮小せず全画素）。正本のfreezeは別PRで行う |
| 解像度 | `edge_64`、`edge_128`、`rgb_128` | キャンバスが32×32なので、特徴の解像度を`min(指定, canvas_side)`にcapする。工程の重みが同じ解像度に縮退し、粗い・細かいの区別が弱くなる。`effective_resolution`に記録する |
| 成分 | 形・輪郭・色・知覚差・文章一致・進捗・コストの別成分（Issue #70） | JSONの特徴を`outline`/`color`/`perceptual`に対応させ、JSONに無い`shape`と`text_match`は学習重み0で記録・評価だけに使う |
| Policyの入力 | [Policy仕様](drawing_policy_accuracy_v1.md)は`seed`と視覚encoderを例示 | `seed`は観測に含めず、`rng`として別に渡す。視覚encoderの代わりに8×8の縮小サマリを使う |
| 行動空間 | 点を1つずつ予測し`POINT_END`、筆圧パターン、消去 | v0は3制御点の固定長ストロークで、筆圧は固定。直積分類はしない。可変長・消去は拡張とする |
| baseline | REINFORCE＋状態価値baselineなど | 工程別の指数移動平均baselineのみ。criticは後続 |
| TextEncoder | 学習可能かは未規定 | v0は学習パラメータを持たない固定のハッシュn-gram。文章の意味理解の主張はしない |
| Controllerの終了理由 | [契約](stage_controller_contract_v2.md)は未完成Draftと終了理由を返すとだけ記述 | `end_reason`の取得方法（`episode()`の属性）をS1bの設計に合わせる。未決事項として11節に残す |
| カリキュラム | [画像のみ学習](image_only_learning_v1.md)の7節は難度別の段階学習 | 工程数を増やすカリキュラムを追加。打切りtrialは完成扱いにせず、主張の条件から除外する |

## 11. 未決事項

- `Budget`の署名と、`Episode`の終了理由フィールド名・値（S1b #67）。
- `stroke_v2`の`samples`の最終キー集合と、レイヤー作成時に必要な`name`・`role`（S1a #66）。
- `register_images`のmanifestから、extractorの`id`・`version`・`hash`、`prompt_builder_version`、`IngestReport`の失敗件数を取り出す具体的な属性名（S2 #69）。
- 32×32での最適な`max_actions`、`B`、`K`、学習率。PR8の実測の後に、事前登録した閾値は変えずに設定値だけを決める。
- 上の見積りが実測と大きく異なる場合のキャンバスサイズ（16×16への縮小）と、その場合のcap規則。
- 文章一致（`text_match`）を学習重みに入れるか。S2のcaptionerの精度を測った後に判断する。
- Reward数値化のv2正本へのfreezeと、Goldenの置き場所（別PR）。

## 12. 実装手順（PR分割）

各PRは1つの目的、500行以下（ソースとテストの合計）とします。800行を超えそうなら分割します。依存するIssueは #66（DrawingEngine）、#67（StageController）、#68（Replayerと受け入れテスト）、#69（Dataset）です。

| PR | 目的 | 依存 | 主なテスト |
| --- | --- | --- | --- |
| 1 | `config.py`、`text_encoder.py`、`pyproject.toml`の`train`extras | なし | TextEncoderの決定性・空文・登録 |
| 2 | `observation.py`と隔離の契約テスト（5.1、5.2節） | #66、#67（fakeで先行可） | キー集合、型注釈、import検査 |
| 3 | `reward_features.py`（暫定数値化とGolden） | なし | 特徴のGolden、解像度cap |
| 4 | `reward.py`（`RewardEvaluator`、`RewardBreakdown`） | #66、PR3 | 成分、工程境界、失敗ペナルティ |
| 5 | `action_codec.py`、`policy.py`、`policy_params.py` | #66、PR2 | 勾配の有限差分、マスク、Engineでの受理 |
| 6 | `rollout.py`、`integrity.py` | #66、#67、#68、PR2、PR5 | 拒否attempt、予算超過、Replay一致、target交換 |
| 7 | `optimizer.py`、`checkpoint.py`、`trainer.py` | PR4〜6 | 更新、再開、再現性 |
| 8 | `experiment.py`、`evaluation.py`、`__main__.py`、統合試験、実測 | #69、PR7 | 8.3、8.4節。実測後の既定値調整は別PR |

PR8の後、実験結果（`learned`/`not_learned`/`inconclusive`）を記録するPRを別に作ります。結果が`not_learned`なら、そのままREADMEとIssue #70に「未完成」と記録します。

## 13. 実装時の差分（claude/70-trainer）

契約を変えない範囲で、実装が設計書から外れた点です。実測（PR8の実測部分）は未実施です。

| 項目 | 実装 |
| --- | --- |
| `ObservationBuilder.build` | 引数は`(text_condition, controller, engine, budget)`。直近Actionは`ProposalResult`ではなく`controller.attempts()`から作る（拒否Actionの種別は`ProposalResult`に無いため）。`LayerBrief`に`layer_id`を含める（`select_layer`の宛先解決に必要） |
| `DrawingPolicy` | `accumulate_entropy_grad`、`params_sha256()`を追加（エントロピー項、checkpoint照合に必要）。連続座標版のガウス幅`stroke_sigma`は離散ストロークへの置換で廃止 |
| `RewardConfig` | `outline`/`color`/`perceptual`は`progress`の特徴群別の分解で、学習重みの既定は0（二重計上しない）。学習信号は`progress`・`action_cost`・`final_bonus`・`failure_penalty`。`text_match`は未計算（常に0.0、学習重み0） |
| `RewardBreakdown` | `attempt_index`を追加。`finish`は専用のActionが無く、7番目の`advance_stage`（refine→finish）のstepが`final_bonus`を受ける |
| 乱数 | 状態を持たない導出（`SeedSequence([master_seed, stream, iteration, ...])`）。再開はcheckpointのiterationだけで一致するため、checkpointへ乱数状態を保存しない |
| `ExperimentRecord` | `abort_reason`を追加。`TrainConfig.checkpoint_every`を追加。checkpointの既定の置き場所は`<output_dir>/checkpoints/` |
| 追加ファイル | `image_ops.py`（箱フィルタ縮小。Policy側とReward側で共有）、`policy_features.py`（特徴ベクトル化）、`reward_spec.py`（`stage_reward_v2.json`の必要部分の検証付き複製。テストで正本と一致を確認） |
| 目標画像 | `RewardTarget.image_sha256`は正規化前のファイルのハッシュで、保存済みPNGと一致しないため検証しない。非正方形は正方形へ伸ばして縮小する（暫定） |
| 終了理由（11節） | `Trial.end_reason`は`controller.termination_reason`を写す（`finished`/`action_budget_exhausted`/`attempt_budget_exhausted`） |
| 学習できるAction | `erase_v2`・`create_layer`・レビュー退出（`review_reason`）はv0のPolicyが出さない |
| CLI（`__main__.py`） | `python -m image_drawer.training train <dataset> [--config JSON] [--output-dir] [--seed] [--iterations] [--resume iter_N.json] [--stop-after N]` と `evaluate <experiment_dir> --dataset <dataset> [--images 8] [--seeds 5]`。出力先の既定は `reports/training/exp-<設定ハッシュ先頭8桁>`。終了コードは 0=完了（`evaluate`の判定が`not_learned`/`inconclusive`でも0。`verdict`を見る）、1=学習が中断、2=実行できない。`evaluate`はtest splitで学習前・学習後・ランダムを同条件（温度は設定値で統一）で測り、`claim.json`・`evaluation.jsonl`を書いて`experiment.json`の`claim`を更新する。`integrity_ok`は学習前Policyの試行のReplay一致と許可Action検査の結果（target隔離は契約テストで確認） |

## 12. 実測記録（合成画像・1回）

`scripts/run_training_experiment.py --images 150 --iterations 100 --seed 0` の結果です。閾値は1.4節のまま変えていません。合成した単純な画像での測定で、実画像・高画質・収束は主張しません。

| 項目 | 値 |
| --- | --- |
| 画像 | test 21枚のうち評価に8枚、評価seed 5個 |
| 学習 | 3200 trial。`finish` 1599、`action_budget` 122、`curriculum_stop` 1479。更新100回、skip 0 |
| 判定 | `not_learned`（M1・M3・M4・M6が合格、M2・M5が不合格） |

| ID | 値 | 合格 |
| --- | --- | --- |
| M1 | `param_delta_norm`=5.81 | 合格 |
| M2 | 完走率の増分 0.0、予算超過率の増分 0.0 | **不合格**（+0.30以上が必要） |
| M3 | 品質の増分の平均 +0.169（信頼区間の下限 +0.160） | 合格 |
| M4 | 平均 +0.197（下限 +0.180） | 合格 |
| M5 | 本文を入れ替えた画像との差の余裕 平均 0.0037（下限 −0.0002） | **不合格**（下限が0以下） |
| M6 | 完走した絵の署名が17種類 | 合格 |

- **学習は進んでいます:** 100 iterationの学習中、全工程が回る50回目以降は`finish_rate`が1.00で、品質は0.61から0.73へ単調に上がりました。
- **M2の不合格:** 完走率の増分が0.0です。合法マスク（4.2節）を入れた後は、未学習のPolicyでも全trialが完走します（12.1節で完走率1.00を確認）。M2aの「+0.30以上」は構造的に満たせません。
- **M5の不合格（100 iteration）:** 本文を入れ替えても絵がほとんど変わりませんでした。12.1節のとおり、300 iterationでは合格したため、学習量の不足が主因でした。
- **測定の修正:** 旧スクリプトの合成画像は色4種と位置の組み合わせで、重複判定により数群にまとまり、test splitが2枚になって判定が必ず`inconclusive`でした。画像ごとに大きさ・位置と小ブロックを変え、test 21枚を得ています。
- **未確認:** 実画像、複数seed、より長い学習。

### 12.1 300 iterationの再測定

同じ画像・seed・評価条件で、学習だけを300 iterationにしました（再開は設定ハッシュに`iterations`が含まれるため拒否される。延長は`--iterations 300 --stop-after N`で止めて再開する）。判定は`not_learned`のままで、不合格はM2だけです。

| ID | 100 iteration | 300 iteration |
| --- | --- | --- |
| M2 完走率の増分 | 0.0 | 0.0（未学習側も完走率1.00） |
| M3 品質の増分（下限） | +0.169（+0.160） | +0.185（+0.175） |
| M4 品質の増分（下限） | +0.197（+0.180） | +0.213（+0.205） |
| M5 本文入替の余裕（下限） | 0.0037（−0.0002）**不合格** | 0.0023（+0.0004）**合格** |
| M6 完走署名の種類 | 17 | 11 |

- 評価のtrialは、未学習40・学習後40・ベースライン40。完走率はどれも1.00、予算超過は0、平均の品質は未学習0.555、学習後0.740、ベースライン0.527です。
- M5の余裕は小さい（平均0.0023、下限0.0004）。本文に応じた描き分けが「ある」とは言えるが、強いとは言えません。
- M6の署名は17から11に減りました。方策が絞られた結果と考えられますが、多様性が下がることは未検証の懸念です。
- **M2の基準の修正:** 合法マスクのもとでは未学習でも完走率が1.00になり、「+0.30以上」は学習の良し悪しと関係なく満たせません（天井効果）。測定結果に合わせた緩和ではなく、満たせる余地が無いことを理由に、M2aを「+0.30、ただし余地（1−学習前の完走率）が小さければ余地まで。学習前を下回らない」へ直しました（`evaluation.py`とテストを同時に変更）。余地が0.30以上あるときの判定は変わりません。この基準を300 iterationの記録の数値に当てはめると、M2は必要増分0・実際の増分0・予算超過の増分0で合格になる計算です（評価の再実行はしていません）。M1〜M6がすべて合格しても、合成画像1回の測定であり、実画像・複数seedでの学習成功は主張しません。

### 12.2 評価基盤（S5）での実測（validation・合成画像・1回）

12.1の`iter_300`checkpointを、`image-drawer-evaluate`でrandom方策と比べました。validation分割の14画像、推論seed 3、`--shuffle-prompts --probe --memorization-tau`を付けた実行です。`claim_level`は`none`で、学習の成功は主張できません（testでの測定ではなく、`policy_kind`も未指定、学習seedは1つ）。

| 指標 | 結果 |
| --- | --- |
| 完成率 | random 0.55 → 学習後 1.00（randomの19件は行動上限で未完成） |
| `prompt_shuffle_delta` | −0.0002（95%区間 −0.0008〜+0.0004、n=42）。promptを使っていない |
| `probe_hit_rate` | before 0.0、after 0.0（n=12）。色の指示に従えていない |
| `nearest_train_hit_rate` | after 0.0、before 0.071（τ=0.404）。学習画像のコピーは見られない |

学習できたのは「描き切ること」で、promptに応じた描き分けは学習できていません。訂正: 当初ここに「合成データはpromptが手がかりになりにくい」と書きましたが、誤りです。データセットのmanifestを確認したところ、promptは「白と赤が中心の正方形の画像（…）」のように長方形の色を含み、150枚で19種類あります。つまりpromptに色の手がかりはあるのに、300 iterationの学習ではそれを使えていません。考えられる原因（hashed-ngramのtext encoder、学習の短さ、32pxでの画像類似度Rewardの弱さなど）はいずれも未検証で、切り分けは未実施です。`generalization_gap`は、trainの先頭14件（画像ID昇順）を同じ条件で実行して計算しました。gain_train=0.0575、validationのgain=0.0677で、gap=−0.0102（rgb_128、信頼区間は未実装）です。validation側の値でありtestの値ではありません。trainだけが改善している兆候（記憶・過学習）は、この規模では見られません。`useful_stroke_rate`とOpenCLIP系は未実装です。

再現の注意: `--train-config`を省くと「設定のハッシュが一致しません」で止まります。checkpointを作った実験の`experiment.json`の`config`を`--train-config`に渡してください。
