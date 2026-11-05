"""学習設定（TrainConfig、CurriculumConfig、RewardConfig）の検証とJSON化。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# 固定の7工程。Policyが飛び先を選べない順序（固定工程表）。
STAGE_IDS: tuple[str, ...] = (
    "rough",
    "sketch",
    "final_drawing",
    "base_color",
    "shading",
    "refine",
    "finish",
)

# 学習重みを持てるReward成分名（RewardBreakdown.componentsのキーと一致）。
REWARD_COMPONENTS: tuple[str, ...] = (
    "shape",
    "outline",
    "color",
    "perceptual",
    "text_match",
    "progress",
    "action_cost",
    "final_bonus",
    "failure_penalty",
)

# 設計書4.3節: 記録・評価だけに使い、更新には使わない成分の既定値(重み0)。
_RECORD_ONLY_COMPONENTS = frozenset({"shape", "text_match"})


# {
#   責務: [
#     RewardConfig: Reward成分の学習重みと正本の定数を保持する
#   ]
#   フィールド: [
#     reward_profile_id: stage_reward_v2.jsonのreward_profile_id
#     reward_impl_version: 暫定数値化の版
#     training_weights: 成分名から学習重みへの対応。0は記録のみ
#     action_cost_scale: 全attemptに課すコストの係数(0.05/max_actions)
#     final_bonus_scale: final_qualityに掛ける係数
#     failure_penalty: 予算超過などの終了時の加点(負)
#   ]
# }
@dataclass(frozen=True)
class RewardConfig:
    reward_profile_id: str = "image-only-stage-reward-baseline-v2"
    reward_impl_version: str = "stage-reward-impl-v0"
    training_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            name: 0.0 if name in _RECORD_ONLY_COMPONENTS else 1.0 for name in REWARD_COMPONENTS
        }
    )
    action_cost_scale: float = 0.05
    final_bonus_scale: float = 2.0
    failure_penalty: float = -1.0

    # {
    #   責務: [__post_init__: 成分名・重み・係数の範囲を検証する]
    #   処理: [
    #     1: 未知の成分名と欠落を拒否する
    #     2: 重みと係数が有限で、重みが0以上であることを確認する
    #   ]
    #   エラー: [ValueError: 成分名の過不足、非有限値、負の重み]
    # }
    def __post_init__(self) -> None:
        # 未知の成分名を黙って無視すると、重みの付け間違いが学習に効かないままになる。
        names = set(self.training_weights)
        if names != set(REWARD_COMPONENTS):
            raise ValueError(
                f"training_weightsの成分名が不正です: 不足={sorted(set(REWARD_COMPONENTS) - names)}"
                f" 未知={sorted(names - set(REWARD_COMPONENTS))}"
            )
        for name, weight in self.training_weights.items():
            _require_finite(f"training_weights[{name}]", weight)
            if weight < 0:
                raise ValueError(f"training_weights[{name}]は0以上にしてください: {weight}")
        _require_finite("action_cost_scale", self.action_cost_scale)
        _require_finite("final_bonus_scale", self.final_bonus_scale)
        _require_finite("failure_penalty", self.failure_penalty)
        if self.action_cost_scale < 0 or self.final_bonus_scale < 0 or self.failure_penalty > 0:
            raise ValueError("action_cost_scale/final_bonus_scaleは0以上、failure_penaltyは0以下")
        if not self.reward_profile_id or not self.reward_impl_version:
            raise ValueError("reward_profile_idとreward_impl_versionは空にできません")


# {
#   責務: [
#     CurriculumConfig: 工程を増やしながら学習する段階を保持する
#   ]
#   フィールド: [
#     enabled: カリキュラムを使うか
#     stage_counts: 各段階で描く先頭工程の数(1〜6)。最後は全7工程(7)でなければならない
#     iterations_per_stage: 各段階に割り当てるiteration数(stage_countsと同じ長さ)
#   ]
# }
@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = False
    stage_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    iterations_per_stage: tuple[int, ...] = (10, 10, 10, 10, 10, 10, 140)

    # {
    #   責務: [__post_init__: 段階の長さ・単調増加・最終段階を検証する]
    #   エラー: [ValueError: 長さの不一致、範囲外、非単調、最終段階が全工程でない]
    # }
    def __post_init__(self) -> None:
        if len(self.stage_counts) != len(self.iterations_per_stage) or not self.stage_counts:
            raise ValueError("stage_countsとiterations_per_stageは同じ長さの非空にしてください")
        if any(not 1 <= count <= len(STAGE_IDS) for count in self.stage_counts):
            raise ValueError(f"stage_countsは1〜{len(STAGE_IDS)}にしてください")
        if list(self.stage_counts) != sorted(set(self.stage_counts)):
            raise ValueError("stage_countsは重複なく増加させてください")
        if self.stage_counts[-1] != len(STAGE_IDS):
            raise ValueError("最終段階は全工程で学習してください(完成の主張は全工程のみ)")
        if any(count <= 0 for count in self.iterations_per_stage):
            raise ValueError("iterations_per_stageは正の整数にしてください")

    # {
    #   責務: [stop_after_stage: iterationに対応する打切り工程を返す]
    #   引数: [iteration: 0始まりのiteration番号]
    #   戻り値: [stage_id: 打切り工程。全工程の段階またはカリキュラム無効ならNone]
    # }
    def stop_after_stage(self, iteration: int) -> str | None:
        if not self.enabled:
            return None
        remaining = iteration
        for count, length in zip(self.stage_counts, self.iterations_per_stage):
            if remaining < length:
                return None if count == len(STAGE_IDS) else STAGE_IDS[count - 1]
            remaining -= length
        # 予定を超えたiterationは最終段階(全工程)として扱う。
        return None


# {
#   責務: [
#     TrainConfig: Trainerの全設定を保持し、検証・JSON化・ハッシュ化を担う
#   ]
#   フィールド: [
#     master_seed: SeedSequenceの根
#     text_encoder_id: register_text_encoderで登録したID
#     canvas_side: キャンバス一辺(画素)
#     max_actions: 受理Actionの上限
#     batch_images / seeds_per_image: B・K
#     iterations: 総iteration数
#     その他: 設計書4.2・4.3節の学習ハイパーパラメータ
#   ]
# }
@dataclass(frozen=True)
class TrainConfig:
    master_seed: int = 0
    text_encoder_id: str = "hashed-ngram-v1"
    canvas_side: int = 32
    max_actions: int = 96
    batch_images: int = 4
    seeds_per_image: int = 8
    iterations: int = 200
    gamma: float = 0.99
    baseline_ema: float = 0.1
    entropy_coef: float = 0.01
    grad_clip_norm: float = 5.0
    learning_rate: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sigma_start: float = 0.15
    sigma_end: float = 0.05
    stroke_samples: int = 6
    tail_length: int = 8
    summary_side: int = 8
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # {
    #   責務: [__post_init__: 数値設定の範囲を検証する]
    #   エラー: [ValueError: 範囲外・非有限・空のID]
    # }
    def __post_init__(self) -> None:
        if not self.text_encoder_id:
            raise ValueError("text_encoder_idは空にできません")
        for name in (
            "canvas_side", "max_actions", "batch_images", "seeds_per_image",
            "iterations", "stroke_samples", "tail_length", "summary_side",
        ):  # fmt: skip
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name}は1以上の整数にしてください")
        if self.stroke_samples < 2:
            raise ValueError("stroke_samplesは2以上にしてください")
        for name in (
            "gamma", "baseline_ema", "entropy_coef", "grad_clip_norm", "learning_rate",
            "adam_beta1", "adam_beta2", "sigma_start", "sigma_end",
        ):  # fmt: skip
            _require_finite(name, getattr(self, name))
        if not 0 < self.gamma <= 1 or not 0 < self.baseline_ema <= 1:
            raise ValueError("gammaとbaseline_emaは(0,1]にしてください")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("adamのbetaは[0,1)にしてください")
        if self.entropy_coef < 0 or self.grad_clip_norm <= 0 or self.learning_rate <= 0:
            raise ValueError("entropy_coefは0以上、grad_clip_normとlearning_rateは正にしてください")
        if not 0 < self.sigma_end <= self.sigma_start:
            raise ValueError("sigmaは 0 < sigma_end <= sigma_start にしてください")

    # {
    #   責務: [max_attempts: attempt上限を契約どおり max(100, 5*max_actions) で返す]
    #   戻り値: [上限: attempt上限]
    # }
    @property
    def max_attempts(self) -> int:
        return max(100, 5 * self.max_actions)

    # {
    #   責務: [sigma_at: iterationに応じたストロークのガウス幅を直線減衰で返す]
    #   引数: [iteration: 0始まりのiteration番号]
    #   戻り値: [sigma: sigma_startからsigma_endの間の値]
    # }
    def sigma_at(self, iteration: int) -> float:
        if self.iterations <= 1:
            return self.sigma_end
        ratio = min(max(iteration / (self.iterations - 1), 0.0), 1.0)
        return self.sigma_start + (self.sigma_end - self.sigma_start) * ratio

    # {
    #   責務: [to_dict: 全項目をJSON互換の辞書にする]
    #   戻り値: [data: ExperimentRecord.configへ入れる辞書]
    # }
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["curriculum"]["stage_counts"] = list(self.curriculum.stage_counts)
        data["curriculum"]["iterations_per_stage"] = list(self.curriculum.iterations_per_stage)
        data["reward"]["training_weights"] = dict(self.reward.training_weights)
        return data

    # {
    #   責務: [from_dict: to_dictの結果から検証付きで復元する]
    #   引数: [data: to_dictの出力]
    #   戻り値: [config: 復元したTrainConfig]
    #   エラー: [ValueError: 未知のキーまたは不正な値]
    # }
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        values = dict(data)
        reward = RewardConfig(**dict(values.pop("reward", {})))
        curriculum_raw = dict(values.pop("curriculum", {}))
        for key in ("stage_counts", "iterations_per_stage"):
            if key in curriculum_raw:
                curriculum_raw[key] = tuple(curriculum_raw[key])
        try:
            return cls(**values, reward=reward, curriculum=CurriculumConfig(**curriculum_raw))
        except TypeError as error:
            raise ValueError(f"TrainConfigに未知のキーがあります: {error}") from error

    # {
    #   責務: [config_hash: 設定のSHA-256を返す(checkpointのconfig_hash用)]
    #   戻り値: [digest: 16進文字列]
    # }
    def config_hash(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# {
#   責務: [_require_finite: 数値が有限の実数であることを確認する]
#   エラー: [ValueError: 実数でない、またはNaN/Inf]
# }
def _require_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}は数値にしてください: {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name}は有限の数値にしてください: {value!r}")
