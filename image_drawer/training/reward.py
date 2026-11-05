"""RewardEvaluator: trial終了後に、目標画像との比較でRewardを一括計算する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from image_drawer.dataset import RewardTarget
from image_drawer.training.config import REWARD_COMPONENTS, RewardConfig
from image_drawer.training.reward_features import (
    REWARD_IMPL_VERSION,
    composite_over_white,
    compute_features,
)
from image_drawer.training.reward_spec import StageRewardSpec, default_stage_reward_spec
from image_drawer.training.rollout import Trial

BreakdownKind = Literal["step", "invalid", "final", "failure", "boundary"]
_GROUP_OF_KIND = {"edge": "outline", "rgb": "color", "rgb_hist": "color", "luma": "perceptual"}
_FOREGROUND_THRESHOLD = 0.1  # 白からこの距離より離れた画素を前景とみなす(shape成分)。
FAILURE_END_REASONS = ("action_budget", "attempt_budget")


# {
#   責務: [ TargetFeatures: 評価用に縮小した目標画像。1画像1回だけ作って再利用する ]
# }
@dataclass(frozen=True)
class TargetFeatures:
    image_id: str
    canvas_side: int
    rgb: NDArray[np.float64]


# {
#   責務: [ RewardBreakdown: 1試行分の成分別Rewardと、学習に使うtotal ]
# }
@dataclass(frozen=True)
class RewardBreakdown:
    stage_id: str
    kind: BreakdownKind
    features: Mapping[str, float]
    effective_resolution: Mapping[str, int]
    potential_before: float | None
    potential_after: float | None
    components: Mapping[str, float]
    total: float
    reward_profile_id: str
    reward_impl_version: str
    attempt_index: int = 0


# {
#   責務: [ TrialReward: trial全体の成分別Rewardとfinal_quality ]
# }
@dataclass(frozen=True)
class TrialReward:
    steps: tuple[RewardBreakdown, ...]
    final_quality: float | None
    total_return: float


# {
#   責務: [ foreground_iou: 白から離れた画素(前景)のIoU。両方空なら1 ]
# }
def foreground_iou(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    a = (1.0 - first).max(axis=2) > _FOREGROUND_THRESHOLD
    b = (1.0 - second).max(axis=2) > _FOREGROUND_THRESHOLD
    union = int((a | b).sum())
    return 1.0 if union == 0 else float((a & b).sum() / union)


# {
#   責務: [ _Level: ある工程・合成画像でのpotentialとその特徴群別の寄与(内部用) ]
# }
@dataclass(frozen=True)
class _Level:
    total: float
    groups: Mapping[str, float]
    features: Mapping[str, float]
    resolutions: Mapping[str, int]
    shape: float


# {
#   責務: [
#     RewardEvaluator: 目標特徴を使ってtrialを採点する。Policy・RolloutRunnerからは参照されない
#   ]
#   フィールド: [ config, canvas_side, spec: 工程別重み, calls: evaluate_trialの呼出し回数(検査用) ]
# }
class RewardEvaluator:
    def __init__(
        self, config: RewardConfig, canvas_side: int, spec: StageRewardSpec | None = None
    ) -> None:
        self.config = config
        self.canvas_side = canvas_side
        self.spec = spec or default_stage_reward_spec()
        self.calls = 0

    # {
    #   責務: [ prepare_target: 目標画像を読み、canvas_sideへ縮小して保持する(ファイルは1回だけ読む) ]
    #   処理: [ 透明部は白へ合成し、非正方形は正方形へ伸ばす(暫定)。縮小は箱フィルタ ]
    #   エラー: [ ValueError: 画像を開けない ]
    # }
    def prepare_target(self, target: RewardTarget) -> TargetFeatures:
        from PIL import Image  # extras trainに含まれるPillow。目標を読むのはここだけ。

        try:
            with Image.open(target.image_path) as opened:
                rgba = opened.convert("RGBA")
        except OSError as error:
            raise ValueError(f"目標画像を開けません: {target.image_id}") from error
        side = self.canvas_side
        small = rgba.resize((side, side), Image.Resampling.BOX)
        rgb = composite_over_white(small.tobytes(), side)
        return TargetFeatures(target.image_id, side, rgb)

    # {
    #   責務: [ _level: 工程のpotentialを計算する。重みの無い工程(finish)は0 ]
    # }
    def _level(self, stage_id: str, rgb: NDArray[np.float64], target: TargetFeatures) -> _Level:
        weights = self.spec.potential_weights.get(stage_id, {})
        shape = foreground_iou(rgb, target.rgb)
        if not weights:
            return _Level(0.0, {}, {}, {}, shape)
        scores, resolutions = compute_features(rgb, target.rgb, self.canvas_side, list(weights))
        groups: dict[str, float] = {}
        for name, weight in weights.items():
            group = _GROUP_OF_KIND[name.rsplit("_", 1)[0]]
            groups[group] = groups.get(group, 0.0) + weight * scores[name]
        return _Level(sum(groups.values()), groups, scores, resolutions, shape)

    # {
    #   責務: [ final_quality: final_quality_weightsで重み付けした完成画像の類似度 ]
    # }
    def final_quality(self, rgb: NDArray[np.float64], target: TargetFeatures) -> float:
        weights = self.spec.final_weights
        scores, _ = compute_features(rgb, target.rgb, self.canvas_side, list(weights))
        return float(sum(weights[name] * scores[name] for name in weights))

    # {
    #   責務: [ _make: 成分から重み合成したtotal付きのRewardBreakdownを作る ]
    # }
    def _make(self, stage_id: str, kind: BreakdownKind, attempt: int, level: _Level | None,
              before: float | None, after: float | None, **components: float) -> RewardBreakdown:  # fmt: skip
        values = {name: float(components.get(name, 0.0)) for name in REWARD_COMPONENTS}
        weights = self.config.training_weights
        return RewardBreakdown(
            stage_id=stage_id,
            kind=kind,
            features=dict(level.features) if level else {},
            effective_resolution=dict(level.resolutions) if level else {},
            potential_before=before,
            potential_after=after,
            components=values,
            total=sum(weights[name] * values[name] for name in REWARD_COMPONENTS),
            reward_profile_id=self.config.reward_profile_id,
            reward_impl_version=self.config.reward_impl_version or REWARD_IMPL_VERSION,
            attempt_index=attempt,
        )

    # {
    #   責務: [ evaluate_trial: trialの全試行を採点する ]
    #   処理: [
    #     1: 白紙のroughを基準に、受理Actionごとにpotentialの差(progress)を求める
    #     2: 拒否は-action_costだけ。工程を進めるActionは新工程の入場時点を新しい基準にする
    #     3: finishしたら2.0*final_quality、予算超過なら最後の試行へ失敗ペナルティを加える
    #   ]
    #   引数: [ trial, target: prepare_targetの結果, training_prompt: 記録用(text_matchは未計算) ]
    #   戻り値: [ reward ]
    #   エラー: [ ValueError: 目標のキャンバス寸法がtrialと違う ]
    # }
    def evaluate_trial(
        self, trial: Trial, target: TargetFeatures, training_prompt: str
    ) -> TrialReward:
        self.calls += 1
        side = self.canvas_side
        if target.canvas_side != side:
            raise ValueError("目標のキャンバス寸法がEvaluatorと一致しません")
        budget = trial.episode.budget or {}
        cost = self.config.action_cost_scale / max(1, int(budget.get("max_actions", 1)))
        stage = "rough"
        previous = np.ones((side, side, 3))
        previous_bytes: bytes | None = None
        level = self._level(stage, previous, target)
        breakdowns: list[RewardBreakdown] = []
        final_quality: float | None = None

        for step in trial.steps:
            if not step.accepted:
                breakdowns.append(
                    self._make(
                        step.stage_id,
                        "invalid",
                        step.attempt_index,
                        None,
                        None,
                        None,
                        action_cost=-cost,
                    )  # fmt: skip
                )
                continue
            assert step.composite_after is not None
            changed = step.composite_after != previous_bytes
            rgb = composite_over_white(step.composite_after, side) if changed else previous
            if step.proposal.type == "advance_stage":
                new_stage = step.events[-1].stage_id or stage
                new_level = self._level(new_stage, rgb, target)
                kind: BreakdownKind = "boundary"
                bonus = 0.0
                if new_stage == "finish" and trial.end_reason == "finish":
                    final_quality = self.final_quality(rgb, target)
                    kind, bonus = "final", self.config.final_bonus_scale * final_quality
                breakdowns.append(
                    self._make(
                        step.stage_id,
                        kind,
                        step.attempt_index,
                        new_level,
                        level.total,
                        new_level.total,
                        action_cost=-cost,
                        final_bonus=bonus,
                        shape=new_level.shape,
                    )  # fmt: skip
                )
                stage, level = new_stage, new_level
            else:
                new_level = self._level(stage, rgb, target) if changed else level
                groups = {
                    name: new_level.groups.get(name, 0.0) - level.groups.get(name, 0.0)
                    for name in ("outline", "color", "perceptual")
                }
                breakdowns.append(
                    self._make(
                        step.stage_id,
                        "step",
                        step.attempt_index,
                        new_level,
                        level.total,
                        new_level.total,
                        progress=new_level.total - level.total,
                        action_cost=-cost,
                        shape=new_level.shape,
                        **groups,
                    )  # fmt: skip
                )
                level = new_level
            previous, previous_bytes = rgb, step.composite_after

        if trial.end_reason in FAILURE_END_REASONS and breakdowns:
            last = breakdowns[-1]
            components = {**last.components, "failure_penalty": self.config.failure_penalty}
            weights = self.config.training_weights
            breakdowns[-1] = replace(
                last,
                kind="failure",
                components=components,
                total=sum(weights[name] * components[name] for name in REWARD_COMPONENTS),
            )
        return TrialReward(
            tuple(breakdowns), final_quality, float(sum(b.total for b in breakdowns))
        )
