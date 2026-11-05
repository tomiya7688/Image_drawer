"""REINFORCE(工程別baseline、エントロピー項、勾配クリップ、Adam)による更新。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from image_drawer.training.config import TrainConfig
from image_drawer.training.policy import DrawingPolicy, PolicySample
from image_drawer.training.policy_params import AdamState
from image_drawer.training.reward import TrialReward
from image_drawer.training.rollout import Trial

NON_UPDATING_END_REASONS = ("policy_error", "engine_error")
MIN_ADVANTAGE_STD = 1e-12


# {
#   責務: [ UpdateStats: 1回の更新の記録。skippedなら更新していない ]
# }
@dataclass(frozen=True)
class UpdateStats:
    skipped: bool
    skip_reason: str | None
    step_count: int
    mean_return: float
    advantage_std: float
    grad_norm: float
    param_delta_norm: float
    mean_entropy: float


# {
#   責務: [ BaselineTable: 工程別の収益の指数移動平均。未観測の工程は最初のbatch平均で初期化する ]
# }
class BaselineTable:
    def __init__(self, values: Mapping[str, float] | None = None) -> None:
        self.values: dict[str, float] = dict(values or {})

    # {
    #   責務: [ get: 工程のbaseline。未観測ならNone ]
    # }
    def get(self, stage_id: str) -> float | None:
        return self.values.get(stage_id)

    # {
    #   責務: [ update: 指数移動平均 b <- (1-alpha) b + alpha mean を進める(初回はmeanで初期化) ]
    # }
    def update(self, stage_id: str, batch_mean: float, alpha: float) -> None:
        previous = self.values.get(stage_id)
        self.values[stage_id] = (
            batch_mean if previous is None else (1.0 - alpha) * previous + alpha * batch_mean
        )


# {
#   責務: [ discounted_returns: G_t = sum_{u>=t} gamma^(u-t) r_u を末尾から求める ]
# }
def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


# {
#   責務: [ collect_batch: 更新に使うtrialから(sample, G, stage_id)を集める。例外trialは除く ]
#   エラー: [ ValueError: trialと報酬の試行数が一致しない ]
# }
def collect_batch(
    pairs: Sequence[tuple[Trial, TrialReward]], gamma: float
) -> list[tuple[PolicySample, float, str]]:
    batch: list[tuple[PolicySample, float, str]] = []
    for trial, reward in pairs:
        if trial.end_reason in NON_UPDATING_END_REASONS:
            continue
        if len(trial.steps) != len(reward.steps):
            raise ValueError("trialと報酬の試行数が一致しません")
        returns = discounted_returns([b.total for b in reward.steps], gamma)
        batch.extend((step.sample, g, step.stage_id) for step, g in zip(trial.steps, returns))
    return batch


# {
#   責務: [ reinforce_update: 方策勾配でpolicyを1回更新する ]
#   処理: [
#     1: 収益から工程別baselineを引いてAdvantageを作り、batch全体で標準化する
#     2: 標準偏差が0なら更新しない(記録だけ残す)
#     3: sum A grad logpi / N + beta grad H を作り、全体ノルム5でクリップしてAdamで更新する
#     4: baselineを指数移動平均で進める
#   ]
#   引数: [ policy, adam, pairs: (trial, 報酬), baselines, config ]
#   戻り値: [ stats ]
# }
def reinforce_update(
    policy: DrawingPolicy,
    adam: AdamState,
    pairs: Sequence[tuple[Trial, TrialReward]],
    baselines: BaselineTable,
    config: TrainConfig,
) -> UpdateStats:
    batch = collect_batch(pairs, config.gamma)
    if not batch:
        return UpdateStats(True, "no_usable_steps", 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    returns = np.array([g for _s, g, _t in batch])
    # baselineは更新前の値を使う。未観測の工程はこのbatchの平均で代用する。
    stage_means = {
        stage: float(np.mean([g for _s, g, t in batch if t == stage]))
        for stage in {t for _s, _g, t in batch}
    }
    bases = np.array(
        [
            stage_means[stage] if baselines.get(stage) is None else baselines.get(stage)
            for _s, _g, stage in batch
        ]
    )
    advantages = returns - bases
    std = float(advantages.std())
    for stage, mean in stage_means.items():
        baselines.update(stage, mean, config.baseline_ema)
    if std < MIN_ADVANTAGE_STD:
        return UpdateStats(
            True, "zero_advantage_std", len(batch), float(returns.mean()), std, 0.0, 0.0, 0.0
        )
    advantages = (advantages - advantages.mean()) / std

    grads = policy.new_grads()
    count = len(batch)
    entropy_total = 0.0
    for (sample, _g, _stage), advantage in zip(batch, advantages):
        policy.accumulate_log_prob_grad(sample, float(advantage) / count, grads)
        entropy_total += policy.accumulate_entropy_grad(sample, config.entropy_coef / count, grads)
    grad_norm = grads.clip(config.grad_clip_norm)
    deltas = adam.step(grads.arrays, config.learning_rate, config.adam_beta1, config.adam_beta2)
    policy.apply_update(deltas)
    delta_norm = float(np.sqrt(sum(float((d**2).sum()) for d in deltas.values())))
    return UpdateStats(
        False, None, count, float(returns.mean()), std, grad_norm, delta_norm, entropy_total / count
    )


# {
#   責務: [ mean_log_prob: 記録済みサンプルの対数確率の平均(勾配は足さない)。更新前後の比較用 ]
# }
def mean_log_prob(policy: DrawingPolicy, samples: Sequence[PolicySample]) -> float:
    grads = policy.new_grads()
    return float(np.mean([policy.accumulate_log_prob_grad(s, 0.0, grads) for s in samples]))
