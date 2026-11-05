"""Policy x StageControllerのループ(rollout)。評価器も目標も受け取らない。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from image_drawer.dataset import PolicyInput
from image_drawer.drawing import RENDERER_V2, DrawingAction, DrawingEpisode, Engine
from image_drawer.drawing.stage import Budget, StageController, judge_style_intent
from image_drawer.training.config import STAGE_IDS, TrainConfig
from image_drawer.training.observation import ObservationBuilder
from image_drawer.training.policy import DrawingPolicy, PolicySample
from image_drawer.training.text_encoder import TextEncoder

EndReason = Literal[
    "finish", "action_budget", "attempt_budget", "curriculum_stop", "policy_error", "engine_error"
]
_END_REASON_BY_TERMINATION: dict[str, EndReason] = {
    "finished": "finish",
    "action_budget_exhausted": "action_budget",
    "attempt_budget_exhausted": "attempt_budget",
}
ALLOWED_EVENT_ACTORS = ("model", "system")


# {
#   責務: [ TrialStep: 試行(受理または拒否)1件。勾配用のPolicySampleと結果を持つ ]
# }
@dataclass(frozen=True)
class TrialStep:
    attempt_index: int
    stage_id: str
    sample: PolicySample
    proposal: DrawingAction
    accepted: bool
    reject_reason: str | None
    events: tuple[DrawingAction, ...]
    composite_after: bytes | None


# {
#   責務: [
#     Trial: 1枚の描画試行の記録。actorはmodel固定で、human/syntheticのEpisodeを受け付けない
#   ]
#   エラー: [ ValueError: actorが"model"でない、またはEpisodeにmodel/system以外のeventがある ]
# }
@dataclass(frozen=True)
class Trial:
    trial_id: str
    image_id: str
    training_prompt: str
    actor: Literal["model"]
    seed: int
    policy_version: str
    policy_sha256: str
    steps: tuple[TrialStep, ...]
    end_reason: EndReason
    final_composite: bytes | None
    episode: DrawingEpisode
    stop_after_stage: str | None

    def __post_init__(self) -> None:
        if self.actor != "model":
            raise ValueError(f"Trialのactorはmodel固定です: {self.actor!r}")
        foreign = {e.actor for e in self.episode.events} - set(ALLOWED_EVENT_ACTORS)
        if foreign:
            raise ValueError(f"学習に使えないactorのeventがあります: {sorted(foreign)}")


# {
#   責務: [ TrainingRollout: 1 iterationの全trialと、採点後に結合した報酬(同順) ]
# }
@dataclass(frozen=True)
class TrainingRollout:
    rollout_id: str
    iteration: int
    trials: tuple[Trial, ...]
    rewards: tuple  # tuple[TrialReward, ...]。reward.pyの型(循環importを避ける)


# {
#   責務: [
#     RolloutRunner: PolicyにObservationだけを渡し、提案をStageControllerへ送って記録する
#   ]
#   フィールド: [ policy, text_encoder, observation_builder, config ]
#   処理: [ 目標画像への参照・評価器を引数にも属性にも持たない ]
# }
class RolloutRunner:
    def __init__(
        self,
        policy: DrawingPolicy,
        text_encoder: TextEncoder,
        observation_builder: ObservationBuilder,
        config: TrainConfig,
    ) -> None:
        self.policy = policy
        self.text_encoder = text_encoder
        self.observation_builder = observation_builder
        self.config = config

    # {
    #   責務: [ _is_past_stop: 打切り工程を抜けた(次の工程に入った)か ]
    # }
    @staticmethod
    def _is_past_stop(stage_id: str, stop_after_stage: str | None) -> bool:
        if stop_after_stage is None:
            return False
        return STAGE_IDS.index(stage_id) > STAGE_IDS.index(stop_after_stage)

    # {
    #   責務: [ run_trial: 1枚を描かせ、全試行を記録したTrialを返す ]
    #   処理: [
    #     1: Engine・Controllerを作る 2: 終了まで 観測->act->propose を繰り返す
    #     3: 例外はpolicy_error/engine_errorとして記録 4: Episodeと終了時の合成を保存する
    #   ]
    #   引数: [ policy_input: promptだけを使う(image_idは記録用), seed, stop_after_stage: 打切り工程 ]
    #   戻り値: [ trial: 記録 ]
    # }
    def run_trial(
        self, policy_input: PolicyInput, seed: int, stop_after_stage: str | None
    ) -> Trial:
        config = self.config
        side = config.canvas_side
        engine = Engine(side, side, RENDERER_V2, seed=seed & 0xFFFFFFFF)
        budget = Budget(config.max_actions, config.max_attempts)
        controller = StageController(engine, judge_style_intent(policy_input.prompt), budget)
        text_condition = self.text_encoder.encode(policy_input.prompt)
        rng = np.random.Generator(np.random.PCG64(seed))
        steps: list[TrialStep] = []
        end_reason: EndReason | None = None

        while not controller.closed:
            if self._is_past_stop(controller.stage_id, stop_after_stage):
                end_reason = "curriculum_stop"
                break
            try:
                observation = self.observation_builder.build(
                    text_condition, controller, engine, budget
                )
                decision = self.policy.act(observation, rng)
            except Exception:
                end_reason = "policy_error"
                break
            stage_id = controller.stage_id
            try:
                result = controller.propose(decision.action)
            except Exception:
                end_reason = "engine_error"
                break
            composite = engine.render().rgba if result.accepted else None
            steps.append(
                TrialStep(
                    result.attempt,
                    stage_id,
                    decision.sample,
                    decision.action,
                    result.accepted,
                    result.reason,
                    tuple(result.events),
                    composite,
                )  # fmt: skip
            )
        if end_reason is None:
            end_reason = _END_REASON_BY_TERMINATION.get(
                controller.termination_reason or "", "engine_error"
            )
        return Trial(
            trial_id=f"{policy_input.image_id}-{seed}",
            image_id=policy_input.image_id,
            training_prompt=policy_input.prompt,
            actor="model",
            seed=seed,
            policy_version=self.policy.policy_version,
            policy_sha256=self.policy.params_sha256(),
            steps=tuple(steps),
            end_reason=end_reason,
            final_composite=engine.render().rgba,
            episode=controller.episode(),
            stop_after_stage=stop_after_stage,
        )
