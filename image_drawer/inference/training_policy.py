"""training（numpyが必要）のDrawingPolicyを推論のLoadedPolicyにする。

観測は学習と同じ`ObservationBuilder`で作る。RewardEvaluator・Datasetは読み込まない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from image_drawer.drawing import DrawingAction, Engine
from image_drawer.drawing.stage import Budget, StageController
from image_drawer.inference.policy import LoadedPolicy, PolicyError, PolicyRegistry, PolicySession
from image_drawer.training.checkpoint import load_checkpoint
from image_drawer.training.config import TrainConfig
from image_drawer.training.observation import ObservationBuilder
from image_drawer.training.optimizer import BaselineTable
from image_drawer.training.policy import DrawingPolicy, ParametricPolicy, RandomLegalPolicy
from image_drawer.training.policy_params import AdamState
from image_drawer.training.text_encoder import (
    CheckpointMismatchError,
    TextEncoder,
    get_text_encoder,
)

RANDOM_LEGAL_ID = "random-legal-v1"


# {
#   責務: [ _TrainingSession: promptを1回符号化し、観測->actで次のActionを返す（乱数はseedだけ） ]
# }
class _TrainingSession:
    def __init__(
        self,
        policy: DrawingPolicy,
        encoder: TextEncoder,
        builder: ObservationBuilder,
        prompt: str,
        seed: int,
    ) -> None:
        self._policy = policy
        self._builder = builder
        self._condition = encoder.encode(prompt)
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def next_action(
        self, controller: StageController, engine: Engine, budget: Budget
    ) -> DrawingAction:
        observation = self._builder.build(self._condition, controller, engine, budget)
        return self._policy.act(observation, self._rng).action


# {
#   責務: [ loaded_from_training: DrawingPolicyとTextEncoderをLoadedPolicyにする ]
#   引数: [ trained: 学習済みか, checkpoint_sha256: 無ければ空 ]
# }
def loaded_from_training(
    policy_id: str,
    policy: DrawingPolicy,
    encoder: TextEncoder,
    config: TrainConfig,
    *,
    trained: bool,
    checkpoint_sha256: str = "",
) -> LoadedPolicy:
    builder = ObservationBuilder(config.tail_length, config.summary_side)

    def open_session(prompt: str, seed: int) -> PolicySession:
        return _TrainingSession(policy, encoder, builder, prompt, seed)

    side = config.canvas_side
    return LoadedPolicy(
        policy_id, policy.policy_version, trained, checkpoint_sha256, (side, side), open_session
    )


# {
#   責務: [ register_random_legal: 学習しない合法手ランダムを未学習・テスト用として登録する ]
# }
def register_random_legal(registry: PolicyRegistry, config: TrainConfig | None = None) -> None:
    config = config or TrainConfig()
    encoder = get_text_encoder(config.text_encoder_id)
    registry.register(
        loaded_from_training(
            RANDOM_LEGAL_ID, RandomLegalPolicy(config), encoder, config, trained=False
        )
    )


# {
#   責務: [ register_checkpoint: checkpoint(sidecar JSON)を検証して学習済みPolicyとして登録する ]
#   処理: [ ガウス幅は学習終了時のsigma_endに固定する。学習済みと呼ぶだけで品質は評価しない ]
#   エラー: [ PolicyError: checkpointの不一致・読めない ]
# }
def register_checkpoint(
    registry: PolicyRegistry, policy_id: str, sidecar_path: Path, config: TrainConfig
) -> None:
    encoder = get_text_encoder(config.text_encoder_id)
    policy = ParametricPolicy(config, encoder.dim, np.random.default_rng(config.master_seed))
    try:
        load_checkpoint(
            sidecar_path, policy, AdamState(policy.state_dict()), BaselineTable(), encoder, config
        )
    except CheckpointMismatchError as error:
        raise PolicyError(f"checkpointを使えません: {error}") from error
    policy.stroke_sigma = config.sigma_end
    digest = json.loads(sidecar_path.read_text(encoding="utf-8"))["npz_sha256"]
    registry.register(
        loaded_from_training(
            policy_id, policy, encoder, config, trained=True, checkpoint_sha256=digest
        )
    )
