"""training用テストの共通補助(台本Policy、合成画像、完走台本)。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from image_drawer.dataset import PolicyInput
from image_drawer.drawing import DrawingAction
from image_drawer.training import HashedNgramTextEncoder, TrainConfig
from image_drawer.training import action_codec as codec
from image_drawer.training.observation import Observation, ObservationBuilder
from image_drawer.training.policy import PolicyDecision, PolicySample
from image_drawer.training.policy_params import PolicyGrads, params_sha256
from image_drawer.training.rollout import RolloutRunner, Trial

SMALL = dict(canvas_side=16, summary_side=4, max_actions=60)


# {
#   責務: [ small_config: 小さなキャンバスのTrainConfigを返す ]
# }
def small_config(**overrides) -> TrainConfig:
    return TrainConfig(**{**SMALL, **overrides})


# {
#   責務: [ dummy_sample: 勾配計算に使わないテスト用のPolicySample ]
# }
def dummy_sample(stage_id: str = "rough") -> PolicySample:
    return PolicySample(
        np.zeros(4),
        stage_id,
        np.ones(6, dtype=bool),
        np.ones(16, dtype=bool),
        codec.ActionChoice(5),
        0.1,
    )


# {
#   責務: [ ScriptedPolicy: 台本のActionを順に返すPolicy(学習しない)。tailを超えたらtail_actionを返す ]
# }
class ScriptedPolicy:
    policy_version = "scripted-test"
    stroke_sigma = 0.1

    def __init__(self, actions: Sequence[DrawingAction], raise_at: int | None = None) -> None:
        self.actions = list(actions)
        self.raise_at = raise_at
        self.calls = 0

    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision:
        if not observation.action_tail:
            self.calls = 0  # 新しいtrialの開始
        index = self.calls
        self.calls += 1
        if self.raise_at is not None and index >= self.raise_at:
            raise RuntimeError("scripted failure")
        action = self.actions[index] if index < len(self.actions) else self.actions[-1]
        return PolicyDecision(action, dummy_sample(observation.stage_id))

    def accumulate_log_prob_grad(self, sample, weight, grads: PolicyGrads) -> float:
        return 0.0

    def accumulate_entropy_grad(self, sample, weight, grads: PolicyGrads) -> float:
        return 0.0

    def new_grads(self) -> PolicyGrads:
        return PolicyGrads({})

    def apply_update(self, deltas: Mapping[str, np.ndarray]) -> None:
        return None

    def state_dict(self) -> dict[str, np.ndarray]:
        return {}

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        return None

    def params_sha256(self) -> str:
        return params_sha256({})


# {
#   責務: [ stroke: 2点(x0,y0)->(x1,y1)を[0,1]で指定するstroke_v2。中間点は線上 ]
# }
def stroke(x0: float, y0: float, x1: float, y1: float, side: int = 16) -> DrawingAction:
    def logit(p: float) -> float:
        p = min(max(p, 1e-4), 1 - 1e-4)
        return float(np.log(p / (1 - p)))

    mid = ((x0 + x1) / 2, (y0 + y1) / 2)
    values = (x0, y0, *mid, x1, y1)
    return codec.decode_action(
        codec.ActionChoice(4, stroke_u=tuple(logit(v) for v in values)),
        _FakeObservation, side, 6, 5,
    )  # fmt: skip


_FakeObservation = None  # stroke以外ではobservationを使わない。


# {
#   責務: [ complete_script: 6工程すべてを描いてfinishまで進める台本 ]
#   引数: [ strokes: 各工程で描く(x0,y0,x1,y1)。1つならすべての工程で同じ線 ]
# }
def complete_script(line=(0.1, 0.5, 0.9, 0.5), color_id: int = 0) -> list[DrawingAction]:
    layers = ["sys_rough_001", "sys_sketch_001", "sys_final_drawing_001", "sys_base_color_001"]
    layers += ["sys_shade_001"]
    script: list[DrawingAction] = []
    for layer_id in layers:
        script += [
            DrawingAction("select_layer", {"layer_id": layer_id}),
            DrawingAction("select_color", {"color_id": color_id}),
            DrawingAction("select_brush", {"brush_id": "ink_gpen"}),
            DrawingAction("set_brush_params", {"size_level_id": 4, "opacity_level_id": 10}),
            stroke(*line),
            DrawingAction("advance_stage", {}),
        ]
    script += [
        DrawingAction("create_layer", {"layer_id": "fix1", "name": "fix", "role": "correction"}),
        DrawingAction("select_layer", {"layer_id": "fix1"}),
        DrawingAction("select_color", {"color_id": color_id}),
        DrawingAction("select_brush", {"brush_id": "ink_gpen"}),
        DrawingAction("set_brush_params", {"size_level_id": 4, "opacity_level_id": 10}),
        stroke(*line),
        DrawingAction("advance_stage", {}),
    ]
    return script


# {
#   責務: [ run_scripted: 台本Policyでtrialを実行する ]
# }
def run_scripted(
    actions: Sequence[DrawingAction],
    config: TrainConfig | None = None,
    prompt: str = "赤い四角",
    image_id: str = "img-1",
    seed: int = 1,
    stop_after_stage: str | None = None,
    raise_at: int | None = None,
) -> Trial:
    config = config or small_config()
    runner = RolloutRunner(
        ScriptedPolicy(actions, raise_at),
        HashedNgramTextEncoder(dim=8),
        ObservationBuilder(config.tail_length, config.summary_side),
        config,
    )
    return runner.run_trial(PolicyInput(image_id, prompt), seed, stop_after_stage)


# {
#   責務: [ make_dataset: 単色の長方形を持つ合成PNGを登録したDatasetを作る(画像だけを登録する) ]
#   戻り値: [ dataset_dir ]
# }
def make_dataset(root, count: int = 12):
    from PIL import Image

    from image_drawer.dataset import create_default_registry, register_images

    source = root / "source"
    source.mkdir()
    palette = [(220, 40, 40), (40, 90, 220), (40, 170, 70), (230, 190, 30)]
    for index in range(count):
        image = Image.new("RGB", (32, 32), (255, 255, 255))
        left = 2 + (index * 5) % 14
        color = palette[index % len(palette)]
        for x in range(left, left + 12):
            for y in range(8, 24):
                image.putpixel((x, y), color)
        image.save(source / f"img_{index:02d}.png")
    dataset_dir = root / "dataset"
    register_images(
        source, dataset_dir, create_default_registry(),
        tagger_id="color-layout-tags", captioner_id="color-layout-caption",
    )  # fmt: skip
    return dataset_dir
