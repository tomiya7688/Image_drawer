"""観測だけから合法と分かるActionのマスク(設計書4.2節)の契約テスト。"""

from __future__ import annotations

import dataclasses

import pytest

np = pytest.importorskip("numpy")

from image_drawer.dataset import PolicyInput  # noqa: E402
from image_drawer.drawing import DrawingAction, Engine  # noqa: E402
from image_drawer.drawing.stage import Budget, StageController, judge_style_intent  # noqa: E402
from image_drawer.training import HashedNgramTextEncoder  # noqa: E402
from image_drawer.training import action_codec as codec  # noqa: E402
from image_drawer.training.observation import ObservationBuilder  # noqa: E402
from image_drawer.training.policy import ParametricPolicy, RandomLegalPolicy  # noqa: E402
from image_drawer.training.rollout import RolloutRunner  # noqa: E402

from training_helpers import complete_script, small_config, stroke  # noqa: E402

KIND = {name: index for index, name in enumerate(codec.ACTION_KINDS)}
TOOLS = (
    DrawingAction("select_layer", {"layer_id": "sys_rough_001"}),
    DrawingAction("select_color", {"color_id": 0}),
    DrawingAction("select_brush", {"brush_id": "ink_gpen"}),
)


def start(side: int = 16):
    engine = Engine(side, side, seed=1)
    budget = Budget(60)
    controller = StageController(engine, judge_style_intent("test"), budget)
    builder = ObservationBuilder(tail_length=8, summary_side=4)
    text = np.linspace(-1, 1, 8, dtype=np.float32)

    def observe():
        return builder.build(text, controller, engine, budget)

    return controller, observe


def run(controller, actions) -> None:
    for action in actions:
        assert controller.propose(action).accepted, action


def allowed(observation) -> set[str]:
    mask = codec.kind_mask(observation)
    return {name for name, flag in zip(codec.ACTION_KINDS, mask) if flag}


def test_initial_observation_masks_tool_dependent_kinds() -> None:
    _controller, observe = start()
    assert allowed(observe()) == {"select_layer", "select_color", "select_brush"}


def test_set_brush_params_needs_selected_brush() -> None:
    controller, observe = start()
    run(controller, TOOLS[:2])
    assert "set_brush_params" not in allowed(observe())
    run(controller, TOOLS[2:])
    assert "set_brush_params" in allowed(observe())


def test_stroke_needs_layer_color_and_brush() -> None:
    controller, observe = start()
    for index, action in enumerate(TOOLS):
        assert "stroke_v2" not in allowed(observe()), index
        run(controller, [action])
    assert "stroke_v2" in allowed(observe())


def test_advance_needs_visible_stage_paint() -> None:
    controller, observe = start()
    run(controller, TOOLS)
    assert "advance_stage" not in allowed(observe())
    run(controller, [stroke(0.1, 0.5, 0.9, 0.5)])
    assert "advance_stage" in allowed(observe())


def test_mask_agrees_with_controller_through_complete_script() -> None:
    # 台本の各提案について、マスクが禁止した種別をControllerが受理しないこと。
    # (マスクは保守的でよいが、受理される提案を禁止する場合は台本を通せない)
    controller, observe = start()
    for action in complete_script():
        observation = observe()
        if action.type != "select_layer" or observation.stage_id != "refine":
            kinds = allowed(observation)
            if action.type in codec.ACTION_KINDS:
                assert action.type in kinds, (observation.stage_id, action.type)
        result = controller.propose(action)
        if action.type == "create_layer":
            continue
        assert result.accepted, (action.type, result.reason)


def test_refine_advance_needs_stroke_since_stage_entry() -> None:
    controller, observe = start()
    script = complete_script()
    refine_at = next(i for i, a in enumerate(script) if a.type == "create_layer")
    run(controller, script[:refine_at])
    assert observe().stage_id == "refine"
    # 前工程の受理済みstrokeがtailに残っていても、refineでは数えない。
    assert any(b.kind == "stroke_v2" and b.accepted for b in observe().action_tail)
    assert "advance_stage" not in allowed(observe())
    run(controller, script[refine_at:-1])
    assert "advance_stage" in allowed(observe())


def test_brush_mask_excludes_erasers() -> None:
    mask = codec.brush_mask()
    for index, brush_id in enumerate(codec.BRUSH_IDS):
        assert bool(mask[index]) == (not brush_id.startswith("eraser")), brush_id
    assert mask.any()


def test_mask_ignores_canvas_and_text() -> None:
    controller, observe = start()
    run(controller, TOOLS)
    observation = observe()
    other = dataclasses.replace(
        observation,
        text_condition=np.ones_like(observation.text_condition),
        canvas_summary=np.ones_like(observation.canvas_summary),
    )
    assert (codec.kind_mask(observation) == codec.kind_mask(other)).all()


def test_escape_kinds_always_remain() -> None:
    controller, observe = start()
    for action in complete_script():
        observation = observe()
        kinds = allowed(observation)
        assert {"select_color", "select_brush"} <= kinds, observation.stage_id
        if controller.closed:
            break
        controller.propose(action)


@pytest.mark.parametrize("make_policy", ["parametric", "random"])
def test_legal_mask_removes_tool_and_exit_rejections(make_policy: str) -> None:
    config = small_config(max_actions=60)
    encoder = HashedNgramTextEncoder(dim=8)
    rng = np.random.default_rng(0)
    policy = (
        ParametricPolicy(config, 8, rng)
        if make_policy == "parametric"
        else RandomLegalPolicy(config)
    )
    runner = RolloutRunner(
        policy, encoder, ObservationBuilder(config.tail_length, config.summary_side), config
    )
    reasons: dict[str, int] = {}
    attempts = 0
    for seed in range(8):
        trial = runner.run_trial(PolicyInput(f"img-{seed}", "赤い四角"), seed, None)
        for step in trial.steps:
            attempts += 1
            if not step.accepted:
                reasons[step.reject_reason or ""] = reasons.get(step.reject_reason or "", 0) + 1
    assert attempts > 0
    # 道具未選択のstroke、ブラシ未選択のset_brush_params、筆跡なしのadvanceは起きない。
    assert reasons == {}, reasons
