"""S1b-5: StageControllerの構築と初期状態。"""

from __future__ import annotations

import inspect

import pytest
from stage_helpers import engine_of, make_controller, model

from image_drawer.drawing import RENDERER_V1, Engine
from image_drawer.drawing.stage import Budget, StageConfigError, StageController, judge_style_intent


def test_controller_starts_in_rough_with_unselected_system_layer():
    controller = make_controller()
    state = engine_of(controller).state
    assert controller.stage_id == "rough" and not controller.closed and not controller.finished
    assert [(layer.id, layer.role, layer.visible) for layer in state.layers] == [
        ("sys_rough_001", "rough", True)
    ]
    assert state.active_layer_id is None and state.stage_id == "rough"
    assert controller.accepted_action_count == 0 and controller.attempt_count == 0
    assert controller.termination_reason is None


def test_initial_group_is_enter_create_clear_with_group_zero():
    events = make_controller().episode().events
    assert [e.type for e in events] == ["enter_stage", "create_layer", "clear_active_layer"]
    assert events[0].payload == {"stage_id": "rough"}
    assert events[1].payload == {"layer_id": "sys_rough_001", "name": "ラフ", "role": "rough"}
    assert [(e.seq, e.stage_id) for e in events] == [(0, "rough"), (1, "rough"), (2, "rough")]
    assert [(e.group.id, e.group.index, e.group.size) for e in events] == [
        (0, 0, 3), (0, 1, 3), (0, 2, 3),
    ]  # fmt: skip


def test_initial_events_are_system_actor_and_time_zero():
    events = make_controller().episode().events
    assert {e.actor for e in events} == {"system"}
    assert {e.time_ms for e in events} == {0}


def test_controller_rejects_raster_v1_engine():
    with pytest.raises(StageConfigError):
        StageController(Engine(8, 8, profile=RENDERER_V1), judge_style_intent("x"), Budget(10))


def test_controller_rejects_engine_with_history():
    engine = Engine(8, 8)
    engine.apply(model("create_layer", {"layer_id": "a", "name": "a", "role": "rough"}))
    with pytest.raises(StageConfigError):
        StageController(engine, judge_style_intent("x"), Budget(10))


def test_controller_rejects_canvas_too_large_for_five_layers():
    for side in (916, 1024):
        with pytest.raises(StageConfigError):
            StageController(Engine(side, side), judge_style_intent("x"), Budget(10))


def test_controller_accepts_canvas_at_five_layer_area_boundary():
    controller = StageController(Engine(915, 915), judge_style_intent("x"), Budget(10))
    assert controller.stage_id == "rough"


def test_controller_rejects_non_intent_argument():
    with pytest.raises(StageConfigError):
        StageController(Engine(8, 8), "線画のみ", Budget(10))
    with pytest.raises(StageConfigError):
        StageController(Engine(8, 8), judge_style_intent("x"), 10)
    with pytest.raises(StageConfigError):
        StageController("engine", judge_style_intent("x"), Budget(10))


def test_controller_constructor_has_no_target_image_parameter():
    names = list(inspect.signature(StageController.__init__).parameters)
    assert names == ["self", "engine", "intent", "budget"]
