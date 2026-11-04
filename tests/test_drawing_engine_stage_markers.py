"""S1b-1: Engineの工程マーカー（enter/exit/advance/clear）とapply_groupの受け入れテスト。"""

from __future__ import annotations

import pytest
from test_drawing_engine_v1 import test_main_golden_matches_every_checkpoint

from image_drawer.drawing import RENDERER_V1, DrawingAction, DrawingError, Engine
from image_drawer.drawing.engine import MAX_GROUP_ACTIONS
from image_drawer.drawing.models import MAX_EVENTS

SAMPLE = {
    "x_q8": 512, "y_q8": 512, "time_us": 0,
    "pressure_level_id": None, "tilt_x_level_id": None, "tilt_y_level_id": None,
}  # fmt: skip


def act(type_: str, payload: dict | None = None, actor: str = "system") -> DrawingAction:
    return DrawingAction(type_, payload or {}, actor)


def make_engine() -> Engine:
    return Engine(8, 8)


def painted_engine() -> Engine:
    """工程に入り、レイヤーに1画素以上を描いたEngine。"""
    engine = make_engine()
    for action in (
        act("enter_stage", {"stage_id": "rough"}),
        act("create_layer", {"layer_id": "w", "name": "w", "role": "rough"}),
        act("select_layer", {"layer_id": "w"}),
        act("select_brush", {"brush_id": "ink_gpen"}),
        act("select_color", {"color_id": 0}),
        act("stroke_v2", {"samples": [SAMPLE]}),
    ):
        engine.apply(action)
    return engine


def test_enter_stage_sets_stage_id():
    engine = make_engine()
    engine.apply(act("enter_stage", {"stage_id": "sketch"}))
    assert engine.state.stage_id == "sketch"


def test_enter_stage_rejects_unknown_stage_id():
    engine = make_engine()
    for bad in ("finish", "nope", 3, None):
        with pytest.raises(DrawingError):
            engine.apply(act("enter_stage", {"stage_id": bad}))
    assert engine.state.stage_id is None and engine.state.history == []


def test_exit_stage_requires_current_stage():
    engine = make_engine()
    engine.apply(act("enter_stage", {"stage_id": "rough"}))
    with pytest.raises(DrawingError):
        engine.apply(act("exit_stage", {"stage_id": "sketch"}))
    engine.apply(act("exit_stage", {"stage_id": "rough"}))
    assert engine.state.stage_id == "rough"


def test_exit_stage_without_entered_stage_is_rejected():
    with pytest.raises(DrawingError):
        make_engine().apply(act("exit_stage", {"stage_id": "rough"}))


def test_advance_stage_accepts_empty_and_review_payload():
    engine = make_engine()
    engine.apply(act("enter_stage", {"stage_id": "rough"}))
    engine.apply(act("advance_stage", {}, "model"))
    engine.apply(act("advance_stage", {"review_reason": "no_correction_needed"}, "model"))
    assert engine.state.stage_id == "rough"


@pytest.mark.parametrize(
    "payload",
    [
        {"review_reason": "nope"}, {"review_reason": None}, {"review_reason": 1},
        {"review_reason": "no_correction_needed", "to": "finish"}, {"to": "finish"},
        {"stage_id": "sketch"},
    ],
)  # fmt: skip
def test_advance_stage_rejects_unknown_review_reason_and_extra_keys(payload):
    engine = make_engine()
    engine.apply(act("enter_stage", {"stage_id": "rough"}))
    with pytest.raises(DrawingError):
        engine.apply(act("advance_stage", payload, "model"))
    assert len(engine.state.history) == 1


def test_advance_stage_requires_entered_stage():
    with pytest.raises(DrawingError):
        make_engine().apply(act("advance_stage", {}, "model"))


def test_clear_active_layer_unselects_and_keeps_pixels():
    engine = painted_engine()
    before = engine.layer_rows("w")
    engine.apply(act("clear_active_layer"))
    assert engine.state.active_layer_id is None
    assert engine.layer_rows("w") == before
    with pytest.raises(DrawingError):
        engine.apply(act("stroke_v2", {"samples": [SAMPLE]}, "model"))
    with pytest.raises(DrawingError):
        engine.apply(act("clear_active_layer", {"x": 1}))


def test_stage_markers_do_not_change_any_pixel():
    engine = painted_engine()
    before = engine.state.to_dict(include_pixels=True)["layers"]
    for action in (
        act("enter_stage", {"stage_id": "sketch"}),
        act("advance_stage", {}, "model"),
        act("exit_stage", {"stage_id": "sketch"}),
        act("clear_active_layer"),
    ):
        engine.apply(action)
    assert engine.state.to_dict(include_pixels=True)["layers"] == before


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("enter_stage", {"stage_id": "rough"}), ("exit_stage", {"stage_id": "rough"}),
        ("advance_stage", {}), ("clear_active_layer", {}),
    ],
)  # fmt: skip
def test_stage_markers_are_rejected_in_raster_v1_profile(kind, payload):
    engine = Engine(8, 8, profile=RENDERER_V1)
    with pytest.raises(DrawingError):
        engine.apply(act(kind, payload))
    assert engine.state.stage_id is None


def test_finish_sets_stage_id_finish_in_v2():
    engine = make_engine()
    engine.apply(act("enter_stage", {"stage_id": "refine"}))
    engine.apply(act("finish"))
    assert engine.state.stage_id == "finish" and engine.state.finished
    bare = make_engine()
    bare.apply(act("finish"))
    assert bare.state.stage_id is None


def test_apply_group_commits_all_events_with_consecutive_seq():
    engine = make_engine()
    engine.apply(act("enter_stage", {"stage_id": "rough"}))
    applied = engine.apply_group(
        [
            act("create_layer", {"layer_id": "a", "name": "a", "role": "rough"}),
            DrawingAction("exit_stage", {"stage_id": "rough"}, "system", None, 30, "rough"),
            act("enter_stage", {"stage_id": "sketch"}),
        ]
    )
    assert [a.seq for a in applied] == [1, 2, 3]
    assert [a.time_ms for a in applied] == [0, 30, 30]
    assert applied[1].stage_id == "rough"
    assert engine.state.history[1:] == applied
    assert engine.state.stage_id == "sketch"


def test_apply_group_rolls_back_layers_visibility_selection_and_stage_on_failure():
    engine = painted_engine()
    before = engine.state.to_dict(include_pixels=True)
    history_ids = [id(a) for a in engine.state.history]
    with pytest.raises(DrawingError):
        engine.apply_group(
            [
                act("set_layer_visibility", {"layer_id": "w", "visible": False}),
                act("enter_stage", {"stage_id": "sketch"}),
                act("create_layer", {"layer_id": "n", "name": "n", "role": "sketch"}),
                act("select_layer", {"layer_id": "n"}),
                act("select_color", {"color_id": 5}),
                act("finish"),
                act("clear_active_layer"),  # finish後の追加で失敗する
            ]
        )
    assert engine.state.to_dict(include_pixels=True) == before
    assert [id(a) for a in engine.state.history] == history_ids
    assert engine.state.layers[0].visible and not engine.state.finished


def test_apply_group_rejects_painting_actions():
    engine = painted_engine()
    before = engine.state.to_dict(include_pixels=True)
    for kind in ("stroke", "stroke_v2", "erase", "erase_v2"):
        with pytest.raises(DrawingError):
            engine.apply_group([act("clear_active_layer"), act(kind, {"samples": [SAMPLE]})])
    assert engine.state.to_dict(include_pixels=True) == before


def test_apply_group_rejects_empty_and_oversized_groups():
    engine = make_engine()
    with pytest.raises(DrawingError):
        engine.apply_group([])
    marker = [act("clear_active_layer")] * (MAX_GROUP_ACTIONS + 1)
    with pytest.raises(DrawingError):
        engine.apply_group(marker)
    assert engine.state.history == []
    assert len(engine.apply_group(marker[:MAX_GROUP_ACTIONS])) == MAX_GROUP_ACTIONS


def test_apply_group_respects_history_limit_boundary():
    engine = make_engine()
    marker = act("clear_active_layer")
    while len(engine.state.history) < MAX_EVENTS - MAX_GROUP_ACTIONS - 3:
        engine.apply_group([marker] * MAX_GROUP_ACTIONS)
    room = MAX_EVENTS - len(engine.state.history)
    before = len(engine.state.history)
    with pytest.raises(DrawingError):
        engine.apply_group([marker] * (room + 1))
    assert len(engine.state.history) == before
    engine.apply_group([marker] * room)
    assert len(engine.state.history) == MAX_EVENTS


def test_v1_golden_still_matches_after_engine_extension():
    test_main_golden_matches_every_checkpoint()
