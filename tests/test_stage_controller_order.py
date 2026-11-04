"""S1b-5: 工程順序・原子性・拒否時の不変性。"""

from __future__ import annotations

import pytest
from stage_helpers import (
    STAGE_ORDER,
    accept,
    accept_all,
    advance,
    advance_to,
    draw_actions,
    engine_of,
    full_erase_actions,
    full_run,
    make_controller,
    model,
    noop_stroke_actions,
    setup_actions,
    stroke_v2,
)

from image_drawer.drawing import DrawingAction, DrawingError, Engine
from image_drawer.drawing.stage import StageClosedError, StageInvariantError

NEXT_STAGE = dict(zip(STAGE_ORDER, (*STAGE_ORDER[1:], "finish")))
EXPECTED_TYPES = {
    "rough": ["advance_stage", "exit_stage", "enter_stage", "create_layer", "clear_active_layer"],
    "sketch": [
        "advance_stage", "exit_stage", "set_layer_visibility", "enter_stage", "create_layer",
        "clear_active_layer",
    ],
    "final_drawing": [
        "advance_stage", "exit_stage", "set_layer_visibility", "enter_stage", "create_layer",
        "clear_active_layer",
    ],
    "base_color": ["advance_stage", "exit_stage", "enter_stage", "create_layer", "clear_active_layer"],
    "shading": ["advance_stage", "exit_stage", "enter_stage", "clear_active_layer"],
    "refine": ["advance_stage", "exit_stage", "finish"],
}  # fmt: skip


def code(result) -> str:
    return result.reason.split(":")[0]


def test_full_run_passes_seven_stages_in_fixed_order():
    controller = make_controller()
    full_run(controller)
    events = controller.episode().events
    entered = [e.payload["stage_id"] for e in events if e.type == "enter_stage"]
    exited = [e.payload["stage_id"] for e in events if e.type == "exit_stage"]
    assert entered == list(STAGE_ORDER)
    assert exited == list(STAGE_ORDER)
    assert controller.finished and controller.closed and controller.stage_id == "finish"


def test_full_run_stage_id_property_follows_each_transition():
    controller = make_controller()
    for stage_id in STAGE_ORDER:
        assert controller.stage_id == stage_id
        accept_all(controller, draw_actions(stage_id))
        accept(controller, advance())
    assert controller.stage_id == "finish"


@pytest.mark.parametrize("stage_id", ["rough", "sketch", "final_drawing"])
def test_advance_without_stroke_is_rejected_in_rough_sketch_final_drawing(stage_id):
    controller = make_controller()
    advance_to(controller, stage_id)
    result = controller.propose(advance())
    assert not result.accepted and code(result) == "stage_not_ready"
    assert controller.stage_id == stage_id


def test_advance_after_color_selection_only_is_rejected():
    controller = make_controller()
    accept_all(controller, setup_actions("sys_rough_001"))
    assert code(controller.propose(advance())) == "stage_not_ready"


def test_advance_after_noop_stroke_is_rejected_when_stroke_required():
    controller = make_controller()
    accept_all(controller, [model("select_layer", {"layer_id": "sys_rough_001"})])
    accept_all(controller, noop_stroke_actions())
    assert controller.accepted_action_count == 5
    assert code(controller.propose(advance())) == "stage_not_ready"


def test_advance_after_erasing_all_pixels_is_rejected_in_rough():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    accept_all(controller, full_erase_actions("sys_rough_001"))
    assert not any(engine_of(controller).state.layers[0].pixels.data[3::4])
    assert code(controller.propose(advance())) == "stage_not_ready"


def test_erase_only_does_not_satisfy_base_color():
    controller = make_controller()
    advance_to(controller, "base_color")
    accept_all(controller, full_erase_actions("sys_base_color_001"))
    assert code(controller.propose(advance())) == "stage_not_ready"


def test_effective_stroke_count_resets_on_stage_entry():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    accept(controller, advance())
    assert code(controller.propose(advance())) == "stage_not_ready"


@pytest.mark.parametrize("payload", [{"to": "finish"}, {"stage_id": "base_color"}, {"next": "x"}])
def test_advance_payload_with_destination_is_rejected(payload):
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    result = controller.propose(model("advance_stage", payload))
    assert not result.accepted and code(result) == "invalid_payload"
    assert controller.stage_id == "rough"


@pytest.mark.parametrize("kind", ["skip_stage", "set_phase"])
def test_skip_stage_and_set_phase_are_action_not_allowed(kind):
    controller = make_controller()
    payload = {"phase": "sketch"} if kind == "set_phase" else {"stage_id": "base_color"}
    assert code(controller.propose(model(kind, payload))) == "action_not_allowed"


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("finish", {}), ("enter_stage", {"stage_id": "sketch"}),
        ("exit_stage", {"stage_id": "rough"}),
        ("set_layer_visibility", {"layer_id": "sys_rough_001", "visible": False}),
        ("clear_active_layer", {}),
    ],
)  # fmt: skip
def test_model_cannot_propose_controller_only_actions(kind, payload):
    controller = make_controller()
    before = engine_of(controller).state.to_dict(include_pixels=True)
    assert code(controller.propose(model(kind, payload))) == "action_not_allowed"
    assert engine_of(controller).state.to_dict(include_pixels=True) == before


def test_no_backward_transition_is_possible():
    controller = make_controller()
    advance_to(controller, "base_color")
    before = engine_of(controller).state.to_dict(include_pixels=True)
    result = controller.propose(model("enter_stage", {"stage_id": "rough"}))
    assert code(result) == "action_not_allowed"
    assert engine_of(controller).state.to_dict(include_pixels=True) == before
    assert controller.stage_id == "base_color"


def test_finish_event_is_emitted_exactly_once_and_last():
    controller = make_controller()
    full_run(controller)
    types = [e.type for e in controller.episode().events]
    assert types.count("finish") == 1 and types[-1] == "finish"
    assert controller.termination_reason == "finished"


def test_propose_after_finish_raises_stage_closed_error():
    controller = make_controller()
    full_run(controller)
    count = controller.attempt_count
    with pytest.raises(StageClosedError):
        controller.propose(stroke_v2())
    assert controller.attempt_count == count


def test_transition_events_form_one_group_with_consecutive_seq():
    controller = make_controller()
    for result in full_run(controller):
        seqs = [e.seq for e in result.events]
        assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
        assert len({e.group.id for e in result.events}) == 1
        assert result.events[0].actor == "model"
        assert {e.actor for e in result.events[1:]} == {"system"}


@pytest.mark.parametrize("stage_id", STAGE_ORDER)
def test_transition_group_has_exact_event_order_for_each_stage(stage_id):
    controller = make_controller()
    advance_to(controller, stage_id)
    accept_all(controller, draw_actions(stage_id))
    result = accept(controller, advance())
    assert [e.type for e in result.events] == EXPECTED_TYPES[stage_id]
    assert result.events[0].stage_id == stage_id
    assert result.events[-1].stage_id == NEXT_STAGE[stage_id]
    if stage_id == "sketch":
        assert result.events[2].payload == {"layer_id": "sys_rough_001", "visible": False}
    if stage_id == "final_drawing":
        assert result.events[2].payload == {"layer_id": "sys_sketch_001", "visible": False}


def test_group_index_and_size_match_for_every_group():
    controller = make_controller()
    full_run(controller)
    groups: dict[int, list] = {}
    for event in controller.episode().events:
        groups.setdefault(event.group.id, []).append(event)
    for events in groups.values():
        assert [e.group.index for e in events] == list(range(len(events)))
        assert {e.group.size for e in events} == {len(events)}


def test_group_ids_start_at_zero_and_increase_by_one():
    controller = make_controller()
    full_run(controller)
    ids = [e.group.id for e in controller.episode().events]
    assert ids[0] == 0 and sorted(set(ids)) == list(range(max(ids) + 1))
    assert ids == sorted(ids)


def test_group_events_share_time_ms():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    result = accept(controller, model("advance_stage", {}, time_ms=900))
    assert {e.time_ms for e in result.events} == {900}
    follow = accept(controller, model("select_layer", {"layer_id": "sys_sketch_001"}))
    assert follow.events[0].time_ms == 900


def test_event_stage_id_equals_engine_stage_id_after_apply():
    controller = make_controller()
    full_run(controller)
    replay = Engine(16, 16)
    for event in controller.episode().events:
        replay.apply(DrawingAction(event.type, event.payload, event.actor))
        assert replay.state.stage_id == event.stage_id


def test_active_layer_is_cleared_on_every_transition():
    # refine→finishは完了グループ（clear_active_layerなし）なので、工程入場を伴う5遷移を見る。
    controller = make_controller()
    for stage_id in STAGE_ORDER[:-1]:
        accept_all(controller, draw_actions(stage_id))
        assert engine_of(controller).state.active_layer_id is not None
        accept(controller, advance())
        assert engine_of(controller).state.active_layer_id is None


def test_brush_color_size_and_opacity_are_preserved_across_transition():
    controller = make_controller()
    accept_all(controller, setup_actions("sys_rough_001", "paint_round_opaque", 7))
    accept(controller, model("set_brush_params", {"size_level_id": 3, "opacity_level_id": 6}))
    accept(controller, stroke_v2())
    state = engine_of(controller).state
    expected = (7, "paint_round_opaque", 3, 6)
    accept(controller, advance())
    got = (
        state.selected_color_id,
        state.selected_brush_id,
        state.size_level_id,
        state.opacity_level_id,
    )
    assert got == expected
