"""S1b-5: 提案の拒否・原子性・記録の独立性（遷移後の選択リセットを含む）。"""

from __future__ import annotations

import pytest
from stage_helpers import (
    accept,
    accept_all,
    advance,
    draw_actions,
    engine_of,
    make_controller,
    model,
    stroke_v2,
)

from image_drawer.drawing import DrawingAction, DrawingError
from image_drawer.drawing.stage import StageClosedError, StageInvariantError


def code(result) -> str:
    return result.reason.split(":")[0]


def test_stroke_without_reselecting_layer_after_transition_is_rejected():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    accept(controller, advance())
    result = controller.propose(stroke_v2())
    assert not result.accepted and code(result) == "engine_rejected"


def test_injected_apply_group_failure_rolls_back_and_closes_controller(monkeypatch):
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    engine = engine_of(controller)
    before = engine.state.to_dict(include_pixels=True)
    events_before = controller.episode().events
    original = engine.apply
    calls = []

    def flaky(action):
        calls.append(action.type)
        if len(calls) == 3:
            raise DrawingError("injected failure")
        return original(action)

    monkeypatch.setattr(engine, "apply", flaky)
    with pytest.raises(StageInvariantError):
        controller.propose(advance())
    monkeypatch.undo()
    assert controller.closed and controller.termination_reason == "internal_error"
    assert engine.state.to_dict(include_pixels=True) == before
    assert controller.episode().events == events_before
    assert controller.attempts()[-1].reason.startswith("internal_error")
    with pytest.raises(StageClosedError):
        controller.propose(stroke_v2())


def test_rejected_proposal_does_not_consume_seq_or_group_id():
    controller = make_controller()
    accept(controller, model("select_layer", {"layer_id": "sys_rough_001"}))
    assert not controller.propose(advance()).accepted
    accept_all(controller, draw_actions("rough")[1:])
    events = controller.episode().events
    assert [e.seq for e in events] == list(range(len(events)))
    assert [e.group.id for e in events if e.actor == "model"] == [1, 2, 3, 4]


def test_rejected_proposal_leaves_engine_state_identical():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    before = engine_of(controller).state.to_dict(include_pixels=True)
    events_before = controller.episode().events
    rejected = [
        advance("no_correction_needed"),
        model("skip_stage"),
        model("select_layer", {"layer_id": "ghost"}),
        model("create_layer", {"layer_id": "sys_x", "name": "x", "role": "rough"}),
        model("stroke_v2", {"samples": []}),
    ]
    for bad in rejected:
        assert not controller.propose(bad).accepted
    assert engine_of(controller).state.to_dict(include_pixels=True) == before
    assert controller.episode().events == events_before
    assert all(a.seqs == () for a in controller.attempts() if not a.accepted)


def test_result_events_are_the_recorded_events():
    controller = make_controller()
    results = accept_all(controller, draw_actions("rough"))
    recorded = controller.episode().events
    flat = [event for result in results for event in result.events]
    assert recorded[3:] == flat
    assert [e.seq for e in flat] == list(range(3, 3 + len(flat)))


def test_proposal_payload_is_copied_on_accept():
    controller = make_controller()
    payload = {"layer_id": "sys_rough_001"}
    accept(controller, model("select_layer", payload))
    payload["layer_id"] = "tampered"
    assert controller.episode().events[3].payload == {"layer_id": "sys_rough_001"}
    assert controller.attempts()[0].action.payload == {"layer_id": "sys_rough_001"}


@pytest.mark.parametrize(
    "bad",
    [
        DrawingAction("select_layer", {"layer_id": "a"}, "model", 5),
        DrawingAction("select_layer", {"layer_id": "a"}, "model", None, None, "rough"),
    ],
)  # fmt: skip
def test_proposal_with_seq_group_or_stage_id_is_invalid_envelope(bad):
    controller = make_controller()
    result = controller.propose(bad)
    assert not result.accepted and code(result) == "invalid_envelope"


def test_proposal_with_group_is_invalid_envelope():
    from image_drawer.drawing import EventGroup

    bad = DrawingAction(
        "select_layer", {"layer_id": "a"}, "model", None, None, None, EventGroup(0, 0, 1)
    )
    assert code(make_controller().propose(bad)) == "invalid_envelope"


@pytest.mark.parametrize("actor", ["human", "system", "synthetic", "derived", "robot"])
def test_proposal_with_non_model_actor_is_invalid_envelope(actor):
    bad = DrawingAction("select_layer", {"layer_id": "sys_rough_001"}, actor)
    assert code(make_controller().propose(bad)) == "invalid_envelope"


@pytest.mark.parametrize(
    "bad",
    ["select_layer", None, {"type": "x"}, DrawingAction("select_layer", ["a"], "model"),
     DrawingAction(5, {}, "model")],
)  # fmt: skip
def test_non_action_object_and_non_dict_payload_are_invalid_envelope(bad):
    controller = make_controller()
    result = controller.propose(bad)
    assert not result.accepted and code(result) == "invalid_envelope"
    assert controller.attempt_count == 1 and controller.attempts()[0].seqs == ()
