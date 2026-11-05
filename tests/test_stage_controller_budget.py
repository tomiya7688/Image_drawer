"""S1b-6: Action予算・attempt予算・閉じる判定。"""

from __future__ import annotations

import pytest
from stage_helpers import (
    accept,
    accept_all,
    advance,
    draw_actions,
    full_run,
    make_controller,
    model,
)

from image_drawer.drawing.stage import StageClosedError

BAD = model("skip_stage")


def run_length() -> int:
    """全工程を完走するのに必要な受理済みmodel Action数。"""
    controller = make_controller()
    full_run(controller)
    return controller.accepted_action_count


def test_attempt_numbers_count_accepted_and_rejected_proposals():
    controller = make_controller()
    results = [
        controller.propose(model("select_layer", {"layer_id": "sys_rough_001"})),
        controller.propose(BAD),
        controller.propose(advance()),
        controller.propose(model("select_color", {"color_id": 0})),
    ]
    assert [r.attempt for r in results] == [1, 2, 3, 4]
    assert [r.accepted for r in results] == [True, False, False, True]
    assert controller.attempt_count == 4


def test_attempts_log_keeps_reason_stage_and_seqs():
    controller = make_controller()
    accept(controller, model("select_layer", {"layer_id": "sys_rough_001"}))
    controller.propose(BAD)
    accept_all(controller, draw_actions("rough")[1:])
    accept(controller, advance())
    log = controller.attempts()
    assert [a.attempt for a in log] == list(range(1, len(log) + 1))
    assert log[0].accepted and log[0].seqs == (3,) and log[0].reason is None
    assert not log[1].accepted and log[1].reason.startswith("action_not_allowed")
    assert log[1].seqs == () and log[1].stage_id == "rough" and log[1].action.type == "skip_stage"
    last = log[-1]
    assert last.stage_id == "rough" and len(last.seqs) == 5  # advance + 遷移のsystem 4件
    assert last.action.type == "advance_stage" and last.action.seq is None


def test_rejected_attempts_do_not_count_as_accepted_actions():
    controller = make_controller()
    for _ in range(5):
        controller.propose(BAD)
    assert controller.accepted_action_count == 0 and controller.attempt_count == 5


def test_system_events_do_not_consume_action_budget():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    result = accept(controller, advance())
    assert len(result.events) == 5
    assert controller.accepted_action_count == 5


def test_action_budget_exhaustion_closes_controller_without_auto_finish():
    controller = make_controller(max_actions=3)
    accept_all(controller, draw_actions("rough")[:3])
    assert controller.closed and controller.termination_reason == "action_budget_exhausted"
    assert not controller.finished
    episode = controller.episode()
    assert episode.status == "draft" and episode.termination_reason == "action_budget_exhausted"
    assert "finish" not in [e.type for e in episode.events]


def test_action_budget_exhaustion_does_not_advance_stage():
    controller = make_controller(max_actions=4)
    accept_all(controller, draw_actions("rough"))
    assert controller.closed and controller.stage_id == "rough"
    assert controller.episode().events[-1].type == "stroke_v2"


def test_attempt_limit_closes_controller_with_draft_episode():
    controller = make_controller(max_actions=5, max_attempts=6)
    for _ in range(6):
        controller.propose(BAD)
    assert controller.closed and controller.termination_reason == "attempt_budget_exhausted"
    episode = controller.episode()
    assert episode.status == "draft" and episode.termination_reason == "attempt_budget_exhausted"


def test_attempt_limit_exact_boundary_last_attempt_is_still_judged():
    controller = make_controller(max_actions=3, max_attempts=3)
    controller.propose(BAD)
    controller.propose(BAD)
    assert not controller.closed
    last = controller.propose(model("select_layer", {"layer_id": "sys_rough_001"}))
    assert last.accepted and last.attempt == 3
    assert controller.closed and controller.termination_reason == "attempt_budget_exhausted"


def test_final_advance_at_last_allowed_action_is_complete_not_draft():
    controller = make_controller(max_actions=run_length())
    full_run(controller)
    episode = controller.episode()
    assert controller.termination_reason == "finished"
    assert episode.status == "complete" and episode.termination_reason == "finished"


def test_final_advance_at_last_allowed_attempt_is_complete_not_draft():
    count = run_length()
    controller = make_controller(max_actions=count, max_attempts=count)
    full_run(controller)
    assert controller.attempt_count == count
    assert controller.episode().status == "complete"


def test_propose_after_budget_exhaustion_raises_stage_closed_error():
    controller = make_controller(max_actions=1)
    accept(controller, model("select_layer", {"layer_id": "sys_rough_001"}))
    with pytest.raises(StageClosedError):
        controller.propose(BAD)
    assert controller.attempt_count == 1


def test_infinite_invalid_proposals_terminate_at_attempt_limit():
    controller = make_controller(max_actions=10)  # 既定の試行上限は100
    steps = 0
    while not controller.closed:
        assert not controller.propose(BAD).accepted
        steps += 1
        assert steps <= 100
    assert steps == 100 and controller.termination_reason == "attempt_budget_exhausted"


def test_default_attempt_limit_for_max_actions_1000_is_5000():
    controller = make_controller(max_actions=1000)
    assert controller.episode().budget == {"max_actions": 1000, "max_attempts": 5000}


def test_termination_reason_priority_finished_over_action_over_attempt():
    count = run_length()
    both = make_controller(max_actions=count, max_attempts=count)
    full_run(both)
    assert both.termination_reason == "finished"
    action_first = make_controller(max_actions=2, max_attempts=2)
    accept_all(action_first, draw_actions("rough")[:2])
    assert action_first.termination_reason == "action_budget_exhausted"
