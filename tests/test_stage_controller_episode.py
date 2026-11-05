"""S1b-6: 受理済みイベントからのDrawingEpisode生成。"""

from __future__ import annotations

from stage_helpers import (
    accept_all,
    draw_actions,
    full_run,
    make_controller,
    model,
    pixels_hex,
)

from image_drawer.drawing import DrawingAction, DrawingEpisode, Engine
from image_drawer.drawing.stage import default_stage_table
from image_drawer.drawing.stage.controller import StageController
from stage_helpers import engine_of


def finished_controller(prompt: str = "猫") -> StageController:
    controller = make_controller(prompt)
    full_run(controller)
    return controller


def test_episode_contains_only_accepted_events():
    controller = make_controller()
    controller.propose(model("skip_stage"))
    accept_all(controller, draw_actions("rough"))
    controller.propose(model("finish"))
    events = controller.episode().events
    accepted_seqs = {seq for a in controller.attempts() if a.accepted for seq in a.seqs}
    assert {e.seq for e in events} == accepted_seqs | {0, 1, 2}
    assert [e.type for e in events] == [
        "enter_stage", "create_layer", "clear_active_layer",
        "select_layer", "select_brush", "select_color", "stroke_v2",
    ]  # fmt: skip


def test_episode_origin_is_mixed_and_seq_is_contiguous():
    episode = finished_controller().episode()
    assert episode.origin == "mixed"
    assert [e.seq for e in episode.events] == list(range(len(episode.events)))
    assert {e.actor for e in episode.events} == {"model", "system"}
    assert episode.renderer_version == "brush-raster-v2" and (episode.width, episode.height) == (
        16,
        16,
    )


def test_episode_of_finished_run_has_status_complete_and_finished_reason():
    controller = finished_controller()
    episode = controller.episode()
    assert (episode.status, episode.termination_reason) == ("complete", "finished")
    assert episode.events[-1].type == "finish"
    # 既存のEpisode検証（seq連続・origin整合・終端finish）を通る。
    reloaded = DrawingEpisode.from_dict(episode.to_dict())
    assert len(reloaded.events) == len(episode.events)


def test_episode_of_open_run_is_draft_in_progress():
    controller = make_controller()
    episode = controller.episode()
    assert (episode.status, episode.termination_reason) == ("draft", "in_progress")
    assert len(episode.events) == 3


def test_episode_of_exhausted_run_records_exhaustion_reason():
    by_action = make_controller(max_actions=2)
    accept_all(by_action, draw_actions("rough")[:2])
    assert by_action.episode().termination_reason == "action_budget_exhausted"
    by_attempt = make_controller(max_actions=2, max_attempts=2)
    for _ in range(2):
        by_attempt.propose(model("skip_stage"))
    episode = by_attempt.episode()
    assert (episode.status, episode.termination_reason) == ("draft", "attempt_budget_exhausted")


def test_episode_records_style_intent_budget_and_workflow_ids():
    controller = make_controller("白黒の線画、影なし", max_actions=50)
    episode = controller.episode()
    table = default_stage_table()
    assert episode.prompt == "白黒の線画、影なし"
    assert episode.style_intent == {
        "version": "style-intent-rules-v1",
        "monochrome_line_art_only": True,
        "flat_style_no_shading": True,
    }
    assert episode.budget == {"max_actions": 50, "max_attempts": 250}
    assert episode.workflow_id == "illustration-standard-v2"
    assert episode.workflow_sha256 == table.workflow_sha256


def test_episode_default_id_is_stable_for_same_inputs():
    first = make_controller("猫", seed=3).episode().episode_id
    again = make_controller("猫", seed=3).episode().episode_id
    assert first == again and first.startswith("ep-") and len(first) == 3 + 16
    assert make_controller("犬", seed=3).episode().episode_id != first
    assert make_controller("猫", seed=4).episode().episode_id != first
    assert make_controller("猫", seed=3).episode("ep-custom").episode_id == "ep-custom"


def test_episode_events_are_independent_copies():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    first = controller.episode()
    first.events[3].payload["layer_id"] = "tampered"
    first.events.pop()
    second = controller.episode()
    assert second.events[3].payload == {"layer_id": "sys_rough_001"}
    assert len(second.events) == 7


def test_episode_events_replay_to_same_pixels_on_fresh_engine():
    controller = finished_controller()
    fresh = Engine(16, 16)
    for event in controller.episode().events:
        fresh.apply(DrawingAction(event.type, event.payload, event.actor, event.seq, event.time_ms))
    assert pixels_hex(fresh) == pixels_hex(engine_of(controller))
    assert fresh.render().rgba == engine_of(controller).render().rgba
    assert fresh.state.stage_id == "finish" and fresh.state.finished
