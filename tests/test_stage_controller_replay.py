"""S1c-5: StageControllerのEpisodeを製品Replayerで再生し、Engine状態と一致させる結合テスト。"""

from __future__ import annotations

import json

from episode_helpers import (
    REVIEW_PROMPT,
    draft_controller,
    full_controller,
    groups_of,
    raw_of,
    signature,
    review_controller,
)
from stage_helpers import (
    STAGE_ORDER,
    accept,
    accept_all,
    advance,
    draw_actions,
    engine_of,
    make_controller,
)


from image_drawer.drawing.episode_io import load_episode, loads_episode, save_episode
from image_drawer.drawing.replay import replay

WIDTH, HEIGHT = 12, 8


def assert_same_state(controller) -> None:
    result = replay(controller.episode())
    assert signature(result.state) == signature(engine_of(controller).state)
    assert result.image.rgba == engine_of(controller).render().rgba


def test_controller_episode_replays_to_identical_state_for_full_run():
    assert_same_state(full_controller())


def test_controller_episode_replays_to_identical_state_for_review_run():
    controller = review_controller()
    assert controller.finished
    assert_same_state(controller)


def test_controller_draft_replays_to_identical_state_at_every_attempt_limit():
    for limit in (5, 8, 13):
        controller = make_controller("猫", WIDTH, HEIGHT, max_actions=limit, max_attempts=limit)
        accept_all(controller, draw_actions("rough")[:2])
        while not controller.closed:
            assert not controller.propose(advance()).accepted  # 筆跡なしの退出は拒否される
        assert controller.termination_reason == "attempt_budget_exhausted"
        assert controller.attempt_count == limit
        assert_same_state(controller)
    assert_same_state(draft_controller())


def test_controller_rejected_attempts_are_absent_from_replayed_episode():
    controller = make_controller("猫", WIDTH, HEIGHT)
    accept_all(controller, draw_actions("rough")[:2])
    events_before = len(controller.episode().events)
    assert not controller.propose(advance()).accepted
    episode = controller.episode()
    assert len(episode.events) == events_before
    assert controller.attempt_count == 3
    assert len(replay(episode).state.history) == events_before


def test_save_and_load_controller_episode_replays_identically(tmp_path):
    controller = full_controller()
    save_episode(controller.episode(), tmp_path / "e.json")
    loaded = load_episode(tmp_path / "e.json")
    assert signature(replay(loaded).state) == signature(engine_of(controller).state)


def drive_with_checkpoints(prompt: str = "猫"):
    """全工程を1提案ずつ進め、受理ごとに（最後のseq、Engine状態）を記録する。"""
    controller = make_controller(prompt, WIDTH, HEIGHT)
    checkpoints = [(2, signature(engine_of(controller).state))]
    for stage_id in STAGE_ORDER:
        for action in [*draw_actions(stage_id), advance()]:
            result = accept(controller, action)
            checkpoints.append((result.events[-1].seq, signature(engine_of(controller).state)))
    return controller, checkpoints


def test_every_committed_group_boundary_replays_to_controller_checkpoint():
    controller, checkpoints = drive_with_checkpoints()
    episode = controller.episode()
    assert checkpoints[-1][0] == len(episode.events) - 1
    for seq, expected in checkpoints:
        assert signature(replay(episode, until_seq=seq).state) == expected


def test_replay_after_recover_equals_controller_state_at_that_group():
    controller, checkpoints = drive_with_checkpoints()
    raw = raw_of(controller.episode())
    groups = groups_of(raw)
    boundary, expected = next(
        (seq, sig)
        for seq, sig in checkpoints
        if any(g[0]["type"] == "advance_stage" and g[0]["seq"] == seq + 1 for g in groups)
    )
    raw["events"] = raw["events"][: boundary + 1 + 2]  # 次の遷移グループの途中で切る
    recovered = loads_episode(json.dumps(raw), recover=True)
    assert recovered.events[-1].seq == boundary
    assert signature(replay(recovered).state) == expected


def test_review_prompt_constant_matches_review_controller_intent():
    assert review_controller().episode().style_intent["monochrome_line_art_only"] is True
    assert REVIEW_PROMPT
