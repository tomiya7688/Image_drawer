"""S4-5〜7: GUIのviewmodel・controller。画面の操作や見た目はここでは確認しない（手動確認）。"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from inference_helpers import SIDE, scripted_registry

from image_drawer.drawing.models import LAYER_ROLES_Z_ORDER, STAGE_IDS
from image_drawer.drawing.replay import replay
from image_drawer.gui.replay_cursor import ReplayCursor
from image_drawer.gui.viewmodel import (
    BrushPanel,
    ViewerController,
    ViewerEmpty,
    ViewerLoaded,
    ViewerLoadError,
    policy_badge,
)
from image_drawer.inference import DrawRequest
from image_drawer.inference.artifacts import write_outcome
from image_drawer.inference.session import generate_episode

FIXTURES = Path(__file__).parent / "fixtures" / "drawing_episode_v2"


def drawn(**kwargs):
    request = DrawRequest("猫", "scripted", width=SIDE, height=SIDE, **kwargs)
    return generate_episode(request, scripted_registry())


def layer_bytes(state) -> list[tuple[str, bytes]]:
    return [(layer.id, bytes(layer.pixels.data)) for layer in state.layers]


def test_replay_cursor_end_state_equals_replayer_result() -> None:
    episode = drawn().episode
    cursor = ReplayCursor(episode)
    cursor.seek_index(cursor.last_index)
    expected = replay(episode)
    assert cursor.image().rgba == expected.image.rgba
    assert layer_bytes(cursor.state) == [(layer.layer_id, layer.rgba) for layer in expected.layers]
    assert cursor.seq == episode.events[-1].seq


def test_replay_cursor_seek_backward_matches_forward_state() -> None:
    episode = drawn().episode
    forward = ReplayCursor(episode)
    backward = ReplayCursor(episode)
    backward.seek_index(backward.last_index)
    for target in range(0, forward.last_index + 1, 5):
        forward.seek_index(target)
        backward.seek_index(target)
        assert backward.image().rgba == forward.image().rgba
        assert layer_bytes(backward.state) == layer_bytes(forward.state)
        assert backward.seq == forward.seq
    backward.seek_index(-1)
    assert backward.state.layers == [] and backward.seq == -1


def test_replay_cursor_stage_jump_lands_on_stage_boundaries() -> None:
    cursor = ReplayCursor(drawn().episode)
    positions = []
    for stage_id in reversed(STAGE_IDS):
        assert cursor.jump_to_stage(stage_id)
        assert cursor.state.stage_id == stage_id
        positions.append(cursor.index)
    assert positions == sorted(positions, reverse=True)
    assert not cursor.jump_to_stage("no-such-stage")
    cursor.seek(10**6)
    assert cursor.index == cursor.last_index


def test_replay_cursor_last_stroke_and_frame_cache_are_bounded() -> None:
    cursor = ReplayCursor(drawn().episode, cache_size=3)
    assert cursor.last_stroke() is None
    for index in range(0, cursor.last_index + 1):
        cursor.seek_index(index)
        cursor.image()
    assert len(cursor._frames) <= 3 and cursor.last_stroke() is not None


def test_layer_panel_order_follows_role_z_order() -> None:
    controller = ViewerController()
    state = controller.load(FIXTURES / "full_seven_stage.episode.json")
    assert isinstance(state, ViewerLoaded)
    rows = state.model.layer_panel()
    order = [LAYER_ROLES_Z_ORDER.index(row.role) for row in rows]
    assert order == sorted(order, reverse=True) and len(rows) >= 5
    first = rows[0].layer_id
    assert state.model.canvas().rgba == state.model.cursor.image().rgba
    state.model.toggle_view_layer(first)
    assert not next(r for r in state.model.layer_panel() if r.layer_id == first).visible
    assert state.model.canvas().rgba != state.model.cursor.image().rgba or not rows[0].has_pixels
    assert state.model.episode.events[-1].type == "finish"


def test_stage_bar_shows_review_reason_for_reviewed_stages() -> None:
    state = ViewerController().load(FIXTURES / "review_only_stages.episode.json")
    items = state.model.stage_bar()
    reasons = {item.stage_id: item.review_reason for item in items if item.review_reason}
    assert set(reasons.values()) >= {"monochrome_line_art_only", "flat_style_no_shading"}
    assert all(item.status == "reviewed" for item in items if item.review_reason)
    state.model.cursor.seek_index(-1)
    assert {item.status for item in state.model.stage_bar()} == {"unreached"}


def test_draft_is_shown_with_draft_badge_and_reason(tmp_path: Path) -> None:
    fixture = ViewerController().load(FIXTURES / "draft_budget_exhausted.episode.json")
    assert fixture.model.draft_badge.startswith("未完成")
    assert "interrupted" in {item.status for item in fixture.model.stage_bar()}
    written = write_outcome(drawn(max_actions=5), tmp_path / "d")
    state = ViewerController().load(written.draft_path)
    assert state.model.draft_badge == "未完成（action_budget_exhausted）"
    assert [r.reason for r in state.model.rejected] == []
    complete = ViewerController().load(write_outcome(drawn(), tmp_path / "c").episode_path)
    assert complete.model.draft_badge is None


def test_viewer_controller_load_error_and_empty_states(tmp_path: Path) -> None:
    controller = ViewerController()
    assert isinstance(controller.state, ViewerEmpty)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert isinstance(controller.load(bad), ViewerLoadError)
    assert isinstance(controller.load(tmp_path / "missing.json"), ViewerLoadError)
    assert isinstance(controller.load(FIXTURES / "full_seven_stage.episode.json"), ViewerLoaded)
    controller.clear()
    assert isinstance(controller.state, ViewerEmpty)


def test_brush_panel_is_read_only() -> None:
    state = ViewerController().load(FIXTURES / "full_seven_stage.episode.json")
    panel = state.model.brush_panel()
    assert isinstance(panel, BrushPanel) and panel.brush_id is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.color_id = 3  # type: ignore[misc]
    assert not [name for name in dir(BrushPanel) if name.startswith(("set", "update", "apply"))]
    assert panel.color_rgb is not None and len(panel.color_rgb) == 3


def test_policy_badge_never_says_high_accuracy_without_report() -> None:
    for trained in (None, False, True):
        for claim in (None, "high_accuracy_claim_allowed", "other"):
            assert "高精度" not in policy_badge(trained, claim_level=claim)
            assert "高精度" not in policy_badge(trained, claim_level=claim, report_ref="")
    assert policy_badge(False) == "未学習／テスト用"
    assert policy_badge(True) == "学習済み（評価は別途）"
    allowed = policy_badge(True, claim_level="high_accuracy_claim_allowed", report_ref="r.json")
    assert "高精度" in allowed and "r.json" in allowed
    assert "高精度" not in policy_badge(False, claim_level="high_accuracy_claim_allowed",
                                        report_ref="r.json")  # fmt: skip


def test_gui_check_flag_stays_headless_and_ok() -> None:
    code = "import sys, image_drawer.gui as g; g.main(['--check']); print('tkinter' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    lines = result.stdout.strip().splitlines()
    assert json.loads(lines[0]) == {"component": "gui", "status": "ok"}
    assert lines[1] == "False" and result.stderr == ""


def test_tk_window_builds_when_display_is_available() -> None:
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"表示が使えないためSKIP（成功ではありません）: {error}")
    root.withdraw()
    try:
        from image_drawer.gui.app import build_main_window

        view = build_main_window(root)
        view.open_path(str(FIXTURES / "full_seven_stage.episode.json"))
        assert view.history.size() > 0
    finally:
        root.destroy()


# --- G2: 生成 ---


def make_generation(tmp_path: Path, service=None):
    from image_drawer.gui.controller import GenerationController

    kwargs = {} if service is None else {"service": service}
    return GenerationController(scripted_registry(), **kwargs)


def request_for(**kwargs) -> DrawRequest:
    return DrawRequest("猫", "scripted", width=SIDE, height=SIDE, **kwargs)


def test_generation_state_machine_idle_generating_complete(tmp_path: Path) -> None:
    import threading

    from image_drawer.gui.controller import GenerationComplete, GenerationIdle, Generating

    gate = threading.Event()

    def gated(request, policies, *, cancel=None, on_progress=None):
        gate.wait(5)
        return generate_episode(request, policies, cancel=cancel, on_progress=on_progress)

    controller = make_generation(tmp_path, gated)
    assert isinstance(controller.state, GenerationIdle)
    assert controller.start(request_for(), tmp_path / "g")
    assert isinstance(controller.state, Generating)
    assert not controller.start(request_for(), tmp_path / "other")  # 生成中は開始しない
    gate.set()
    controller.wait(10)
    state = controller.poll()
    assert isinstance(state, GenerationComplete) and state.episode_path.is_file()
    assert state.policy_trained is False
    loaded = ViewerController().load(state.episode_path, policy_trained=state.policy_trained)
    assert loaded.model.policy_badge == "未学習／テスト用"
    controller.reset()
    assert isinstance(controller.state, GenerationIdle)


def test_generation_stop_requests_cancel_and_shows_draft(tmp_path: Path) -> None:
    import time

    from image_drawer.gui.controller import GenerationDraft

    def until_cancelled(request, policies, *, cancel=None, on_progress=None):
        while not cancel.is_set():
            time.sleep(0.005)
        return generate_episode(request, policies, cancel=cancel)

    controller = make_generation(tmp_path, until_cancelled)
    controller.start(request_for(), tmp_path / "g")
    controller.stop()
    controller.wait(10)
    state = controller.poll()
    assert isinstance(state, GenerationDraft) and state.reason == "cancelled"
    assert state.draft_path.name == "draft.json" and not (tmp_path / "g" / "episode.json").exists()
    shown = ViewerController().load(state.draft_path)
    assert shown.model.draft_badge == "未完成（cancelled）"


def test_generation_failure_and_progress_are_reported(tmp_path: Path) -> None:
    from image_drawer.gui.controller import GenerationFailed, Generating

    controller = make_generation(tmp_path)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x.txt").write_text("x", encoding="utf-8")
    controller.start(request_for(), tmp_path / "busy")
    controller.wait(10)
    assert isinstance(controller.poll(), GenerationFailed)
    controller.reset()
    seen: list[object] = []

    def slow(request, policies, *, cancel=None, on_progress=None):
        outcome = generate_episode(request, policies, cancel=cancel, on_progress=on_progress)
        return outcome

    controller = make_generation(tmp_path, slow)
    controller.start(request_for(), tmp_path / "ok")
    controller.wait(10)
    seen.append(controller._messages.qsize())
    assert seen[0] > 1  # 受理ごとの進捗がキューに積まれる
    assert not isinstance(controller.poll(), Generating)


def test_gui_generation_calls_the_same_service_as_cli(tmp_path: Path) -> None:
    from image_drawer.gui.controller import GenerationController
    from image_drawer.inference import cli, replay
    from image_drawer.steps import draw_episode

    assert GenerationController(scripted_registry()).service is generate_episode
    assert cli.generate_episode is generate_episode
    assert draw_episode.generate_episode is generate_episode
    assert replay.replay_episode_file is cli.replay_episode_file
