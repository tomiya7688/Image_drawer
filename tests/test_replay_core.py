"""S1c-3: v2 Replayerの基本動作（Controllerの状態との一致・until_seq・失敗・決定性）。"""

from __future__ import annotations

import copy
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
from episode_helpers import (
    draft_controller,
    full_controller,
    full_raw,
    groups_of,
    raw_of,
    rebuild,
    replay_pythonhashseed_script,
    review_controller,
    transition_index,
)
from stage_helpers import engine_of

from image_drawer.drawing import DrawingEpisode, Engine
from image_drawer.drawing import replay as replay_module
from image_drawer.drawing.episode_io import save_episode
from image_drawer.drawing.models import LAYER_ROLES_Z_ORDER
from image_drawer.drawing.pixels import PixelBuffer, composite_on_white
from image_drawer.drawing.replay import iter_replay, replay
from image_drawer.drawing.validation import ReplayError

ROOT = Path(__file__).resolve().parents[1]


def signature(state) -> tuple:
    """画素・レイヤー（id・role・visible）・選択状態・工程・finishedの比較用の値。"""
    layers = [(la.id, la.role, la.visible, bytes(la.pixels.data)) for la in state.layers]
    return (
        layers, state.active_layer_id, state.selected_color_id, state.selected_brush_id,
        state.size_level_id, state.opacity_level_id, state.stage_id, state.finished,
    )  # fmt: skip


def test_replay_of_v2_episode_matches_controller_engine_pixels():
    controller = full_controller()
    result = replay(controller.episode())
    assert result.image.rgba == engine_of(controller).render().rgba
    assert signature(result.state) == signature(engine_of(controller).state)


def test_replay_matches_layers_visibility_selection_and_stage_of_controller():
    for controller in (full_controller(), review_controller(), draft_controller()):
        result = replay(controller.episode())
        expected = engine_of(controller).state
        assert signature(result.state) == signature(expected)
        assert result.state.stage_id == controller.stage_id
        assert [(la.layer_id, la.visible) for la in result.layers] == [
            (la.id, la.visible) for la in expected.layers
        ]


def test_replay_returns_layers_in_creation_order_with_roles():
    result = replay(full_controller().episode())
    assert [(la.layer_id, la.role) for la in result.layers] == [
        ("sys_rough_001", "rough"), ("sys_sketch_001", "sketch"),
        ("sys_final_drawing_001", "final_drawing"), ("sys_base_color_001", "base_color"),
        ("sys_shade_001", "shade"),
    ]  # fmt: skip
    assert all(len(la.rgba) == 12 * 8 * 4 for la in result.layers)


def test_replay_image_is_composited_by_role_z_order():
    result = replay(full_controller().episode())
    visible = sorted(
        (la for la in result.layers if la.visible),
        key=lambda la: LAYER_ROLES_Z_ORDER.index(la.role),
    )
    buffers = []
    for layer in visible:
        buffer = PixelBuffer(12, 8)
        buffer.data[:] = layer.rgba
        buffers.append(buffer)
    assert result.image.rgba == composite_on_white(buffers, 12, 8)


def group_end_seq(episode: DrawingEpisode, group_index: int) -> int:
    return groups_of(episode.to_dict())[group_index][-1]["seq"]


def test_replay_until_seq_at_group_end_matches_intermediate_state():
    episode = full_controller().episode()
    for group_index in (0, 5, 12, 20):
        seq = group_end_seq(episode, group_index)
        partial = dataclasses.replace(
            episode, events=episode.events[: seq + 1], status="draft",
            termination_reason="in_progress",
        )  # fmt: skip
        assert signature(replay(episode, until_seq=seq).state) == signature(replay(partial).state)


def test_replay_until_seq_inside_group_raises():
    episode = full_controller().episode()
    index = transition_index(episode.to_dict())
    inside = groups_of(episode.to_dict())[index][1]["seq"]
    with pytest.raises(ReplayError, match="group"):
        replay(episode, until_seq=inside)


def test_replay_until_seq_out_of_range_raises():
    episode = full_controller().episode()
    for seq in (-1, len(episode.events), "a", 1.5):
        with pytest.raises(ReplayError):
            replay(episode, until_seq=seq)


def test_replay_of_draft_returns_state_before_unfinished_stage_end():
    controller = draft_controller()
    assert controller.termination_reason == "action_budget_exhausted"
    result = replay(controller.episode())
    assert not result.state.finished and result.state.stage_id == "final_drawing"
    assert signature(result.state) == signature(engine_of(controller).state)


def test_replay_does_not_mutate_episode():
    episode = full_controller().episode()
    before = copy.deepcopy(episode)
    replay(episode)
    list(iter_replay(episode))
    assert episode == before


def inserted_raw(event: dict, after_group: int = 0) -> dict:
    raw = full_raw()
    groups = groups_of(raw)
    groups.insert(after_group + 1, [event])
    return rebuild(raw, groups)


def model_event(type_: str, payload: dict, stage: str = "rough") -> dict:
    return {
        "seq": 0, "time_ms": 0, "actor": "model", "type": type_, "payload": payload,
        "stage_id": stage, "group": {"id": 0, "index": 0, "size": 1},
    }  # fmt: skip


def test_replay_raises_replay_error_with_seq_on_engine_rejection():
    raw = inserted_raw(model_event("select_layer", {"layer_id": "nope"}))
    with pytest.raises(ReplayError) as info:
        replay(DrawingEpisode.from_dict(raw))
    assert info.value.seq == 3 and "seq 3" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_replay_detects_stage_id_annotation_mismatch():
    raw = inserted_raw(model_event("select_layer", {"layer_id": "sys_rough_001"}, "sketch"))
    with pytest.raises(ReplayError, match="annotation"):
        replay(DrawingEpisode.from_dict(raw))


def test_replay_detects_hash_or_header_mismatch_before_applying(monkeypatch):
    episode = full_controller().episode()
    applied = []
    monkeypatch.setattr(Engine, "apply", lambda self, action: applied.append(action))
    for change in ({"workflow_sha256": "0" * 64}, {"workflow_id": "x"}, {"seed": -1}):
        with pytest.raises(ReplayError):
            replay(dataclasses.replace(episode, **change))
    assert applied == []


def test_replay_is_deterministic_across_two_runs():
    episode = full_controller().episode()
    first, second = replay(episode), replay(episode)
    assert first.image == second.image and first.layers == second.layers


def run_with_hashseed(path: Path, seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
    done = subprocess.run(
        [sys.executable, "-c", replay_pythonhashseed_script(str(path))],
        capture_output=True, text=True, env=env, cwd=ROOT, check=True, timeout=120,
    )  # fmt: skip
    return done.stdout.strip()


def test_replay_is_independent_of_pythonhashseed(tmp_path):
    save_episode(full_controller().episode(), tmp_path / "e.json")
    digests = {run_with_hashseed(tmp_path / "e.json", seed) for seed in ("1", "12345")}
    assert len(digests) == 1 and len(digests.pop()) == 64


def test_replay_uses_group_atomic_apply_for_transition_groups():
    raw = full_raw()
    groups = groups_of(raw)
    index = transition_index(raw, 1)
    create = next(e for e in groups[index] if e["type"] == "create_layer")
    create["payload"]["layer_id"] = "sys_rough_001"  # 重複IDで末尾近くのeventをEngineに拒否させる
    episode = DrawingEpisode.from_dict(raw)
    engine = Engine(12, 8, seed=0)
    for group in replay_module._partition(episode, True)[:index]:
        replay_module._apply(engine, group, True)
    before = signature(engine.state), len(engine.state.history)
    with pytest.raises(ReplayError):
        replay_module._apply(engine, replay_module._partition(episode, True)[index], True)
    assert (signature(engine.state), len(engine.state.history)) == before


def test_iter_replay_yields_one_frame_per_event_and_marks_group_ends():
    episode = full_controller().episode()
    frames = list(iter_replay(episode))
    assert [f.event.seq for f in frames] == list(range(len(episode.events)))
    ends = [f.event.seq for f in frames if f.is_group_end]
    assert ends == [g[-1]["seq"] for g in groups_of(episode.to_dict())]


def test_iter_replay_capture_matches_replay_until_seq():
    episode = review_controller().episode()
    for frame in iter_replay(episode):
        if frame.is_group_end:
            expected = replay(episode, until_seq=frame.event.seq)
            captured = frame.capture()
            assert captured.image == expected.image and captured.layers == expected.layers
            assert signature(captured.state) == signature(expected.state)
        elif frame.event.group.index == 0 and frame.event.group.size > 1:
            with pytest.raises(ReplayError):
                frame.capture()


def test_replay_hidden_layers_are_excluded_from_image_but_kept_in_layers():
    result = replay(full_controller().episode())
    hidden = [la for la in result.layers if not la.visible]
    assert {la.role for la in hidden} == {"rough", "sketch"}
    assert any(any(la.rgba) for la in hidden)
    visible_only = []
    for layer in sorted(
        (la for la in result.layers if la.visible),
        key=lambda la: LAYER_ROLES_Z_ORDER.index(la.role),
    ):
        buffer = PixelBuffer(12, 8)
        buffer.data[:] = layer.rgba
        visible_only.append(buffer)
    assert result.image.rgba == composite_on_white(visible_only, 12, 8)


def test_raw_of_roundtrip_helper_is_consistent():
    assert DrawingEpisode.from_dict(raw_of(full_controller().episode())).to_dict() == full_raw()
