"""S1c-4: 7工程fixtureを製品コードのReplayerで再生し、記録したframesと画素一致させる。

fixtureの作り直し: python tests/test_replay_stage_episodes.py --regenerate
（再生成の差分は、画素の16進を目視でレビューする）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from episode_fixture_gen import FIXTURE_DIR, NAMES, build_fixture, regenerate  # noqa: E402

from image_drawer.drawing.episode_io import dumps_episode, load_episode  # noqa: E402
from image_drawer.drawing.models import STAGE_IDS  # noqa: E402
from image_drawer.drawing.palette import palette_rgb  # noqa: E402
from image_drawer.drawing.replay import replay  # noqa: E402

COMPLETE = ("full_seven_stage", "review_only_stages")


def episode_of(name: str):
    return load_episode(FIXTURE_DIR / f"{name}.episode.json")


def frames_of(name: str) -> list[dict]:
    return json.loads((FIXTURE_DIR / f"{name}.frames.json").read_text(encoding="utf-8"))["frames"]


def assert_replays_to_frames(name: str) -> None:
    episode = episode_of(name)
    frames = frames_of(name)
    assert frames[-1]["after_seq"] == len(episode.events) - 1
    for frame in frames:
        result = replay(episode, until_seq=frame["after_seq"])
        assert result.state.stage_id == frame["stage_id"]
        assert result.image.rgba.hex() == frame["rgba_hex"]
        assert [
            {"id": la.layer_id, "role": la.role, "visible": la.visible, "rgba_hex": la.rgba.hex()}
            for la in result.layers
        ] == frame["layers"]


def test_full_seven_stage_replays_to_golden_frames_at_every_group_end():
    assert_replays_to_frames("full_seven_stage")
    assert replay(episode_of("full_seven_stage")).state.finished


def test_review_only_stages_replays_to_golden_frames_at_every_group_end():
    assert_replays_to_frames("review_only_stages")
    assert replay(episode_of("review_only_stages")).state.finished


def test_draft_fixture_replays_to_golden_frames_and_is_not_finished():
    assert_replays_to_frames("draft_budget_exhausted")
    episode = episode_of("draft_budget_exhausted")
    assert (episode.status, episode.termination_reason) == ("draft", "action_budget_exhausted")
    assert not replay(episode).state.finished


@pytest.mark.parametrize("name", NAMES)
def test_fixtures_load_with_strict_loader_and_pass_structure_validation(name):
    episode = episode_of(name)  # load_episodeがstrict loaderと構造検証を通す
    assert episode.renderer_version == "brush-raster-v2"
    assert (episode.width, episode.height) == (12, 8)


@pytest.mark.parametrize("name", NAMES)
def test_fixtures_are_canonical_bytes(name):
    path = FIXTURE_DIR / f"{name}.episode.json"
    assert path.read_bytes() == dumps_episode(episode_of(name)).encode("utf-8")
    assert (
        b"\r" not in path.read_bytes()
        and b"\r" not in path.with_name(f"{name}.frames.json").read_bytes()
    )


@pytest.mark.parametrize("name", NAMES)
def test_fixture_frames_match_controller_regeneration(name):
    episode, frames = build_fixture(name)
    assert episode == episode_of(name)
    assert frames == frames_of(name)


def final_pixels(role: str) -> list[list[int]]:
    result = replay(episode_of("full_seven_stage"))
    layer = next(la for la in result.layers if la.role == role)
    return [list(layer.rgba[i : i + 4]) for i in range(0, len(layer.rgba), 4)]


def test_full_fixture_base_color_does_not_cover_final_drawing_line():
    result = replay(episode_of("full_seven_stage"))
    final, highlight = final_pixels("final_drawing"), final_pixels("highlight")
    correction, base = final_pixels("correction"), final_pixels("base_color")
    assert any(pixel[3] == 255 for pixel in base)  # 基本色が実際に塗られている
    checked = 0
    for index, pixel in enumerate(final):
        if pixel[3] == 255 and highlight[index][3] == 0 and correction[index][3] == 0:
            assert list(result.image.rgba[index * 4 : index * 4 + 3]) == list(palette_rgb(0))
            checked += 1
    assert checked > 0


def test_full_fixture_rough_and_sketch_layers_are_hidden_but_present():
    result = replay(episode_of("full_seven_stage"))
    by_role = {la.role: la for la in result.layers}
    for role in ("rough", "sketch"):
        assert by_role[role].visible is False and any(by_role[role].rgba)


def test_full_fixture_highlight_pixels_render_above_final_drawing():
    result = replay(episode_of("full_seven_stage"))
    final, highlight = final_pixels("final_drawing"), final_pixels("highlight")
    correction = final_pixels("correction")
    checked = 0
    for index, pixel in enumerate(highlight):
        if pixel[3] == 255 and final[index][3] == 255 and correction[index][3] == 0:
            assert list(result.image.rgba[index * 4 : index * 4 + 3]) == list(palette_rgb(215))
            checked += 1
    assert checked > 0


def test_review_fixture_has_enter_and_exit_events_for_reviewed_stages():
    events = episode_of("review_only_stages").events
    for stage in ("base_color", "shading", "refine"):
        for kind in ("enter_stage", "exit_stage"):
            assert any(e.type == kind and e.payload["stage_id"] == stage for e in events)


def test_review_fixture_has_no_stroke_events_in_reviewed_stages():
    events = episode_of("review_only_stages").events
    reviewed = {"base_color", "shading", "refine"}
    assert not [e for e in events if e.stage_id in reviewed and e.type in ("stroke_v2", "erase_v2")]
    reasons = [e.payload.get("review_reason") for e in events if e.type == "advance_stage"]
    assert reasons[3:] == [
        "monochrome_line_art_only",
        "flat_style_no_shading",
        "no_correction_needed",
    ]


@pytest.mark.parametrize("name", COMPLETE)
def test_every_fixture_contains_all_six_enter_stage_events_in_order_when_complete(name):
    events = episode_of(name).events
    assert [e.payload["stage_id"] for e in events if e.type == "enter_stage"] == list(STAGE_IDS)


@pytest.mark.parametrize("name", NAMES)
def test_fixture_actors_are_model_and_system_only_and_origin_is_mixed(name):
    episode = episode_of(name)
    assert episode.origin == "mixed"
    assert {e.actor for e in episode.events} == {"model", "system"}


@pytest.mark.parametrize("name", NAMES)
def test_fixture_metadata_marks_scripted_source(name):
    assert episode_of(name).metadata["source"] == "scripted-fixture"


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        print("usage: python tests/test_replay_stage_episodes.py --regenerate")
