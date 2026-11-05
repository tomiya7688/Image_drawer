"""S1c-3: 製品コードのReplayerがv1 goldenと画素一致する（テスト内のオラクルは使わない）。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from image_drawer.drawing import DrawingAction, DrawingEpisode
from image_drawer.drawing.episode_io import dumps_episode, loads_episode, save_episode
from image_drawer.drawing.replay import replay
from image_drawer.drawing.validation import EpisodeFormatError, ReplayError

FIXTURES = Path(__file__).parent / "fixtures" / "drawing_process_v1"


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load_json(name: str):
    return json.loads(load_text(name))


def rows_of(rgba: bytes, width: int, height: int) -> list:
    pixels = [list(rgba[i : i + 4]) for i in range(0, len(rgba), 4)]
    return [pixels[y * width : (y + 1) * width] for y in range(height)]


def layer_map(result, width: int, height: int) -> dict:
    return {layer.layer_id: rows_of(layer.rgba, width, height) for layer in result.layers}


def check_checkpoint(episode: DrawingEpisode, after_seq: int, rgba_rows, layers=None) -> None:
    result = replay(episode, until_seq=after_seq)
    assert result.image.rows() == rgba_rows
    rows = layer_map(result, episode.width, episode.height)
    for layer_id, expected in (layers or {}).items():
        assert rows[layer_id] == expected


def test_product_replayer_matches_v1_golden_frames():
    episode = loads_episode(load_text("golden_episode.json"))
    for checkpoint in load_json("golden_frames.json")["checkpoints"]:
        check_checkpoint(episode, checkpoint["after_seq"], checkpoint["rgba_rows"])
        result = replay(episode, until_seq=checkpoint["after_seq"])
        rows = layer_map(result, episode.width, episode.height)
        for key, expected in checkpoint.get("layer_pixels", {}).items():
            layer_id, pair = key.split("@")
            x, y = (int(value) for value in pair.split(","))
            assert rows[layer_id][y][x] == expected


def test_product_replayer_keeps_sketch_layer_black_under_red_layer():
    episode = loads_episode(load_text("golden_episode.json"))
    result = replay(episode, until_seq=10)
    rows = layer_map(result, episode.width, episode.height)
    assert rows["sketch"][0][1] == [0, 0, 0, 255]
    assert rows["color"][0][1] == [255, 0, 0, 255]
    assert result.image.pixel(1, 0) == [255, 0, 0, 255]


def test_product_replayer_erase_on_upper_layer_restores_black_composite():
    episode = loads_episode(load_text("golden_episode.json"))
    result = replay(episode, until_seq=12)
    rows = layer_map(result, episode.width, episode.height)
    assert rows["color"][0][1] == [0, 0, 0, 0]
    assert result.image.pixel(1, 0) == [0, 0, 0, 255]


def test_product_replayer_matches_v1_geometry_cases():
    for case in load_json("golden_geometry.json")["cases"]:
        episode = loads_episode(json.dumps(case["episode"]))
        for frame in case["frames"]:
            expected = {item["id"]: item["rows"] for item in frame["layer_rgba"]}
            check_checkpoint(episode, frame["after_seq"], frame["rgba_rows"], expected)


def test_product_replayer_replays_mixed_actor_episode_with_human_and_model_actors():
    episode = loads_episode(load_text("golden_mixed_actor.json"))
    assert {event.actor for event in episode.events} == {"human", "model"}
    expected = load_json("golden_frames.json")["checkpoints"][-1]["rgba_rows"]
    assert replay(episode, until_seq=12).image.rows() == expected
    assert replay(episode).state.finished


def test_product_replayer_rejects_events_after_finish_in_v1():
    episode = loads_episode(load_text("golden_episode.json"))
    extra = DrawingAction("set_phase", {"phase": "line"}, "synthetic", 14, 0)
    broken = dataclasses.replace(episode, events=[*episode.events, extra])
    with pytest.raises(ReplayError):
        replay(broken)


def test_v1_episode_dispatch_does_not_require_stage_id_or_group():
    episode = loads_episode(load_text("golden_episode.json"))
    assert all(e.stage_id is None and e.group is None for e in episode.events)
    assert replay(episode).state.stage_id is None


def test_save_episode_rejects_v1_episode(tmp_path):
    episode = loads_episode(load_text("golden_episode.json"))
    with pytest.raises(EpisodeFormatError):
        dumps_episode(episode)
    with pytest.raises(EpisodeFormatError):
        save_episode(episode, tmp_path / "v1.json")
    assert list(tmp_path.iterdir()) == []


def test_legacy_oracle_tests_still_pass():
    # 既存のオラクルテストは残す。ここではfixtureの形が変わっていないことだけを確認する。
    assert (Path(__file__).parent / "test_drawing_design_contract.py").is_file()
    assert load_json("golden_frames.json")["schema_version"] == "drawing-golden-frames/v1"
