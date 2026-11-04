"""製品コードのDrawingEngine（raster-v1）が旧v1 goldenと画素一致することを検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_drawer.drawing import (
    PALETTE,
    RENDERER_V1,
    DrawingAction,
    DrawingEpisode,
    DrawingError,
    Engine,
    loads_strict,
    palette_rgb,
)

FIXTURES = Path(__file__).parent / "fixtures" / "drawing_process_v1"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def run_episode(episode: dict, on_event=None) -> Engine:
    """Episode辞書のeventを1件ずつEngineへ渡す。各event直後にon_eventを呼ぶ。"""
    engine = Engine(episode["canvas"]["width"], episode["canvas"]["height"], profile=RENDERER_V1)
    for event in episode["events"]:
        engine.apply_event(event)
        if on_event is not None:
            on_event(event["seq"], engine)
    return engine


def test_palette_matches_design_values():
    assert len(set(PALETTE)) == 256
    expected = {0: (0, 0, 0), 5: (0, 0, 255), 30: (0, 255, 0), 180: (255, 0, 0),
                215: (255, 255, 255), 216: (6, 6, 6), 255: (249, 249, 249)}  # fmt: skip
    for color_id, rgb in expected.items():
        assert palette_rgb(color_id) == rgb


@pytest.mark.parametrize("bad", [-1, 256, True, 5.0, None, "5"])
def test_palette_rejects_invalid_ids(bad):
    with pytest.raises(DrawingError):
        palette_rgb(bad)


def test_main_golden_matches_every_checkpoint():
    episode = load("golden_episode.json")
    checkpoints = {c["after_seq"]: c for c in load("golden_frames.json")["checkpoints"]}
    seen = []

    def check(seq: int, engine: Engine) -> None:
        if seq not in checkpoints:
            return
        seen.append(seq)
        assert engine.render().rows() == checkpoints[seq]["rgba_rows"]
        for key, expected in checkpoints[seq].get("layer_pixels", {}).items():
            layer_id, pair = key.split("@")
            x, y = (int(value) for value in pair.split(","))
            assert engine.layer_rows(layer_id)[y][x] == expected

    run_episode(episode, check)
    assert seen == sorted(checkpoints)


def test_geometry_goldens_match_layers_and_composite():
    cases = load("golden_geometry.json")["cases"]
    assert len(cases) == 3
    for case in cases:
        frames = {f["after_seq"]: f for f in case["frames"]}

        def check(seq: int, engine: Engine, frames=frames) -> None:
            if seq not in frames:
                return
            assert engine.render().rows() == frames[seq]["rgba_rows"]
            for layer in frames[seq]["layer_rgba"]:
                assert engine.layer_rows(layer["id"]) == layer["rows"]

        run_episode(case["episode"], check)


def test_mixed_actor_golden_replays_to_final_checkpoint():
    episode = load("golden_mixed_actor.json")
    expected = load("golden_frames.json")["checkpoints"][-1]["rgba_rows"]
    engine = run_episode(episode)
    assert engine.state.history[12].actor == "human"
    assert engine.render().rows() == expected


def test_golden_episodes_pass_header_validation():
    for name in ("golden_episode.json", "golden_mixed_actor.json"):
        parsed = DrawingEpisode.from_dict(loads_strict((FIXTURES / name).read_text("utf-8")))
        assert parsed.to_dict()["events"][0]["type"] == "create_layer"


def test_same_actions_give_same_pixels_and_state_round_trips():
    episode = load("golden_episode.json")
    first, second = run_episode(episode), run_episode(episode)
    assert first.render() == second.render()
    assert first.state.to_dict(include_pixels=True) == second.state.to_dict(include_pixels=True)


def make_engine() -> Engine:
    engine = Engine(3, 2, profile=RENDERER_V1)
    for kind, payload in (
        ("create_layer", {"layer_id": "a", "name": "A"}),
        ("select_layer", {"layer_id": "a"}),
        ("set_phase", {"phase": "sketch"}),
        ("select_brush", {"brush_id": "hard_round_v1"}),
        ("select_color", {"color_id": 180}),
    ):
        engine.apply(DrawingAction(kind, payload))
    return engine


def test_selection_actions_do_not_change_pixels():
    engine = make_engine()
    assert engine.layer_rows("a") == [[[0, 0, 0, 0]] * 3] * 2


def test_half_opacity_red_over_white():
    engine = make_engine()
    engine.apply(DrawingAction("stroke", {"points": [[0, 0]], "width_px": 1, "opacity_u8": 127}))
    assert engine.layer_rows("a")[0][0] == [255, 0, 0, 127]
    assert engine.render().pixel(0, 0) == [255, 128, 128, 255]


STROKE = {"points": [[0, 0]], "width_px": 1, "opacity_u8": 255}


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("select_layer", {"layer_id": "missing"}),
        ("create_layer", {"layer_id": "a", "name": "dup"}),
        ("create_layer", {"layer_id": "1bad", "name": "x"}),
        ("select_color", {"color_id": 256}),
        ("select_color", {"color_id": True}),
        ("select_brush", {"brush_id": "ink_gpen"}),
        ("set_phase", {"phase": "paint"}),
        ("stroke", {**STROKE, "points": []}),
        ("stroke", {**STROKE, "points": [[3, 0]]}),
        ("stroke", {**STROKE, "points": [[-1, 0]]}),
        ("stroke", {**STROKE, "points": [[0.0, 0]]}),
        ("stroke", {**STROKE, "points": [[True, 0]]}),
        ("stroke", {**STROKE, "points": [[0]]}),
        ("stroke", {**STROKE, "width_px": 0}),
        ("stroke", {**STROKE, "width_px": 65}),
        ("stroke", {**STROKE, "width_px": True}),
        ("stroke", {**STROKE, "opacity_u8": 0}),
        ("stroke", {**STROKE, "opacity_u8": 256}),
        ("stroke", {**STROKE, "opacity_u8": "255"}),
        ("stroke", {**STROKE, "extra": 1}),
        ("erase", {"points": [[0, 0]], "width_px": 1, "opacity_u8": 1}),
        ("stroke_v2", {"samples": []}),
        ("set_brush_params", {"size_level_id": 1, "opacity_level_id": 1}),
        ("finish", {"x": 1}),
        ("unknown_type", {}),
    ],
)
def test_invalid_v1_actions_raise_and_leave_state_unchanged(kind, payload):
    engine = make_engine()
    before = engine.state.to_dict(include_pixels=True)
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction(kind, payload))
    assert engine.state.to_dict(include_pixels=True) == before


@pytest.mark.parametrize("missing", ["layer", "phase", "brush", "color"])
def test_stroke_requires_layer_phase_brush_and_color(missing):
    engine = Engine(3, 2, profile=RENDERER_V1)
    steps = {
        "layer": [("create_layer", {"layer_id": "a", "name": "A"}),
                  ("select_layer", {"layer_id": "a"})],
        "phase": [("set_phase", {"phase": "sketch"})],
        "brush": [("select_brush", {"brush_id": "hard_round_v1"})],
        "color": [("select_color", {"color_id": 0})],
    }  # fmt: skip
    for name, actions in steps.items():
        if name != missing:
            for kind, payload in actions:
                engine.apply(DrawingAction(kind, payload))
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction("stroke", STROKE))


def test_envelope_errors_and_finish_lock():
    engine = Engine(3, 2, profile=RENDERER_V1)
    layer = {"layer_id": "a", "name": "A"}
    for bad in (
        DrawingAction("create_layer", layer, seq=1),
        DrawingAction("create_layer", layer, actor="robot"),
        DrawingAction("create_layer", layer, time_ms=-1),
        DrawingAction("create_layer", layer, time_ms=604_800_001),
    ):
        with pytest.raises(DrawingError):
            engine.apply(bad)
    engine.apply(DrawingAction("create_layer", layer, time_ms=5))
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction("select_layer", {"layer_id": "a"}, time_ms=4))
    engine.apply(DrawingAction("finish", {}))
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction("select_layer", {"layer_id": "a"}))


def test_layer_limit_and_canvas_validation():
    engine = Engine(3, 2, profile=RENDERER_V1)
    for index in range(16):
        engine.apply(DrawingAction("create_layer", {"layer_id": f"l{index}", "name": "x"}))
    with pytest.raises(DrawingError, match="layer limit"):
        engine.apply(DrawingAction("create_layer", {"layer_id": "extra", "name": "x"}))
    for width, height in ((0, 1), (1025, 1), (True, 1), (1, 1.5)):
        with pytest.raises(DrawingError):
            Engine(width, height)
    with pytest.raises(DrawingError):
        Engine(1, 1, profile="nope")


def test_area_budget_rejects_before_any_pixel_changes():
    engine = Engine(1024, 1024, profile=RENDERER_V1)
    for index in range(4):
        engine.apply(DrawingAction("create_layer", {"layer_id": f"l{index}", "name": "x"}))
    with pytest.raises(DrawingError, match="area"):
        engine.apply(DrawingAction("create_layer", {"layer_id": "l4", "name": "x"}))


def test_strict_json_rejects_decimals_duplicates_and_nan():
    for text in ('{"a": 5.0}', '{"a": 5e0}', '{"a": 1, "a": 2}', '{"a": NaN}'):
        with pytest.raises(DrawingError):
            loads_strict(text)


def test_episode_header_mismatches_are_rejected():
    raw = load("golden_episode.json")
    for key, value in (("palette_id", "other"), ("brush_set_id", "brush-v2-core12"),
                       ("renderer_version", "raster-v9"), ("origin", "human")):  # fmt: skip
        with pytest.raises(DrawingError):
            DrawingEpisode.from_dict({**raw, key: value})
    with pytest.raises(DrawingError):
        DrawingEpisode.from_dict({**raw, "events": raw["events"][:-1]})
    with pytest.raises(DrawingError):
        DrawingEpisode.from_dict({**raw, "unknown": 1})
