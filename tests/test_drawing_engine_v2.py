"""brush-raster-v2の回帰用画素golden（12ブラシ）と異常系。

goldenは「現在の式が同じ画素を出し続けること」の回帰用であり、画質の良し悪しを主張しない。
式は docs/developers/drawing_engine_v2.md を参照。fixtureを作り直すには、リポジトリ直下で
`python tests/test_drawing_engine_v2.py --regenerate` を実行する（式を意図して変えたときだけ）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from image_drawer.drawing import (  # noqa: E402
    RENDERER_V2,
    DrawingAction,
    DrawingEpisode,
    DrawingError,
    Engine,
    default_catalog,
)
from image_drawer.drawing.catalog import load_catalog  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "drawing_engine_v2" / "golden_brushes.json"
DOCS = ROOT / "docs" / "developers"
WIDTH, HEIGHT, SEED = 20, 14, 12345
SIZE_LEVEL_BY_BRUSH = {
    "pencil_graphite": 3, "ink_gpen": 4, "ink_fine": 1, "marker_chisel": 6,
    "paint_round_opaque": 7, "paint_flat": 7, "watercolor_soft": 7, "airbrush_soft": 7,
    "pastel_oil": 7, "dry_texture": 7, "eraser_hard": 6, "eraser_soft": 7,
}  # fmt: skip
PRESSURES = [2, 5, 10, 6, 0, 8]
PATH_Q8 = [(640, 832), (2048, 1100), (4300, 1500), (3100, 2400), (1280, 2900), (2200, 3300)]


def hex_rows(rows: list[list[list[int]]]) -> list[str]:
    """画素を行ごとの16進文字列（rrggbbaa連結）にする。"""
    return ["".join(f"{r:02x}{g:02x}{b:02x}{a:02x}" for r, g, b, a in row) for row in rows]


def samples(path=PATH_Q8, pressures=PRESSURES) -> list[dict]:
    return [
        {"x_q8": x, "y_q8": y, "time_us": 16000 * index, "pressure_level_id": p,
         "tilt_x_level_id": None, "tilt_y_level_id": None}
        for index, ((x, y), p) in enumerate(zip(path, pressures))
    ]  # fmt: skip


def brush_actions(brush_id: str) -> list[dict]:
    """1ブラシ分の固定Action列。消しゴムは先に黒の太線で塗ってから消す。"""
    catalog = default_catalog()
    preset = catalog.get(brush_id)
    layer = [
        {"type": "create_layer", "payload": {"layer_id": "w", "name": "work", "role": "rough"}},
        {"type": "select_layer", "payload": {"layer_id": "w"}},
    ]
    if preset.mode == "erase":
        layer += [
            {"type": "select_brush", "payload": {"brush_id": "paint_round_opaque"}},
            {"type": "set_brush_params", "payload": {"size_level_id": 10, "opacity_level_id": 10}},
            {"type": "select_color", "payload": {"color_id": 0}},
            {"type": "stroke_v2", "payload": {"samples": samples(
                [(0, 1024), (4864, 1024), (4864, 2560), (0, 2560)], [10, 10, 10, 10])}},
        ]  # fmt: skip
    return layer + [
        {"type": "select_brush", "payload": {"brush_id": brush_id}},
        {"type": "set_brush_params", "payload": {
            "size_level_id": SIZE_LEVEL_BY_BRUSH[brush_id],
            "opacity_level_id": preset.default_opacity_level_id}},
        {"type": "select_color", "payload": {"color_id": 180}},
        {"type": "erase_v2" if preset.mode == "erase" else "stroke_v2",
         "payload": {"samples": samples()}},
    ]  # fmt: skip


def run_actions(actions: list[dict], seed: int = SEED) -> Engine:
    engine = Engine(WIDTH, HEIGHT, profile=RENDERER_V2, seed=seed)
    for item in actions:
        engine.apply(DrawingAction(item["type"], item["payload"], actor="synthetic"))
    return engine


def roles_actions() -> list[dict]:
    """z順（作成順ではなくrole順）と非表示を確かめる小さなシナリオ。"""

    def stroke(color: int, y_q8: int) -> list[dict]:
        return [
            {"type": "select_color", "payload": {"color_id": color}},
            {"type": "stroke_v2", "payload": {"samples": samples(
                [(512, y_q8), (4608, y_q8)], [10, 10])}},
        ]  # fmt: skip

    setup = [
        {"type": "create_layer", "payload": {"layer_id": "top", "name": "t", "role": "highlight"}},
        {"type": "create_layer", "payload": {"layer_id": "low", "name": "l", "role": "rough"}},
        {"type": "select_brush", "payload": {"brush_id": "ink_fine"}},
        {"type": "set_brush_params", "payload": {"size_level_id": 3, "opacity_level_id": 10}},
        {"type": "select_layer", "payload": {"layer_id": "top"}},
    ]
    return (
        setup
        + stroke(0, 768)
        + [{"type": "select_layer", "payload": {"layer_id": "low"}}]
        + stroke(180, 768)
        + stroke(5, 1792)
        + [{"type": "set_layer_visibility", "payload": {"layer_id": "low", "visible": False}}]
    )


def build_fixture() -> dict:
    catalog = default_catalog()
    cases = []
    for preset in catalog.presets:
        actions = brush_actions(preset.id)
        engine = run_actions(actions)
        cases.append({"brush_id": preset.id, "actions": actions,
                      "layer_hex": hex_rows(engine.layer_rows("w"))})  # fmt: skip
    roles = roles_actions()
    engine = run_actions(roles)
    return {
        "note": "回帰用golden。画質の良し悪しを主張しない。docs/developers/drawing_engine_v2.md参照。",
        "canvas": {"width": WIDTH, "height": HEIGHT}, "seed": SEED,
        "brush_sha256": catalog.brush_sha256, "control_sha256": catalog.control_sha256,
        "cases": cases,
        "roles_case": {"actions": roles, "render_hex": hex_rows(engine.render().rows())},
    }  # fmt: skip


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


GOLDEN = load_fixture() if FIXTURE.exists() else {"cases": []}


def test_fixture_covers_all_twelve_brushes_with_visible_pixels():
    ids = [case["brush_id"] for case in GOLDEN["cases"]]
    assert ids == [preset.id for preset in default_catalog().presets]
    assert GOLDEN["brush_sha256"] == default_catalog().brush_sha256
    for case in GOLDEN["cases"]:
        assert any(set(row) != {"0"} for row in case["layer_hex"]), case["brush_id"]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["brush_id"])
def test_brush_pixels_match_golden(case):
    engine = run_actions(case["actions"], seed=GOLDEN["seed"])
    assert hex_rows(engine.layer_rows("w")) == case["layer_hex"]


def test_roles_case_uses_role_z_order_and_visibility():
    engine = run_actions(GOLDEN["roles_case"]["actions"])
    assert hex_rows(engine.render().rows()) == GOLDEN["roles_case"]["render_hex"]
    # lowはrough（下）、topはhighlight（上）。非表示のlowは合成に出ないが画素は残る。
    assert engine.layer_rows("low")[3][4][3] > 0
    assert engine.render().pixel(4, 3)[:3] in ([0, 0, 0], [255, 255, 255])


def test_catalog_is_validated_and_matches_docs_copies():
    for name in ("brush_presets_v2.json", "brush_control_levels_v2.json"):
        packaged = Path(__file__).parents[1] / "image_drawer" / "drawing" / "data" / name
        assert packaged.read_bytes() == (DOCS / name).read_bytes()
    catalog = load_catalog(DOCS)
    assert (len(catalog.presets), len(catalog.size_px), len(catalog.opacity_percent)) == (
        12,
        19,
        11,
    )
    assert catalog.size_px[:3] == (1, 2, 3) and catalog.size_px[-1] == 256
    assert catalog.opacity_u16(10) == 65535 and catalog.pressure_u16(None) == 65535


def test_same_actions_same_pixels_and_seed_changes_only_grain():
    actions = brush_actions("pastel_oil")
    assert run_actions(actions).render() == run_actions(actions).render()
    assert run_actions(actions, seed=1).render() != run_actions(actions, seed=2).render()
    smooth = brush_actions("ink_gpen")
    assert run_actions(smooth, seed=1).render() == run_actions(smooth, seed=2).render()


def test_pressure_changes_width_and_opacity_monotonically():
    def inked(pressure: int) -> list[int]:
        engine = Engine(12, 12, profile=RENDERER_V2)
        for kind, payload in (
            ("create_layer", {"layer_id": "w", "name": "w", "role": "rough"}),
            ("select_layer", {"layer_id": "w"}),
            ("select_brush", {"brush_id": "paint_round_opaque"}),
            ("set_brush_params", {"size_level_id": 7, "opacity_level_id": 10}),
            ("select_color", {"color_id": 0}),
            ("stroke_v2", {"samples": samples([(1536, 1536)], [pressure])}),
        ):
            engine.apply(DrawingAction(kind, payload))
        return [px[3] for row in engine.layer_rows("w") for px in row]

    weak, strong = inked(0), inked(10)
    assert sum(strong) > sum(weak) > 0
    assert max(strong) == 255 and max(weak) < 255


def test_eraser_only_lowers_alpha_of_the_selected_layer():
    actions = GOLDEN["cases"][-2]["actions"]  # eraser_hard
    before = run_actions(actions[:-1]).layer_rows("w")
    after = run_actions(actions).layer_rows("w")
    alphas = [(b[3], a[3]) for rb, ra in zip(before, after) for b, a in zip(rb, ra)]
    assert any(new < old for old, new in alphas)
    assert all(new <= old for old, new in alphas)
    assert all(a[:3] == b[:3] for rb, ra in zip(before, after) for b, a in zip(rb, ra) if a[3])


def test_v2_episode_header_round_trip_and_checksum_guard():
    actions = brush_actions("ink_fine")
    engine = run_actions(actions)
    engine.apply(DrawingAction("finish", {}, actor="synthetic"))
    episode = DrawingEpisode(
        "e1", "synthetic", "p", WIDTH, HEIGHT, RENDERER_V2, engine.state.history, {}, SEED
    )
    raw = episode.to_dict()
    assert DrawingEpisode.from_dict(raw).to_dict() == raw
    with pytest.raises(DrawingError):
        DrawingEpisode.from_dict({**raw, "brush_manifest_sha256": "0" * 64})


def base_engine() -> Engine:
    engine = Engine(WIDTH, HEIGHT, profile=RENDERER_V2)
    for kind, payload in (
        ("create_layer", {"layer_id": "w", "name": "w", "role": "rough"}),
        ("select_layer", {"layer_id": "w"}),
        ("select_brush", {"brush_id": "ink_gpen"}),
        ("select_color", {"color_id": 0}),
    ):
        engine.apply(DrawingAction(kind, payload))
    return engine


GOOD = samples([(512, 512)], [5])[0]


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("select_color", {"color_id": 256}),
        ("select_brush", {"brush_id": "hard_round_v1"}),
        ("select_brush", {"brush_id": "nope"}),
        ("set_brush_params", {"size_level_id": 19, "opacity_level_id": 0}),
        ("set_brush_params", {"size_level_id": 0, "opacity_level_id": 11}),
        ("set_brush_params", {"size_level_id": -1, "opacity_level_id": 0}),
        ("set_brush_params", {"size_level_id": True, "opacity_level_id": 0}),
        ("set_brush_params", {"size_level_id": 1}),
        ("select_layer", {"layer_id": "missing"}),
        ("create_layer", {"layer_id": "x", "name": "x"}),
        ("create_layer", {"layer_id": "x", "name": "x", "role": "bogus"}),
        ("create_layer", {"layer_id": "w", "name": "x", "role": "sketch"}),
        ("set_phase", {"phase": "sketch"}),
        ("stroke", {"points": [[0, 0]], "width_px": 1, "opacity_u8": 9}),
        ("set_layer_visibility", {"layer_id": "w", "visible": 1}),
        ("stroke_v2", {"samples": []}),
        ("stroke_v2", {"samples": [{**GOOD, "x_q8": -1}]}),
        ("stroke_v2", {"samples": [{**GOOD, "x_q8": (WIDTH - 1) * 256 + 1}]}),
        ("stroke_v2", {"samples": [{**GOOD, "y_q8": (HEIGHT - 1) * 256 + 1}]}),
        ("stroke_v2", {"samples": [{**GOOD, "x_q8": float("nan")}]}),
        ("stroke_v2", {"samples": [{**GOOD, "x_q8": float("inf")}]}),
        ("stroke_v2", {"samples": [{**GOOD, "x_q8": 1.5}]}),
        ("stroke_v2", {"samples": [{**GOOD, "time_us": 5}]}),
        ("stroke_v2", {"samples": [{**GOOD, "pressure_level_id": 11}]}),
        ("stroke_v2", {"samples": [{**GOOD, "pressure_level_id": True}]}),
        ("stroke_v2", {"samples": [{**GOOD, "tilt_x_level_id": 3}]}),
        ("stroke_v2", {"samples": [{**GOOD, "tilt_x_level_id": 13, "tilt_y_level_id": 0}]}),
        ("stroke_v2", {"samples": [{k: v for k, v in GOOD.items() if k != "time_us"}]}),
        ("stroke_v2", {"samples": [GOOD] * 4097}),
        ("stroke_v2", {"samples": [GOOD], "extra": 1}),
        ("erase_v2", {"samples": [GOOD]}),
    ],
)
def test_invalid_v2_actions_raise_and_leave_state_unchanged(kind, payload):
    engine = base_engine()
    before = engine.state.to_dict(include_pixels=True)
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction(kind, payload))
    assert engine.state.to_dict(include_pixels=True) == before


def test_time_us_must_not_decrease_and_tilt_levels_are_accepted():
    engine = base_engine()
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction("stroke_v2", {"samples": [
            GOOD, {**GOOD, "time_us": 100}, {**GOOD, "time_us": 50}]}))  # fmt: skip
    tilted = {**GOOD, "tilt_x_level_id": 6, "tilt_y_level_id": 12}
    engine.apply(DrawingAction("stroke_v2", {"samples": [tilted]}))


def test_brush_mode_and_selection_preconditions():
    engine = base_engine()
    engine.apply(DrawingAction("select_brush", {"brush_id": "eraser_hard"}))
    with pytest.raises(DrawingError, match="cannot be used"):
        engine.apply(DrawingAction("stroke_v2", {"samples": [GOOD]}))
    engine.apply(DrawingAction("erase_v2", {"samples": [GOOD]}))  # 色を参照しない

    bare = Engine(WIDTH, HEIGHT, profile=RENDERER_V2)
    with pytest.raises(DrawingError):
        bare.apply(DrawingAction("set_brush_params", {"size_level_id": 0, "opacity_level_id": 0}))
    with pytest.raises(DrawingError):
        bare.apply(DrawingAction("stroke_v2", {"samples": [GOOD]}))


def test_select_brush_resets_levels_and_set_params_applies():
    engine = base_engine()
    preset = default_catalog().get("ink_gpen")
    assert engine.state.size_level_id == preset.default_size_level_id
    engine.apply(DrawingAction("set_brush_params", {"size_level_id": 7, "opacity_level_id": 8}))
    assert (engine.state.size_level_id, engine.state.opacity_level_id) == (7, 8)
    engine.apply(DrawingAction("select_brush", {"brush_id": "ink_fine"}))
    assert engine.state.size_level_id == default_catalog().get("ink_fine").default_size_level_id


def test_hidden_layer_cannot_be_written_and_v2_rejects_finished_actions():
    engine = base_engine()
    engine.apply(DrawingAction("set_layer_visibility", {"layer_id": "w", "visible": False}))
    with pytest.raises(DrawingError, match="hidden"):
        engine.apply(DrawingAction("stroke_v2", {"samples": [GOOD]}))
    engine.apply(DrawingAction("finish", {}))
    with pytest.raises(DrawingError, match="finished"):
        engine.apply(DrawingAction("select_color", {"color_id": 1}))


def test_stamp_budget_rejects_before_changing_pixels():
    engine = Engine(1024, 1024, profile=RENDERER_V2)
    for kind, payload in (
        ("create_layer", {"layer_id": "w", "name": "w", "role": "rough"}),
        ("select_layer", {"layer_id": "w"}),
        ("select_brush", {"brush_id": "ink_fine"}),
        ("select_color", {"color_id": 0}),
    ):
        engine.apply(DrawingAction(kind, payload))
    far = samples([(0, 0), (1023 * 256, 1023 * 256)] * 2048, [10] * 4096)
    before = engine.state.to_dict(include_pixels=True)
    with pytest.raises(DrawingError):
        engine.apply(DrawingAction("stroke_v2", {"samples": far}))
    assert engine.state.to_dict(include_pixels=True) == before


if __name__ == "__main__" and "--regenerate" in sys.argv:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(build_fixture(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXTURE}")
