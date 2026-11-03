"""独自ブラシプリセットの静的契約を検証する。Renderer v2の画素精度テストではない。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "docs" / "developers" / "brush_presets_v2.json"
LEVELS = Path(__file__).resolve().parents[1] / "docs" / "developers" / "brush_control_levels_v2.json"

FIELDS = {
    "id",
    "name_ja",
    "mode",
    "tip",
    "family",
    "default_size_level_id",
    "default_opacity_level_id",
    "spacing_q16",
    "hardness_u16",
    "flow_u16",
    "grain_u16",
    "aspect_u16",
    "angle_cdeg",
    "pressure_min_size_u16",
    "pressure_min_opacity_u16",
}
Q16_FIELDS = {
    "spacing_q16",
    "hardness_u16",
    "flow_u16",
    "grain_u16",
    "aspect_u16",
    "pressure_min_size_u16",
    "pressure_min_opacity_u16",
}


def test_preset_catalog_is_fixed_and_independent_from_color():
    raw = json.loads(CATALOG.read_text(encoding="utf-8"))
    presets = raw["presets"]
    ids = [entry["id"] for entry in presets]
    assert raw["schema_version"] == "idraw-brush-presets/2"
    assert raw["brush_set_id"] == "brush-v2-core12"
    assert raw["palette_reference"] == "fixed-256-independent"
    assert len(presets) == len(set(ids)) == 12
    assert sum(p["mode"] == "stroke" for p in presets) == 10
    assert sum(p["mode"] == "erase" for p in presets) == 2
    assert {"round", "flat", "soft_round", "grain_round"} == {p["tip"] for p in presets}
    assert ids[-2:] == ["eraser_hard", "eraser_soft"]
    assert raw["adjustable_fields"] == ["size_level_id", "opacity_level_id"]
    assert raw["control_set_id"] == json.loads(LEVELS.read_text(encoding="utf-8"))["control_set_id"]
    assert all(set(p.keys()) == FIELDS for p in presets)

    for p in presets:
        assert p["mode"] in {"stroke", "erase"}
        assert p["id"].replace("_", "").isalnum()
        assert p["id"].startswith(tuple("abcdefghijklmnopqrstuvwxyz"))
        assert type(p["default_size_level_id"]) is int
        assert type(p["default_opacity_level_id"]) is int
        assert 0 <= p["default_size_level_id"] < 19
        assert 0 <= p["default_opacity_level_id"] < 11
        assert 0 <= p["angle_cdeg"] < 36000
        for key in Q16_FIELDS:
            assert type(p[key]) is int
            assert 0 <= p[key] <= 65535
        assert p["spacing_q16"] > 0
        assert p["flow_u16"] > 0
        assert p["aspect_u16"] > 0
        assert p["pressure_min_size_u16"] > 0
        assert p["pressure_min_opacity_u16"] > 0
        assert not any("color" in name or "rgb" in name for name in p)


def test_brush_catalog_hash_is_deterministic():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(canonical).hexdigest()
    reversed_keys = dict(reversed(list(data.items())))
    reordered = json.dumps(
        reversed_keys, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert canonical == reordered
    assert digest == hashlib.sha256(reordered).hexdigest()
    assert len(digest) == 64


def test_preset_names_and_indices_are_stable():
    presets = json.loads(CATALOG.read_text(encoding="utf-8"))["presets"]
    expected = [
        "pencil_graphite",
        "ink_gpen",
        "ink_fine",
        "marker_chisel",
        "paint_round_opaque",
        "paint_flat",
        "watercolor_soft",
        "airbrush_soft",
        "pastel_oil",
        "dry_texture",
        "eraser_hard",
        "eraser_soft",
    ]
    assert [p["id"] for p in presets] == expected

def test_temporary_brush_preset_ruff_diff():
    import subprocess

    result = subprocess.run(["ruff", "format", "--diff", __file__], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
