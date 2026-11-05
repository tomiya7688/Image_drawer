"""S1b-2: StageTable（Stage JSONの複製・固定順・Engineとの整合）。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from image_drawer.drawing.catalog import DATA_DIR, canonical_sha256
from image_drawer.drawing.models import LAYER_ROLES_Z_ORDER, STAGE_IDS
from image_drawer.drawing.stage import StageTableError, default_stage_table, parse_stage_table

DOCS_JSON = Path(__file__).resolve().parents[1] / "docs/developers/drawing_stage_workflow_v2.json"


def document() -> dict:
    return json.loads(DOCS_JSON.read_text(encoding="utf-8"))


def test_stage_table_has_six_stages_in_fixed_order():
    table = default_stage_table()
    assert tuple(spec.id for spec in table.stages) == STAGE_IDS
    assert table.workflow_id == "illustration-standard-v2"
    assert table.spec("shading").write_layer_roles == ("shade", "highlight")
    assert table.spec("refine").min_effectful_strokes == 0
    assert table.entry_default_roles["refine"] is None
    assert table.hide_on_enter == {"final_drawing": ("rough",), "base_color": ("sketch",)}
    assert table.next_stage_id("rough") == "sketch" and table.next_stage_id("refine") is None


def test_stage_table_package_copy_is_byte_identical_to_docs_json():
    packaged = DATA_DIR / "drawing_stage_workflow_v2.json"
    assert packaged.read_bytes() == DOCS_JSON.read_bytes()


def test_stage_table_rejects_reordered_stages():
    doc = document()
    doc["stages"][0], doc["stages"][1] = doc["stages"][1], doc["stages"][0]
    with pytest.raises(StageTableError):
        parse_stage_table(doc)


def test_stage_table_rejects_role_missing_from_engine_z_order():
    doc = document()
    doc["stages"][0]["layer_roles"] = ["no_such_role"]
    with pytest.raises(StageTableError):
        parse_stage_table(doc)
    doc = document()
    doc["layer_z_order_bottom_to_top"] = list(reversed(LAYER_ROLES_Z_ORDER))
    with pytest.raises(StageTableError):
        parse_stage_table(doc)


def test_stage_table_workflow_sha256_uses_canonical_json():
    doc = document()
    expected = canonical_sha256(doc)
    assert default_stage_table().workflow_sha256 == expected
    # 整形（キー順・空白）が違っても、同じ内容なら同じSHA-256になる。
    shuffled = json.loads(json.dumps(doc, sort_keys=True, indent=7))
    assert parse_stage_table(shuffled).workflow_sha256 == expected
    changed = copy.deepcopy(doc)
    changed["description"] += "x"
    assert parse_stage_table(changed).workflow_sha256 != expected
