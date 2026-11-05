"""S1c-1: Episode v2のJSON Schema（Draft 2020-12）と製品コードの契約が一致することを確認する。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from episode_helpers import full_raw

jsonschema = pytest.importorskip("jsonschema")

from image_drawer.drawing.episode import (  # noqa: E402
    STYLE_INTENT_KEYS,
    V1_KEYS,
    V2_EXTRA_KEYS,
    V2_WORKFLOW_KEYS,
)
from image_drawer.drawing.models import MAX_EVENTS, REVIEW_REASONS, STAGE_IDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs" / "developers" / "schemas" / "drawing_episode_v2.schema.json"
FIXTURES = Path(__file__).parent / "fixtures" / "drawing_episode_v2"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
SAMPLE = {
    "x_q8": 256, "y_q8": 256, "time_us": 0, "pressure_level_id": 10, "tilt_x_level_id": None,
    "tilt_y_level_id": None,
}  # fmt: skip


def valid(raw: dict) -> bool:
    return VALIDATOR.is_valid(raw)


def event_raw(type_: str, actor: str, payload: dict, stage: str = "rough") -> dict:
    return {
        "seq": 0, "time_ms": 0, "actor": actor, "type": type_, "payload": payload,
        "stage_id": stage, "group": {"id": 0, "index": 0, "size": 1},
    }  # fmt: skip


EXAMPLES = [
    event_raw("create_layer", "model", {"layer_id": "a", "name": "x", "role": "rough"}),
    event_raw("create_layer", "system", {"layer_id": "sys_a_001", "name": "x", "role": "rough"}),
    event_raw("select_layer", "model", {"layer_id": "a"}),
    event_raw("select_color", "model", {"color_id": 255}),
    event_raw("select_brush", "model", {"brush_id": "ink_gpen"}),
    event_raw("set_brush_params", "model", {"size_level_id": 18, "opacity_level_id": 10}),
    event_raw("stroke_v2", "model", {"samples": [SAMPLE]}),
    event_raw("erase_v2", "model", {"samples": [SAMPLE]}),
    event_raw("advance_stage", "model", {}),
    event_raw("advance_stage", "model", {"review_reason": "no_correction_needed"}),
    event_raw("set_layer_visibility", "system", {"layer_id": "a", "visible": False}),
    event_raw("enter_stage", "system", {"stage_id": "sketch"}, "sketch"),
    event_raw("exit_stage", "system", {"stage_id": "rough"}),
    event_raw("clear_active_layer", "system", {}),
    event_raw("finish", "system", {}, "finish"),
]


def with_events(events: list[dict]) -> dict:
    raw = full_raw()
    raw["events"] = events
    return raw


def test_schema_is_valid_draft_2020_12():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_schema_accepts_every_event_type_example():
    assert len({item["type"] for item in EXAMPLES}) == 13
    for example in EXAMPLES:
        assert valid(with_events([example])), example["type"]


def test_schema_accepts_controller_generated_episode():
    assert valid(full_raw())


def test_schema_rejects_v1_schema_version():
    raw = full_raw()
    raw["schema_version"] = "drawing-episode/v1"
    assert not valid(raw)


def test_schema_rejects_event_without_stage_id_or_group():
    for key in ("stage_id", "group"):
        raw = full_raw()
        del raw["events"][0][key]
        assert not valid(raw)


def test_schema_rejects_unknown_event_type_and_extra_keys():
    unknown = copy.deepcopy(EXAMPLES[2])
    unknown["type"] = "skip_stage"
    extra = copy.deepcopy(EXAMPLES[2])
    extra["note"] = 1
    assert not valid(with_events([unknown]))
    assert not valid(with_events([extra]))


def test_schema_rejects_controller_only_event_with_model_actor():
    for example in EXAMPLES:
        if example["type"] in ("enter_stage", "exit_stage", "clear_active_layer", "finish"):
            assert not valid(with_events([{**example, "actor": "model"}]))


def test_schema_rejects_model_only_event_with_system_actor():
    for example in EXAMPLES:
        if example["type"] in ("select_layer", "stroke_v2", "advance_stage", "select_color"):
            assert not valid(with_events([{**example, "actor": "system"}]))


def test_schema_rejects_complete_status_with_non_finished_reason():
    raw = full_raw()
    raw["termination_reason"] = "in_progress"
    assert not valid(raw)


def test_schema_rejects_draft_status_with_finished_reason():
    raw = full_raw()
    raw["status"] = "draft"
    assert not valid(raw)


def test_schema_rejects_float_and_bool_in_integer_fields():
    for value in (1.5, True):
        raw = full_raw()
        raw["seed"] = value
        assert not valid(raw)
        raw = full_raw()
        raw["events"][0]["seq"] = value
        assert not valid(raw)


def test_schema_group_bounds():
    base = EXAMPLES[2]
    for group, expected in (
        ({"id": 0, "index": 63, "size": 64}, True),
        ({"id": 0, "index": 0, "size": 65}, False),
        ({"id": 0, "index": 64, "size": 64}, False),
        ({"id": 0, "index": 0, "size": 0}, False),
        ({"id": -1, "index": 0, "size": 1}, False),
    ):
        assert valid(with_events([{**base, "group": group}])) is expected, group


def test_schema_review_reason_enum_has_exactly_three_codes():
    assert SCHEMA["$defs"]["review_reason"]["enum"] == list(REVIEW_REASONS)
    assert len(REVIEW_REASONS) == 3


def test_schema_stage_enum_has_no_skip_stage():
    assert SCHEMA["$defs"]["stage_id"]["enum"] == list(STAGE_IDS)
    assert SCHEMA["$defs"]["stage_id_after"]["enum"] == [*STAGE_IDS, "finish"]


def test_python_validator_and_schema_agree_on_header_key_set():
    expected = V1_KEYS | V2_EXTRA_KEYS | V2_WORKFLOW_KEYS
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"]) == expected
    assert set(SCHEMA["properties"]["style_intent"]["required"]) == STYLE_INTENT_KEYS


def test_schema_events_max_items_equals_engine_limit():
    assert SCHEMA["properties"]["events"]["maxItems"] == MAX_EVENTS


def test_schema_accepts_all_v2_fixtures():
    files = sorted(FIXTURES.glob("*.episode.json"))
    assert len(files) >= 3
    for path in files:
        assert valid(json.loads(path.read_text(encoding="utf-8"))), path.name
