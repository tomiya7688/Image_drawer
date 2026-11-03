"""制作工程v2の固定順序・スキップ制約を独立に検査する設計契約。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1] / "docs" / "developers" / "drawing_stage_workflow_v2.json"
)


def _spec() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _stage_indices() -> dict[str, int]:
    return {stage["id"]: i for i, stage in enumerate(_spec()["stages"])}


def _can_advance(stage_id: str, stroke_count: int, dedicated: bool) -> bool:
    current = _spec()["stages"][_stage_indices()[stage_id]]
    return stroke_count >= current["min_strokes"] and (
        not current["dedicated_layer"] or dedicated
    )


def _can_skip(stage_id: str, reason: str) -> bool:
    current = _spec()["stages"][_stage_indices()[stage_id]]
    return not current["required"] and reason in current.get("skip_reason_codes", [])


def test_stages_are_fixed_and_do_not_repeat():
    spec = _spec()
    ids = [stage["id"] for stage in spec["stages"]]
    assert spec["schema_version"] == "idraw-stage-workflow/v2"
    assert spec["workflow_id"] == "illustration-standard-v2"
    assert ids == [
        "rough",
        "sketch",
        "final_drawing",
        "base_color",
        "shading",
        "refine",
    ]
    assert len(ids) == len(set(ids))
    assert spec["initial_stage"] == "rough"
    assert spec["terminal"] == "finish"
    assert spec["forbid_stage_reordering"] is True
    assert spec["forbid_backwards_transition"] is True


@pytest.mark.parametrize("stage_id", ["rough", "sketch", "final_drawing", "refine"])
def test_mandatory_stage_cannot_be_skipped(stage_id: str):
    assert _spec()["stages"][_stage_indices()[stage_id]]["required"] is True
    assert not _can_skip(stage_id, "no_color_required")


@pytest.mark.parametrize(
    ("stage_id", "reason"),
    [("base_color", "no_color_required"), ("shading", "flat_style_requested")],
)
def test_only_optional_stages_can_skip_with_explicit_reason(stage_id: str, reason: str):
    assert _can_skip(stage_id, reason)
    assert not _can_skip(stage_id, "")
    assert not _can_skip(stage_id, "unrelated")


@pytest.mark.parametrize("stage_id", ["rough", "sketch", "final_drawing"])
def test_early_stages_require_real_stroke_on_separate_layer(stage_id: str):
    assert not _can_advance(stage_id, 0, True)
    assert not _can_advance(stage_id, 1, False)
    assert _can_advance(stage_id, 1, True)


def test_finish_requires_refine_to_have_paint_stroke():
    assert not _can_advance("refine", 0, False)
    assert _can_advance("refine", 1, False)


def test_next_stage_is_fixed_and_cannot_jump():
    ids = [s["id"] for s in _spec()["stages"]]
    index = _stage_indices()
    assert ids[index["rough"] + 1] == "sketch"
    assert ids[index["sketch"] + 1] == "final_drawing"
    assert ids[index["final_drawing"] + 1] == "base_color"
    assert ids[index["base_color"] + 1] == "shading"
    assert ids[index["shading"] + 1] == "refine"
