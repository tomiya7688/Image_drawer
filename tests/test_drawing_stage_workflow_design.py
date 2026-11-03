"""制作工程v2の固定順序・操作許可・レイヤー役割を確認する静的契約テスト。

実行時のStageController/Rendererの実装済み判定ではない。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1] / "docs" / "developers" / "drawing_stage_workflow_v2.json"
)

EXPECTED = [
    "rough",
    "sketch",
    "final_drawing",
    "base_color",
    "shading",
    "refine",
]


def _spec() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _stage(stage_id: str) -> dict:
    return next(stage for stage in _spec()["stages"] if stage["id"] == stage_id)


def _can_advance(
    stage_id: str,
    stroke_count: int,
    write_role: str | None,
    review_reason: str | None = None,
    pixels_remain: bool = True,
) -> bool:
    """既知の合格条件だけを検査する独立の簡易オラクル。"""
    stage = _stage(stage_id)
    if review_reason is not None:
        return (
            stroke_count == 0
            and review_reason in stage["review_only_reason_codes"]
        )
    if stroke_count == 0:
        return False
    return (
        write_role in stage["write_layer_roles"]
        and stroke_count >= stage["min_effectful_strokes"]
        and pixels_remain
    )


def test_stages_are_fixed_and_not_skippable():
    spec = _spec()
    ids = [stage["id"] for stage in spec["stages"]]
    assert ids == EXPECTED
    assert len(ids) == len(set(ids))
    assert spec["schema_version"] == "idraw-stage-workflow/v2"
    assert spec["workflow_id"] == "illustration-standard-v2"
    assert spec["initial_stage"] == "rough"
    assert spec["terminal"] == "finish"
    assert spec["allow_skip_stage"] is False
    assert spec["forbid_stage_reordering"] is True
    assert spec["forbid_backwards_transition"] is True
    assert spec["terminal_after"] == "refine"
    assert all(stage["required"] for stage in spec["stages"])


def test_model_and_controller_actions_do_not_overlap():
    spec = _spec()
    model = set(spec["allowed_model_actions"])
    controller = set(spec["controller_only_actions"])
    forbidden = set(spec["forbidden_action_types"])
    assert not model & controller
    assert not model & forbidden
    assert {"advance_stage", "stroke_v2", "erase_v2"} <= model
    assert {"enter_stage", "set_layer_visibility", "finish"} <= controller
    assert {"skip_stage", "set_phase", "finish"} <= forbidden


@pytest.mark.parametrize("stage_id", ["rough", "sketch", "final_drawing"])
def test_early_stages_require_real_strokes_on_own_layer(stage_id: str):
    stage = _stage(stage_id)
    role = stage["layer_roles"][0]
    assert stage["min_effectful_strokes"] == 1
    assert stage["review_only_reason_codes"] == []
    assert not _can_advance(stage_id, 0, role)
    assert not _can_advance(stage_id, 1, "base_color")
    assert not _can_advance(stage_id, 1, role, pixels_remain=False)
    assert _can_advance(stage_id, 1, role)
    assert not _can_advance(stage_id, 0, None, "no_correction_needed")


@pytest.mark.parametrize(
    ("stage_id", "code"),
    [
        ("base_color", "monochrome_line_art_only"),
        ("shading", "flat_style_no_shading"),
        ("refine", "no_correction_needed"),
    ],
)
def test_review_only_is_not_skipping_a_stage(stage_id: str, code: str):
    stage = _stage(stage_id)
    assert stage["required"] is True
    assert _can_advance(stage_id, 0, None, code)
    assert not _can_advance(stage_id, 0, None)
    assert not _can_advance(stage_id, 0, None, "unrelated_reason")


def test_refine_can_change_prior_visible_layers():
    stage = _stage("refine")
    assert set(stage["write_layer_roles"]) == {
        "base_color",
        "shade",
        "final_drawing",
        "highlight",
        "correction",
    }
    assert _can_advance("refine", 1, "final_drawing")
    assert not _can_advance("refine", 1, "rough")


@pytest.mark.parametrize("stage_id", EXPECTED)
def test_create_layer_roles_are_stage_bound(stage_id: str):
    stage = _stage(stage_id)
    assert set(stage["layer_roles"]) <= set(stage["write_layer_roles"])
    assert set(stage["layer_roles"]).isdisjoint({"canvas", "background"})
    assert "rough" not in stage["write_layer_roles"] or stage_id == "rough"


def test_color_layers_do_not_hide_final_drawing():
    spec = _spec()
    z = spec["layer_z_order_bottom_to_top"]
    assert len(z) == len(set(z))
    assert z.index("base_color") < z.index("final_drawing")
    assert z.index("shade") < z.index("final_drawing")
    assert z.index("final_drawing") < z.index("highlight")
    assert z.index("highlight") < z.index("correction")
    assert spec["layer_visibility_transitions"] == [
        {"on_enter_stage": "final_drawing", "hide_layer_roles": ["rough"]},
        {"on_enter_stage": "base_color", "hide_layer_roles": ["sketch"]},
    ]


def test_stage_transition_cannot_jump_or_move_backwards():
    ids = EXPECTED + [_spec()["terminal"]]
    for i, current in enumerate(EXPECTED):
        assert ids[i] == current
        assert ids[i + 1] != current
        assert ids[i + 1] == (EXPECTED[i + 1] if i < len(EXPECTED) - 1 else "finish")


def test_image_target_is_never_policy_observation():
    spec = _spec()
    assert spec["target_image_policy"] == "evaluator_only_never_policy_observation"
    assert spec["workflow_is_not_human_traces"] is True
    assert spec["legacy_v1_phase_mode"] == "legacy golden replay only"
