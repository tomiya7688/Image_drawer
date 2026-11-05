"""S1b-5: レビュー退出（理由コード・StyleIntent・描画済みの禁止）。"""

from __future__ import annotations

import pytest
from stage_helpers import (
    REFINE_LAYER,
    accept,
    accept_all,
    advance,
    advance_to,
    draw_actions,
    engine_of,
    full_erase_actions,
    make_controller,
    model,
    noop_stroke_actions,
)

MONO = "白黒の線画で描く"
FLAT = "影なしでベタ塗り"
BOTH = "白黒の線画、影なし"


def code(result) -> str:
    return result.reason.split(":")[0]


def at_stage(prompt: str, stage_id: str):
    controller = make_controller(prompt)
    advance_to(controller, stage_id)
    return controller


def test_review_base_color_is_accepted_when_monochrome_intent_true():
    controller = at_stage(MONO, "base_color")
    result = accept(controller, advance("monochrome_line_art_only"))
    assert controller.stage_id == "shading" and result.events[0].payload == {
        "review_reason": "monochrome_line_art_only"
    }


def test_review_base_color_is_rejected_when_intent_false():
    controller = at_stage("猫を描く", "base_color")
    result = controller.propose(advance("monochrome_line_art_only"))
    assert not result.accepted and code(result) == "review_intent_not_confirmed"
    assert controller.stage_id == "base_color"


def test_review_shading_is_accepted_when_flat_intent_true():
    controller = at_stage(FLAT, "shading")
    accept(controller, advance("flat_style_no_shading"))
    assert controller.stage_id == "refine"


def test_review_shading_is_rejected_when_intent_false():
    controller = at_stage(MONO, "shading")
    result = controller.propose(advance("flat_style_no_shading"))
    assert code(result) == "review_intent_not_confirmed"


def test_review_refine_no_correction_needed_is_accepted():
    controller = at_stage("猫", "refine")
    accept(controller, advance("no_correction_needed"))
    assert controller.finished and controller.stage_id == "finish"


def test_review_reason_of_other_stage_is_rejected():
    controller = at_stage(BOTH, "base_color")
    for reason in ("flat_style_no_shading", "no_correction_needed"):
        assert code(controller.propose(advance(reason))) == "review_reason_not_allowed"
    shading = at_stage(BOTH, "shading")
    for reason in ("monochrome_line_art_only", "no_correction_needed"):
        assert code(shading.propose(advance(reason))) == "review_reason_not_allowed"


@pytest.mark.parametrize("stage_id", ["rough", "sketch", "final_drawing"])
def test_review_is_rejected_in_rough_sketch_final_drawing(stage_id):
    controller = at_stage(BOTH, stage_id)
    for reason in ("monochrome_line_art_only", "flat_style_no_shading", "no_correction_needed"):
        assert code(controller.propose(advance(reason))) == "review_reason_not_allowed"
    assert controller.stage_id == stage_id


def test_review_is_rejected_after_effective_stroke_in_same_stage():
    controller = at_stage(MONO, "base_color")
    accept_all(controller, draw_actions("base_color"))
    assert code(controller.propose(advance("monochrome_line_art_only"))) == "review_after_drawing"
    accept(controller, advance())


def test_review_is_rejected_after_effective_erase_in_same_stage():
    controller = at_stage("猫", "refine")
    accept_all(controller, full_erase_actions(REFINE_LAYER))
    assert code(controller.propose(advance("no_correction_needed"))) == "review_after_drawing"
    accept(controller, advance())


def test_review_after_noop_stroke_is_accepted():
    controller = at_stage(MONO, "base_color")
    accept(controller, model("select_layer", {"layer_id": "sys_base_color_001"}))
    accept_all(controller, noop_stroke_actions())
    assert not any(engine_of(controller).state.find_layer("sys_base_color_001").pixels.data[3::4])
    accept(controller, advance("monochrome_line_art_only"))
    assert controller.stage_id == "shading"


@pytest.mark.parametrize("reason", [None, 3, "", "nope", ["no_correction_needed"], True])
def test_review_with_unknown_null_or_non_string_reason_is_rejected(reason):
    controller = at_stage("猫", "refine")
    result = controller.propose(model("advance_stage", {"review_reason": reason}))
    assert not result.accepted and code(result) == "review_reason_not_allowed"


def test_review_still_records_exit_and_enter_events():
    controller = at_stage(MONO, "base_color")
    result = accept(controller, advance("monochrome_line_art_only"))
    assert [e.type for e in result.events] == [
        "advance_stage", "exit_stage", "enter_stage", "create_layer", "clear_active_layer",
    ]  # fmt: skip
    assert result.events[1].payload == {"stage_id": "base_color"}
    assert result.events[2].payload == {"stage_id": "shading"}


def test_review_exit_creates_next_stage_default_layer():
    controller = at_stage(MONO, "base_color")
    accept(controller, advance("monochrome_line_art_only"))
    assert "sys_shade_001" in [layer.id for layer in engine_of(controller).state.layers]


def test_refine_empty_normal_advance_is_stage_not_ready():
    controller = at_stage("猫", "refine")
    result = controller.propose(advance())
    assert not result.accepted and code(result) == "stage_not_ready"


def test_refine_advance_after_effective_erase_is_accepted():
    controller = at_stage("猫", "refine")
    accept_all(controller, full_erase_actions(REFINE_LAYER))
    accept(controller, advance())
    assert controller.finished


@pytest.mark.parametrize("stage_id", ["rough", "sketch", "final_drawing"])
def test_intent_true_does_not_waive_effective_stroke_requirement_in_rough(stage_id):
    controller = at_stage(BOTH, stage_id)
    assert code(controller.propose(advance())) == "stage_not_ready"


def test_all_three_reviews_in_one_run_reach_finish():
    controller = make_controller(BOTH)
    for stage_id in ("rough", "sketch", "final_drawing"):
        accept_all(controller, draw_actions(stage_id))
        accept(controller, advance())
    accept(controller, advance("monochrome_line_art_only"))
    accept(controller, advance("flat_style_no_shading"))
    accept(controller, advance("no_correction_needed"))
    assert controller.finished and controller.termination_reason == "finished"
    assert controller.accepted_action_count == 3 * 5 + 3
