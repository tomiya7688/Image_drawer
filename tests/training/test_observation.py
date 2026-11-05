"""Observationのtarget隔離の契約(設計書5.1・5.2節、8.1節)。"""

from __future__ import annotations

import dataclasses

import pytest

np = pytest.importorskip("numpy")

from image_drawer.drawing import Engine, DrawingAction  # noqa: E402
from image_drawer.drawing.stage import Budget, StageController, judge_style_intent  # noqa: E402
from image_drawer.training.observation import (  # noqa: E402
    OBSERVATION_KEYS,
    Observation,
    ObservationBuilder,
    classify_reject_reason,
)

FORBIDDEN_NAME_PARTS = ("target", "reward", "score", "diff", "goal")


def _controller(max_actions: int = 24) -> tuple[StageController, Engine, Budget]:
    engine = Engine(16, 16, seed=3)
    budget = Budget(max_actions)
    return StageController(engine, judge_style_intent("赤い四角"), budget), engine, budget


def test_observation_keys_match_contract() -> None:
    names = {field.name for field in dataclasses.fields(Observation)}
    assert names == OBSERVATION_KEYS


def test_observation_has_no_target_like_field_names() -> None:
    for field in dataclasses.fields(Observation):
        assert not any(part in field.name.lower() for part in FORBIDDEN_NAME_PARTS)
    from image_drawer.training import observation

    for cls in (
        observation.LayerBrief,
        observation.ToolState,
        observation.BudgetView,
        observation.ActionBrief,
    ):
        for field in dataclasses.fields(cls):
            assert not any(part in field.name.lower() for part in FORBIDDEN_NAME_PARTS)


def test_observation_builder_reads_only_own_canvas_and_attempt_log() -> None:
    controller, engine, budget = _controller()
    builder = ObservationBuilder(tail_length=3, summary_side=4)
    text = np.ones(8, dtype=np.float32)
    first = builder.build(text, controller, engine, budget)
    assert first.stage_id == "rough"
    assert first.canvas_summary.shape == (3, 4, 4) and np.allclose(first.canvas_summary, 1.0)
    assert [layer.role for layer in first.layer_summary] == ["rough"]
    assert not first.layer_summary[0].is_active and first.action_tail == ()
    assert first.budget.remaining_actions_ratio == 1.0

    # 拒否は種別と粗い分類だけが観測に戻る。
    controller.propose(DrawingAction("advance_stage", {}))
    controller.propose(DrawingAction("select_layer", {"layer_id": "sys_rough_001"}))
    controller.propose(DrawingAction("select_color", {"color_id": 0}))
    controller.propose(DrawingAction("stroke_v2", {"samples": []}))
    second = builder.build(text, controller, engine, budget)
    assert [(a.kind, a.accepted) for a in second.action_tail] == [
        ("select_layer", True),
        ("select_color", True),
        ("stroke_v2", False),
    ]
    assert second.action_tail[-1].reject_class in {"invalid_value", "no_layer", "other"}
    assert second.tool_state.layer_selected and second.tool_state.color_selected
    assert second.budget.remaining_actions_ratio == pytest.approx(1 - 2 / 24)


def test_reject_reasons_map_to_coarse_classes() -> None:
    assert classify_reject_reason("stage_not_ready: no stroke") == "illegal_stage"
    assert classify_reject_reason("layer_not_selectable: x") == "no_layer"
    assert classify_reject_reason("engine_rejected: bad") == "invalid_value"
    assert classify_reject_reason("layer_capacity: x") == "budget"
    assert classify_reject_reason("something_new") == "other"
