"""S1b-4: グループ生成と提案審査（Controllerなしで拒否コードを検証する純関数テスト）。"""

from __future__ import annotations

import pytest
from stage_helpers import advance, model, setup_actions, stroke_v2

from image_drawer.drawing import Engine
from image_drawer.drawing.stage import default_stage_table, judge_style_intent
from image_drawer.drawing.stage.groups import (
    build_initial_group,
    build_transition_group,
    system_layer_id,
    with_group,
)
from image_drawer.drawing.stage.progress import StageProgress, has_visible_paint
from image_drawer.drawing.stage.rules import check_proposal, remaining_system_layers

TABLE = default_stage_table()
MONO_FLAT = judge_style_intent("白黒の線画、影なし")
NEUTRAL = judge_style_intent("猫")


class Rig:
    """Engineに初期グループを適用した、審査関数の入力一式。"""

    def __init__(self, width: int = 16, height: int = 16) -> None:
        self.engine = Engine(width, height)
        self.engine.apply_group(build_initial_group(TABLE, 0))
        self.progress = StageProgress()

    def check(self, action, intent=NEUTRAL):
        return check_proposal(TABLE, self.engine.state, self.progress, intent, action)

    def paint(self, layer_id: str = "sys_rough_001") -> None:
        """有効な筆跡を1つ描いて計数する。"""
        for action in (*setup_actions(layer_id), stroke_v2()):
            self.engine.apply(action)
        self.progress.record("stroke_v2")

    def move_to(self, stage_id: str, intent=NEUTRAL) -> None:
        """遷移グループを適用して指定工程へ進める（筆跡は描かない）。"""
        while self.engine.state.stage_id != stage_id:
            events = build_transition_group(TABLE, self.engine.state, advance())
            self.engine.apply_group(with_group(events, 99, 0))
            self.progress.reset()


def code(reason: str | None) -> str | None:
    return None if reason is None else reason.split(":")[0]


def test_initial_group_events_and_names():
    events = build_initial_group(TABLE, 0)
    assert [e.type for e in events] == ["enter_stage", "create_layer", "clear_active_layer"]
    assert events[1].payload == {"layer_id": "sys_rough_001", "name": "ラフ", "role": "rough"}
    assert [(e.group.id, e.group.index, e.group.size) for e in events] == [
        (0, 0, 3),
        (0, 1, 3),
        (0, 2, 3),
    ]
    assert {e.actor for e in events} == {"system"} and {e.time_ms for e in events} == {0}


def test_transition_group_does_not_mutate_state_and_hides_visible_layers_only():
    rig = Rig()
    rig.move_to("sketch")
    before = rig.engine.state.to_dict(include_pixels=True)
    events = build_transition_group(TABLE, rig.engine.state, advance())
    assert rig.engine.state.to_dict(include_pixels=True) == before
    assert [e.type for e in events] == [
        "advance_stage", "exit_stage", "set_layer_visibility", "enter_stage",
        "create_layer", "clear_active_layer",
    ]  # fmt: skip
    assert events[2].payload == {"layer_id": "sys_rough_001", "visible": False}
    assert [e.stage_id for e in events] == [
        "sketch", "sketch", "sketch", "final_drawing", "final_drawing", "final_drawing",
    ]  # fmt: skip
    rig.engine.state.find_layer("sys_rough_001").visible = False
    again = build_transition_group(TABLE, rig.engine.state, advance())
    assert "set_layer_visibility" not in [e.type for e in again]


def test_system_layer_id_counts_existing_prefixed_layers():
    rig = Rig()
    assert system_layer_id(rig.engine.state, "rough") == "sys_rough_002"
    assert system_layer_id(rig.engine.state, "sketch") == "sys_sketch_001"


def test_remaining_system_layers_counts_later_default_layers():
    assert remaining_system_layers(TABLE, "rough") == 4
    assert remaining_system_layers(TABLE, "base_color") == 1
    assert remaining_system_layers(TABLE, "shading") == 0
    assert remaining_system_layers(TABLE, "refine") == 0


@pytest.mark.parametrize(
    "kind", ["skip_stage", "set_phase", "finish", "enter_stage", "exit_stage",
             "set_layer_visibility", "clear_active_layer", "stroke", "erase", "mystery"],
)  # fmt: skip
def test_action_not_allowed(kind):
    assert code(Rig().check(model(kind))) == "action_not_allowed"


def test_invalid_payload_for_advance():
    rig = Rig()
    assert code(rig.check(model("advance_stage", {"to": "finish"}))) == "invalid_payload"
    assert (
        code(rig.check(model("advance_stage", {"review_reason": "x", "to": "a"})))
        == "invalid_payload"
    )


def test_reserved_layer_id_and_role_not_creatable():
    rig = Rig()
    reserved = model("create_layer", {"layer_id": "sys_rough_002", "name": "a", "role": "rough"})
    assert code(rig.check(reserved)) == "reserved_layer_id"
    wrong_role = model("create_layer", {"layer_id": "mine", "name": "a", "role": "sketch"})
    assert code(rig.check(wrong_role)) == "role_not_creatable"
    ok = model("create_layer", {"layer_id": "mine", "name": "a", "role": "rough"})
    assert rig.check(ok) is None


def test_layer_capacity_counts_future_system_layers():
    rig = Rig()
    for index in range(10):  # 既存1 + 10 + 将来4 = 15
        add = model("create_layer", {"layer_id": f"m{index}", "name": "a", "role": "rough"})
        assert rig.check(add) is None
        rig.engine.apply(add)
    last = model("create_layer", {"layer_id": "m10", "name": "a", "role": "rough"})
    assert rig.check(last) is None  # 12 + 4 = 16
    rig.engine.apply(last)
    over = model("create_layer", {"layer_id": "m11", "name": "a", "role": "rough"})
    assert code(rig.check(over)) == "layer_capacity"


def test_layer_capacity_by_area_budget():
    rig = Rig(1024, 1024)  # 1024x1024では必須5枚も保持できないため、追加は常に拒否される
    extra = model("create_layer", {"layer_id": "m0", "name": "a", "role": "rough"})
    assert code(rig.check(extra)) == "layer_capacity"


def test_select_layer_rules():
    rig = Rig()
    assert rig.check(model("select_layer", {"layer_id": "sys_rough_001"})) is None
    assert rig.check(model("select_layer", {"layer_id": "missing"})) is None  # Engineが拒否する
    rig.move_to("sketch")
    assert (
        code(rig.check(model("select_layer", {"layer_id": "sys_rough_001"})))
        == "layer_not_selectable"
    )
    rig.move_to("final_drawing")
    assert (
        code(rig.check(model("select_layer", {"layer_id": "sys_sketch_001"})))
        == "layer_not_selectable"
    )


def test_layer_not_writable_is_defensive():
    rig = Rig()
    rig.move_to("sketch")
    rig.engine.state.active_layer_id = "sys_rough_001"
    assert code(rig.check(stroke_v2())) == "layer_not_writable"
    assert code(rig.check(model("erase_v2", {"samples": []}))) == "layer_not_writable"


def test_stage_not_ready_codes():
    rig = Rig()
    assert code(rig.check(advance())) == "stage_not_ready"
    rig.paint()
    assert rig.check(advance()) is None
    # 筆跡を数えていても、可視の画素が残っていなければ退出できない。
    rig.engine.state.find_layer("sys_rough_001").pixels.data[:] = bytes(16 * 16 * 4)
    assert has_visible_paint(rig.engine.state, ("rough",)) is False
    assert code(rig.check(advance())) == "stage_not_ready"


def test_review_codes():
    rig = Rig()
    assert code(rig.check(advance("no_correction_needed"))) == "review_reason_not_allowed"
    for bad in (None, 3, "", ["x"]):
        assert (
            code(rig.check(model("advance_stage", {"review_reason": bad})))
            == "review_reason_not_allowed"
        )
    rig.move_to("base_color")
    assert code(rig.check(advance("monochrome_line_art_only"))) == "review_intent_not_confirmed"
    assert rig.check(advance("monochrome_line_art_only"), MONO_FLAT) is None
    assert (
        code(rig.check(advance("flat_style_no_shading"), MONO_FLAT)) == "review_reason_not_allowed"
    )
    rig.paint("sys_base_color_001")
    assert code(rig.check(advance("monochrome_line_art_only"), MONO_FLAT)) == "review_after_drawing"


def test_refine_review_and_normal_exit():
    rig = Rig()
    rig.move_to("refine")
    assert code(rig.check(advance())) == "stage_not_ready"
    assert rig.check(advance("no_correction_needed")) is None
    rig.progress.record("erase_v2")
    assert rig.check(advance()) is None
    assert code(rig.check(advance("no_correction_needed"))) == "review_after_drawing"
