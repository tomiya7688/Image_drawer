"""S1b-5: レイヤー権限・自動作成・非表示・固定z順の合成。"""

from __future__ import annotations

import pytest
from stage_helpers import (
    REFINE_LAYER,
    STAGE_ORDER,
    accept,
    accept_all,
    advance,
    advance_to,
    draw_actions,
    engine_of,
    full_cover_actions,
    full_run,
    make_controller,
    model,
    stroke_v2,
)

from image_drawer.drawing.models import LAYER_ROLES_Z_ORDER
from image_drawer.drawing.palette import palette_rgb
from image_drawer.drawing.stage import default_stage_table


def code(result) -> str:
    return result.reason.split(":")[0]


def new_layer(layer_id: str, role: str) -> object:
    return model("create_layer", {"layer_id": layer_id, "name": layer_id, "role": role})


def visibility(controller) -> dict[str, bool]:
    return {layer.id: layer.visible for layer in engine_of(controller).state.layers}


def rgb_at(controller, x: int, y: int) -> tuple[int, int, int]:
    return tuple(engine_of(controller).render().pixel(x, y)[:3])


def test_select_layer_of_other_stage_role_is_rejected():
    controller = make_controller()
    advance_to(controller, "base_color")
    result = controller.propose(model("select_layer", {"layer_id": "sys_final_drawing_001"}))
    assert not result.accepted and code(result) == "layer_not_selectable"


def test_select_hidden_layer_is_rejected():
    controller = make_controller()
    advance_to(controller, "final_drawing")
    assert visibility(controller)["sys_rough_001"] is False
    result = controller.propose(model("select_layer", {"layer_id": "sys_rough_001"}))
    assert code(result) == "layer_not_selectable"


def test_select_unknown_layer_is_engine_rejected():
    result = make_controller().propose(model("select_layer", {"layer_id": "ghost"}))
    assert code(result) == "engine_rejected"


@pytest.mark.parametrize("stage_id", STAGE_ORDER)
def test_create_layer_role_must_belong_to_current_stage(stage_id):
    controller = make_controller()
    advance_to(controller, stage_id)
    allowed = default_stage_table().spec(stage_id).layer_roles
    for role in LAYER_ROLES_Z_ORDER:
        result = controller.propose(new_layer(f"probe_{role}", role))
        assert result.accepted == (role in allowed), (role, result.reason)
        if not result.accepted:
            assert code(result) == "role_not_creatable"


def test_shading_allows_shade_and_highlight_layers():
    controller = make_controller()
    advance_to(controller, "shading")
    accept(controller, new_layer("extra_shade", "shade"))
    accept(controller, new_layer("hl", "highlight"))
    accept(controller, model("select_layer", {"layer_id": "hl"}))
    accept(controller, model("select_layer", {"layer_id": "extra_shade"}))


def test_refine_create_layer_allows_only_correction():
    controller = make_controller()
    advance_to(controller, "refine")
    for role in ("base_color", "shade", "final_drawing", "highlight", "rough", "sketch"):
        assert code(controller.propose(new_layer(f"x_{role}", role))) == "role_not_creatable"
    accept(controller, new_layer("fix", "correction"))


def test_refine_can_select_visible_prior_roles():
    controller = make_controller()
    advance_to(controller, "shading")
    accept(controller, new_layer("hl", "highlight"))
    accept_all(controller, draw_actions("shading"))
    accept(controller, advance())
    accept(controller, new_layer("fix", "correction"))
    for layer_id in ("sys_base_color_001", "sys_shade_001", "sys_final_drawing_001", "hl", "fix"):
        accept(controller, model("select_layer", {"layer_id": layer_id}))


def test_refine_cannot_select_hidden_rough_or_sketch():
    controller = make_controller()
    advance_to(controller, "refine")
    for layer_id in ("sys_rough_001", "sys_sketch_001"):
        assert code(controller.propose(model("select_layer", {"layer_id": layer_id}))) == (
            "layer_not_selectable"
        )


def test_stroke_on_layer_with_unwritable_role_is_rejected():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    accept(controller, advance())
    state = engine_of(controller).state
    state.find_layer("sys_rough_001").visible = True  # 防御的検査: 状態を直接設定する
    state.active_layer_id = "sys_rough_001"
    result = controller.propose(stroke_v2())
    assert not result.accepted and code(result) == "layer_not_writable"


@pytest.mark.parametrize("layer_id", ["sys_x", "sys_rough_002", "sys_sketch_001"])
def test_model_layer_id_with_sys_prefix_is_rejected(layer_id):
    controller = make_controller()
    assert code(controller.propose(new_layer(layer_id, "rough"))) == "reserved_layer_id"


def test_model_layer_id_cannot_collide_with_future_system_layer():
    controller = make_controller()
    assert code(controller.propose(new_layer("sys_sketch_001", "rough"))) == "reserved_layer_id"
    accept_all(controller, draw_actions("rough"))
    accept(controller, advance())  # sys_sketch_001が衝突なく作られる
    assert "sys_sketch_001" in visibility(controller)


def test_create_layer_that_starves_mandatory_layers_is_rejected():
    controller = make_controller()
    accepted = 0
    while controller.propose(new_layer(f"m{accepted}", "rough")).accepted:
        accepted += 1
    # 既存1枚 + 追加11枚 + これから作る4枚 = 16枚
    assert accepted == 11 and len(engine_of(controller).state.layers) == 12
    assert code(controller.propose(new_layer("late", "rough"))) == "layer_capacity"
    full_run(controller)  # 必須レイヤーを作る余裕が残っているため最後まで完走できる
    assert controller.finished and len(engine_of(controller).state.layers) == 16


def test_create_layer_that_exceeds_area_budget_is_rejected():
    controller = make_controller(width=915, height=915)
    result = controller.propose(new_layer("m0", "rough"))
    assert not result.accepted and code(result) == "layer_capacity"


def test_system_layer_ids_follow_role_and_sequence():
    controller = make_controller()
    advance_to(controller, "refine")
    layers = engine_of(controller).state.layers
    assert [(layer.id, layer.role) for layer in layers] == [
        ("sys_rough_001", "rough"), ("sys_sketch_001", "sketch"),
        ("sys_final_drawing_001", "final_drawing"), ("sys_base_color_001", "base_color"),
        ("sys_shade_001", "shade"),
    ]  # fmt: skip
    assert [layer.name for layer in layers] == ["ラフ", "下書き", "本描き", "基本色", "陰影"]


def test_rough_layers_are_hidden_on_entering_final_drawing():
    controller = make_controller()
    advance_to(controller, "sketch")
    assert visibility(controller)["sys_rough_001"] is True
    advance_to(controller, "final_drawing")
    assert visibility(controller) == {
        "sys_rough_001": False, "sys_sketch_001": True, "sys_final_drawing_001": True,
    }  # fmt: skip


def test_sketch_layers_are_hidden_on_entering_base_color():
    controller = make_controller()
    advance_to(controller, "base_color")
    flags = visibility(controller)
    assert flags["sys_sketch_001"] is False and flags["sys_final_drawing_001"] is True
    assert flags["sys_base_color_001"] is True


def test_every_visible_layer_of_hidden_role_gets_one_visibility_event():
    controller = make_controller()
    accept(controller, new_layer("r2", "rough"))
    accept(controller, new_layer("r3", "rough"))
    accept_all(controller, draw_actions("rough"))
    accept(controller, advance())
    accept_all(controller, draw_actions("sketch"))
    result = accept(controller, advance())
    hidden = [e.payload for e in result.events if e.type == "set_layer_visibility"]
    assert hidden == [
        {"layer_id": "sys_rough_001", "visible": False},
        {"layer_id": "r2", "visible": False},
        {"layer_id": "r3", "visible": False},
    ]
    assert len(result.events) == 5 + 3
    assert {e.group.size for e in result.events} == {8}


def test_already_hidden_layers_get_no_visibility_event():
    controller = make_controller()
    accept_all(controller, draw_actions("rough"))
    accept(controller, advance())
    accept_all(controller, draw_actions("sketch"))
    engine_of(controller).state.find_layer("sys_rough_001").visible = False
    result = accept(controller, advance())
    assert "set_layer_visibility" not in [e.type for e in result.events]
    assert len(result.events) == 5


def test_hidden_layers_are_not_deleted():
    controller = make_controller()
    advance_to(controller, "refine")
    ids = [layer.id for layer in engine_of(controller).state.layers]
    assert "sys_rough_001" in ids and "sys_sketch_001" in ids
    assert any(engine_of(controller).state.find_layer("sys_rough_001").pixels.data[3::4])


def test_base_color_layer_does_not_cover_final_drawing_line_in_render():
    controller = make_controller()
    advance_to(controller, "final_drawing")
    accept_all(controller, draw_actions("final_drawing", color=0))
    accept(controller, advance())
    accept_all(controller, full_cover_actions("sys_base_color_001", color=5))
    assert rgb_at(controller, 2, 2) == palette_rgb(0)
    assert rgb_at(controller, 14, 14) == palette_rgb(5)


def test_shade_renders_below_and_highlight_above_final_drawing():
    controller = make_controller()
    advance_to(controller, "final_drawing")
    accept_all(controller, draw_actions("final_drawing", color=0))
    advance_to(controller, "shading")
    accept(controller, new_layer("hl", "highlight"))
    accept_all(controller, full_cover_actions("sys_shade_001", color=5))
    assert rgb_at(controller, 2, 2) == palette_rgb(0)  # shadeはfinal_drawingの下
    assert rgb_at(controller, 14, 14) == palette_rgb(5)
    accept_all(controller, full_cover_actions("hl", color=9))
    assert rgb_at(controller, 2, 2) == palette_rgb(9)  # highlightはfinal_drawingの上


def test_correction_layer_renders_on_top():
    controller = make_controller()
    advance_to(controller, "refine")
    accept(controller, new_layer("fix", "correction"))
    accept_all(controller, full_cover_actions(REFINE_LAYER, color=5))
    accept_all(controller, full_cover_actions("fix", color=9))
    assert rgb_at(controller, 2, 2) == palette_rgb(9)


def test_same_role_layers_stack_in_creation_order():
    controller = make_controller()
    accept(controller, new_layer("r2", "rough"))
    accept_all(controller, full_cover_actions("r2", color=9))
    accept_all(controller, full_cover_actions("sys_rough_001", color=5))
    assert rgb_at(controller, 8, 8) == palette_rgb(9)  # 後から作ったr2が上
