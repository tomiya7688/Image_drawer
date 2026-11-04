"""工程Controllerのテストが共有する入力生成。Controllerの動作は再実装しない。"""

from __future__ import annotations

from image_drawer.drawing import DrawingAction, Engine

FULL_COVER_LEVEL = 18  # size_px=256。小さな画布全体を不透明に塗る
STAGE_LAYER = {
    "rough": "sys_rough_001",
    "sketch": "sys_sketch_001",
    "final_drawing": "sys_final_drawing_001",
    "base_color": "sys_base_color_001",
    "shading": "sys_shade_001",
}


def model(type_: str, payload: dict | None = None, time_ms: int | None = None) -> DrawingAction:
    """modelの提案Action。"""
    return DrawingAction(type_, payload or {}, "model", None, time_ms)


def sample(x: int, y: int, index: int = 0) -> dict:
    """画素座標(x, y)のサンプル1点（筆圧最大）。"""
    return {
        "x_q8": x * 256, "y_q8": y * 256, "time_us": index * 1000, "pressure_level_id": 10,
        "tilt_x_level_id": None, "tilt_y_level_id": None,
    }  # fmt: skip


def line_samples(start: tuple[int, int], end: tuple[int, int]) -> list[dict]:
    return [sample(*start, 0), sample(*end, 1)]


def stroke_v2(start: tuple[int, int] = (2, 2), end: tuple[int, int] = (9, 6)) -> DrawingAction:
    return model("stroke_v2", {"samples": line_samples(start, end)})


def erase_v2(start: tuple[int, int] = (2, 2), end: tuple[int, int] = (9, 6)) -> DrawingAction:
    return model("erase_v2", {"samples": line_samples(start, end)})


def setup_actions(layer_id: str, brush: str = "ink_gpen", color: int = 0) -> list[DrawingAction]:
    """レイヤー選択・ブラシ・色を整えるActionの列。"""
    return [
        model("select_layer", {"layer_id": layer_id}),
        model("select_brush", {"brush_id": brush}),
        model("select_color", {"color_id": color}),
    ]


def full_cover_actions(layer_id: str, color: int = 0) -> list[DrawingAction]:
    """画布全体を不透明に塗るActionの列。同じ列を繰り返すと画素が変わらない（no-op）。"""
    return [
        *setup_actions(layer_id, "ink_gpen", color),
        model(
            "set_brush_params",
            {"size_level_id": FULL_COVER_LEVEL, "opacity_level_id": 10},
        ),
        model("stroke_v2", {"samples": [sample(8, 8)]}),
    ]


def advance(review_reason: str | None = None) -> DrawingAction:
    return model("advance_stage", {} if review_reason is None else {"review_reason": review_reason})


def pixels_hex(engine: Engine) -> str:
    """全レイヤーの画素を連結した16進文字列（比較・ハッシュ用）。"""
    return "|".join(layer.pixels.data.hex() for layer in engine.state.layers)
