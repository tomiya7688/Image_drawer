"""工程Controllerのテストが共有する入力生成。Controllerの動作は再実装しない。"""

from __future__ import annotations

from image_drawer.drawing import DrawingAction, Engine
from image_drawer.drawing.stage import Budget, judge_style_intent
from image_drawer.drawing.stage.controller import ProposalResult, StageController

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


# --- Controller向けの共通入力（製品コードのControllerを通して使う） ---

STAGE_ORDER = ("rough", "sketch", "final_drawing", "base_color", "shading", "refine")
REFINE_LAYER = "sys_final_drawing_001"


def make_controller(
    prompt: str = "猫", width: int = 16, height: int = 16, max_actions: int = 1000,
    max_attempts: int | None = None, seed: int = 0,
) -> StageController:  # fmt: skip
    engine = Engine(width, height, seed=seed)
    return StageController(engine, judge_style_intent(prompt), Budget(max_actions, max_attempts))


def engine_of(controller: StageController) -> Engine:
    """Controllerが所有するEngine（状態の比較にだけ使う）。"""
    return controller._engine


def accept(controller: StageController, action: DrawingAction) -> ProposalResult:
    """受理されるはずの提案。拒否されたら理由つきで失敗する。"""
    result = controller.propose(action)
    assert result.accepted, result.reason
    return result


def accept_all(controller: StageController, actions: list[DrawingAction]) -> list[ProposalResult]:
    return [accept(controller, action) for action in actions]


def draw_actions(stage_id: str, color: int = 0) -> list[DrawingAction]:
    """工程の書込先に有効な筆跡を1本描くActionの列（退出は含まない）。"""
    layer_id = STAGE_LAYER.get(stage_id, REFINE_LAYER)
    return [*setup_actions(layer_id, "ink_gpen", color), stroke_v2()]


def draw_and_advance(controller: StageController, stage_id: str, color: int = 0) -> ProposalResult:
    """現在の工程に筆跡を描いてadvance_stageを受理させる。"""
    assert controller.stage_id == stage_id
    accept_all(controller, draw_actions(stage_id, color))
    return accept(controller, advance())


def advance_to(controller: StageController, stage_id: str) -> None:
    """指定した工程の入場直後まで、筆跡を描いて進める。"""
    while controller.stage_id != stage_id:
        draw_and_advance(controller, controller.stage_id)


def full_run(controller: StageController) -> list[ProposalResult]:
    """rough〜refineを描いてfinishまで進め、各遷移の結果を返す。"""
    return [draw_and_advance(controller, stage_id) for stage_id in STAGE_ORDER]


def noop_stroke_actions(width: int = 16, height: int = 16, seed: int = 0) -> list[DrawingAction]:
    """どの画素も変えない（grain欠落で全画素が落ちる）1画素のstroke_v2を含むActionの列。"""
    scratch = Engine(width, height, seed=seed)
    scratch.apply(DrawingAction("create_layer", {"layer_id": "w", "name": "w", "role": "rough"}))
    scratch.apply(DrawingAction("select_layer", {"layer_id": "w"}))
    setup = [
        model("select_brush", {"brush_id": "pencil_graphite"}),
        model("set_brush_params", {"size_level_id": 0, "opacity_level_id": 10}),
        model("select_color", {"color_id": 0}),
    ]
    for action in setup:
        scratch.apply(action)
    blank = bytes(scratch.state.layers[0].pixels.data)
    for y in range(height):
        for x in range(width):
            probe = model("stroke_v2", {"samples": [sample(x, y)]})
            scratch.apply(probe)
            if bytes(scratch.state.layers[0].pixels.data) == blank:
                return [*setup, probe]
    raise AssertionError("no no-op stroke position found")


def full_erase_actions(layer_id: str) -> list[DrawingAction]:
    """選択レイヤー全面を消すActionの列。"""
    return [
        model("select_layer", {"layer_id": layer_id}),
        model("select_brush", {"brush_id": "eraser_hard"}),
        model("set_brush_params", {"size_level_id": FULL_COVER_LEVEL, "opacity_level_id": 10}),
        model("erase_v2", {"samples": [sample(8, 8)]}),
    ]
