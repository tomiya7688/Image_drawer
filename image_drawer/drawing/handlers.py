"""v1・v2共通のAction処理。各関数は検証を終えてから状態を変更する（原子性）。"""

from __future__ import annotations

from typing import Any

from .models import (
    FINISH_STAGE_ID,
    LAYER_ROLES_Z_ORDER,
    MAX_LAYER_AREA,
    MAX_LAYERS,
    REVIEW_REASONS,
    STAGE_IDS,
    DrawingState,
    Layer,
)
from .palette import palette_rgb
from .pixels import PixelBuffer
from .validation import DrawingError, require_exact_keys, require_identifier

PHASES = ("sketch", "line", "color", "correct", "other")
MAX_POINTS_TOTAL = 200_000
MAX_AREA_TOTAL = 50_000_000


# {
#   責務: [ handle_create_layer: 透明レイヤーを最上位（作成順の末尾）へ追加する ]
#   処理: [ 1: キー・ID・名前・role検査 2: 重複・枚数上限・面積上限を確認 3: 追加 ]
#   引数: [ state: 状態, payload: {layer_id,name}（v2は+role）, uses_roles: v2ならTrue ]
#   エラー: [ DrawingError: 検査違反 ]
# }
def handle_create_layer(state: DrawingState, payload: Any, uses_roles: bool) -> None:
    keys = {"layer_id", "name", "role"} if uses_roles else {"layer_id", "name"}
    data = require_exact_keys(payload, keys, "create_layer.payload")
    layer_id = require_identifier(data["layer_id"], "layer_id")
    name = data["name"]
    if type(name) is not str or not 1 <= len(name) <= 128:
        raise DrawingError("layer name must be a string of 1..128 characters")
    role = data["role"] if uses_roles else None
    if uses_roles and role not in LAYER_ROLES_Z_ORDER:
        raise DrawingError(f"unknown layer role: {role!r}")
    if any(layer.id == layer_id for layer in state.layers):
        raise DrawingError(f"layer_id already used: {layer_id}")
    if len(state.layers) + 1 > MAX_LAYERS:
        raise DrawingError(f"layer limit exceeded: {MAX_LAYERS}")
    if state.width * state.height * (len(state.layers) + 1) > MAX_LAYER_AREA:
        raise DrawingError("canvas area x layer count exceeds the budget")
    pixels = PixelBuffer(state.width, state.height)
    state.layers.append(Layer(layer_id, name, role, pixels))


# {
#   責務: [ handle_select_layer: 既存レイヤーを作業対象にする（画素は変えない） ]
# }
def handle_select_layer(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"layer_id"}, "select_layer.payload")
    layer = state.find_layer(require_identifier(data["layer_id"], "layer_id"))
    state.active_layer_id = layer.id


# {
#   責務: [ handle_select_color: 固定256色の番号を選ぶ（画素は変えない） ]
# }
def handle_select_color(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"color_id"}, "select_color.payload")
    palette_rgb(data["color_id"])
    state.selected_color_id = data["color_id"]


# {
#   責務: [ handle_set_phase: v1の工程ラベルを記録する（順序は強制しない） ]
# }
def handle_set_phase(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"phase"}, "set_phase.payload")
    if data["phase"] not in PHASES:
        raise DrawingError(f"unknown phase: {data['phase']!r}")
    state.phase = data["phase"]


# {
#   責務: [ handle_finish: エピソードを終了する。payloadは空。v2で工程設定済みなら工程を"finish"にする ]
#   引数: [ state: 状態, payload: 空の辞書, is_v2: v2 profileか ]
# }
def handle_finish(state: DrawingState, payload: Any, is_v2: bool = False) -> None:
    require_exact_keys(payload, set(), "finish.payload")
    state.finished = True
    if is_v2 and state.stage_id is not None:
        state.stage_id = FINISH_STAGE_ID


# {
#   責務: [ handle_enter_stage: 工程マーカー。現在の工程を設定する（画素は変えない、v2のみ） ]
#   エラー: [ DrawingError: 未知のstage_id ]
# }
def handle_enter_stage(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"stage_id"}, "enter_stage.payload")
    if data["stage_id"] not in STAGE_IDS:
        raise DrawingError(f"unknown stage_id: {data['stage_id']!r}")
    state.stage_id = data["stage_id"]


# {
#   責務: [ handle_exit_stage: 工程マーカー。現在の工程と一致することだけを確認する（状態は変えない） ]
#   エラー: [ DrawingError: 未入場、または現在の工程と不一致 ]
# }
def handle_exit_stage(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"stage_id"}, "exit_stage.payload")
    if state.stage_id is None or data["stage_id"] != state.stage_id:
        raise DrawingError(f"exit_stage must name the current stage {state.stage_id!r}")


# {
#   責務: [ handle_advance_stage: 工程移行の提案マーカー。形式と工程設定済みだけを確認する（状態は変えない） ]
#   処理: [ payloadは{}または{review_reason: 許可コード}。移行先は含められない ]
#   エラー: [ DrawingError: 未入場、キー不一致、未知のreview_reason ]
# }
def handle_advance_stage(state: DrawingState, payload: Any) -> None:
    if state.stage_id is None or state.stage_id == FINISH_STAGE_ID:
        raise DrawingError("advance_stage requires an entered, unfinished stage")
    if isinstance(payload, dict) and payload:
        data = require_exact_keys(payload, {"review_reason"}, "advance_stage.payload")
        if data["review_reason"] not in REVIEW_REASONS:
            raise DrawingError(f"unknown review_reason: {data['review_reason']!r}")
    else:
        require_exact_keys(payload, set(), "advance_stage.payload")


# {
#   責務: [ handle_clear_active_layer: 選択中レイヤーを未選択にする（画素は変えない、v2のみ） ]
# }
def handle_clear_active_layer(state: DrawingState, payload: Any) -> None:
    require_exact_keys(payload, set(), "clear_active_layer.payload")
    state.active_layer_id = None


# {
#   責務: [ handle_set_layer_visibility: レイヤーの可視性を切り替える（Controller用、v2のみ） ]
# }
def handle_set_layer_visibility(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"layer_id", "visible"}, "set_layer_visibility.payload")
    if type(data["visible"]) is not bool:
        raise DrawingError("visible must be a boolean")
    state.find_layer(require_identifier(data["layer_id"], "layer_id")).visible = data["visible"]


# {
#   責務: [ writable_layer: 描画先として有効な選択レイヤーを返す ]
#   処理: [ 未選択・非表示（v2）なら拒否する ]
#   エラー: [ DrawingError: 描画先がない ]
# }
def writable_layer(state: DrawingState) -> Layer:
    if state.active_layer_id is None:
        raise DrawingError("no layer is selected")
    layer = state.find_layer(state.active_layer_id)
    if not layer.visible:
        raise DrawingError(f"layer {layer.id!r} is hidden and cannot be written")
    return layer


# {
#   責務: [ charge_budget: 累積の点数・候補面積予算を消費する（超過なら描画前に拒否） ]
#   引数: [ state: 状態, points: 今回の点数, area: 今回の候補面積 ]
#   エラー: [ DrawingError: 上限超過（この場合は状態を変えない） ]
# }
def charge_budget(state: DrawingState, points: int, area: int) -> None:
    if state.points_used + points > MAX_POINTS_TOTAL:
        raise DrawingError(f"total point budget exceeded: {MAX_POINTS_TOTAL}")
    if state.area_used + area > MAX_AREA_TOTAL:
        raise DrawingError(f"total candidate-area budget exceeded: {MAX_AREA_TOTAL}")
    state.points_used += points
    state.area_used += area


__all__ = [
    "PHASES",
    "charge_budget",
    "handle_advance_stage",
    "handle_clear_active_layer",
    "handle_create_layer",
    "handle_enter_stage",
    "handle_exit_stage",
    "handle_finish",
    "handle_select_color",
    "handle_select_layer",
    "handle_set_layer_visibility",
    "handle_set_phase",
    "writable_layer",
]
