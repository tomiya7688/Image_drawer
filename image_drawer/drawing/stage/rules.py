"""提案Actionの工程による権限審査。状態を変更せず、同じ入力から必ず同じ結果を返す。"""

from __future__ import annotations

from ..models import MAX_LAYER_AREA, MAX_LAYERS, DrawingAction, DrawingState
from .exit_rules import check_advance
from .intent import StyleIntent
from .progress import StageProgress
from .table import StageSpec, StageTable

RESERVED_LAYER_PREFIX = "sys_"
PAINTING_TYPES = ("stroke_v2", "erase_v2")


# {
#   責務: [ remaining_system_layers: 現在工程より後の入場で、Controllerがこれから作るレイヤー数を数える ]
# }
def remaining_system_layers(table: StageTable, stage_id: str) -> int:
    later = table.stages[[spec.id for spec in table.stages].index(stage_id) + 1 :]
    return sum(1 for spec in later if table.entry_default_roles[spec.id] is not None)


# {
#   責務: [ _check_create_layer: create_layerの予約ID・role・これからのsystemレイヤー分の余裕を確認する ]
# }
def _check_create_layer(
    table: StageTable, spec: StageSpec, state: DrawingState, action: DrawingAction
) -> str | None:
    layer_id = action.payload.get("layer_id")
    if isinstance(layer_id, str) and layer_id.startswith(RESERVED_LAYER_PREFIX):
        return f"reserved_layer_id: {layer_id!r} starts with {RESERVED_LAYER_PREFIX!r}"
    role = action.payload.get("role")
    if role not in spec.layer_roles:
        return f"role_not_creatable: {role!r} cannot be created in {spec.id}"
    needed = len(state.layers) + 1 + remaining_system_layers(table, spec.id)
    if needed > MAX_LAYERS or state.width * state.height * needed > MAX_LAYER_AREA:
        return "layer_capacity: creating this layer would starve the mandatory stage layers"
    return None


# {
#   責務: [ _check_select_layer: 対象が可視で、現在工程の書込roleであることを確認する（未知IDはEngineが拒否） ]
# }
def _check_select_layer(spec: StageSpec, state: DrawingState, action: DrawingAction) -> str | None:
    layer_id = action.payload.get("layer_id")
    layer = next((item for item in state.layers if item.id == layer_id), None)
    if layer is None:
        return None
    if not layer.visible or layer.role not in spec.write_layer_roles:
        return f"layer_not_selectable: {layer.id!r} (role={layer.role}) in {spec.id}"
    return None


# {
#   責務: [ _check_painting: 選択中レイヤーのroleが書込可能であることを確認する（防御的検査） ]
# }
def _check_painting(spec: StageSpec, state: DrawingState) -> str | None:
    layer = next((item for item in state.layers if item.id == state.active_layer_id), None)
    if layer is not None and layer.role not in spec.write_layer_roles:
        return f"layer_not_writable: {layer.id!r} (role={layer.role}) in {spec.id}"
    return None


# {
#   責務: [ check_proposal: 提案Actionが現在工程で許されるかを審査する ]
#   処理: [
#     1: 提案可能な種別か 2: 種別ごとの権限（create_layer・select_layer・描画・advance_stage）
#     3: 設定系は工程に関わらず可（値はEngineが検査）
#   ]
#   引数: [ table: 工程表, state: 状態（読取のみ）, progress: 工程内計数, intent: StyleIntent, action: 提案 ]
#   戻り値: [ reason: "<code>: <detail>"。受理可能ならNone ]
# }
def check_proposal(
    table: StageTable,
    state: DrawingState,
    progress: StageProgress,
    intent: StyleIntent,
    action: DrawingAction,
) -> str | None:
    if action.type not in table.allowed_model_actions:
        return f"action_not_allowed: {action.type!r}"
    spec = table.spec(state.stage_id)
    if action.type == "create_layer":
        return _check_create_layer(table, spec, state, action)
    if action.type == "select_layer":
        return _check_select_layer(spec, state, action)
    if action.type in PAINTING_TYPES:
        return _check_painting(spec, state)
    if action.type == "advance_stage":
        return check_advance(spec, state, progress, intent, action.payload)
    return None
