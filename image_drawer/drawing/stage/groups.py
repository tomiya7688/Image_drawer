"""原子的な記録グループ（初期・遷移・完了）のイベント列を作る純関数。状態は変更しない。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models import FINISH_STAGE_ID, STAGE_IDS, DrawingAction, DrawingState, EventGroup
from .table import ROLE_LAYER_NAMES, StageTable

SYSTEM = "system"


# {
#   責務: [ _system_event: actor=systemのイベントを作る（stage_idは適用直後の工程、groupは後で付ける） ]
# }
def _system_event(action_type: str, payload: dict[str, Any], stage_id: str) -> DrawingAction:
    return DrawingAction(action_type, payload, SYSTEM, None, None, stage_id)


# {
#   責務: [ system_layer_id: roleと既存のsystem作成数だけから決まるレイヤーIDを返す ]
#   戻り値: [ layer_id: sys_<role>_<連番3桁>（連番は1始まり） ]
# }
def system_layer_id(state: DrawingState, role: str) -> str:
    prefix = f"sys_{role}_"
    count = sum(1 for layer in state.layers if layer.id.startswith(prefix))
    return f"{prefix}{count + 1:03d}"


# {
#   責務: [ _entry_events: 工程入場の列（enter_stage、既定roleがあればcreate_layer、clear_active_layer）を作る ]
# }
def _entry_events(table: StageTable, state: DrawingState, stage_id: str) -> list[DrawingAction]:
    events = [_system_event("enter_stage", {"stage_id": stage_id}, stage_id)]
    role = table.entry_default_roles[stage_id]
    if role is not None:
        payload = {
            "layer_id": system_layer_id(state, role),
            "name": ROLE_LAYER_NAMES[role],
            "role": role,
        }
        events.append(_system_event("create_layer", payload, stage_id))
    events.append(_system_event("clear_active_layer", {}, stage_id))
    return events


# {
#   責務: [ build_initial_group: 構築時のgroup 0（enter_stage(rough)、create_layer、clear_active_layer）を作る ]
#   引数: [ table: 工程表, time_ms: 全イベントに付ける時刻 ]
#   戻り値: [ events: groupとtime_ms付きの3件 ]
# }
def build_initial_group(table: StageTable, time_ms: int) -> list[DrawingAction]:
    empty_state = DrawingState(0, 0, "", 0)
    return with_group(_entry_events(table, empty_state, STAGE_IDS[0]), 0, time_ms)


# {
#   責務: [ build_transition_group: advance_stageの受理に伴うイベント列を作る（最終工程ではfinalize） ]
#   処理: [
#     通常: advance, exit_stage, set_layer_visibility(false)×k, enter_stage, create_layer?, clear_active_layer
#     refine: advance, exit_stage(refine), finish
#   ]
#   引数: [ table: 工程表, state: 現在の状態（読取のみ）, advance: 審査済みのadvance_stage提案 ]
#   戻り値: [ events: stage_id付き・group未付与のイベント列（先頭はadvance自身） ]
# }
def build_transition_group(
    table: StageTable, state: DrawingState, advance: DrawingAction
) -> list[DrawingAction]:
    current = state.stage_id
    events = [replace(advance, stage_id=current)]
    events.append(_system_event("exit_stage", {"stage_id": current}, current))
    next_id = table.next_stage_id(current)
    if next_id is None:
        events.append(_system_event("finish", {}, FINISH_STAGE_ID))
        return events
    hidden_roles = table.hide_on_enter.get(next_id, ())
    for layer in state.layers:
        if layer.visible and layer.role in hidden_roles:
            payload = {"layer_id": layer.id, "visible": False}
            events.append(_system_event("set_layer_visibility", payload, current))
    return events + _entry_events(table, state, next_id)


# {
#   責務: [ with_group: 全イベントへ同一のgroup（id・index・size）とtime_msを付ける ]
#   引数: [ events: 注釈前のイベント列, group_id: グループ番号, time_ms: 共通の時刻 ]
#   戻り値: [ annotated: 注釈付きの新しい列（入力は変えない） ]
# }
def with_group(events: list[DrawingAction], group_id: int, time_ms: int) -> list[DrawingAction]:
    size = len(events)
    return [
        replace(event, time_ms=time_ms, group=EventGroup(group_id, index, size))
        for index, event in enumerate(events)
    ]
