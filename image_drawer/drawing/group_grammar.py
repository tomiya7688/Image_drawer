"""v2 Episodeのグループ文法（initial・action・transition・finalize）と、工程・レイヤー表の連鎖検査。

StageControllerのgroups・rules・progressは使わず、ここで独立に文法を実装する。
これにより、Controllerの出力をこの実装で検査でき、同じ誤りを共有しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import FINISH_STAGE_ID, STAGE_IDS, DrawingAction
from .stage.table import ROLE_LAYER_NAMES, StageTable
from .validation import EpisodeFormatError

SYSTEM_LAYER_PREFIX = "sys_"
ACTION_TYPES = (
    "create_layer", "select_layer", "select_color", "select_brush", "set_brush_params",
    "stroke_v2", "erase_v2",
)  # fmt: skip
INTENT_REASONS = ("monochrome_line_art_only", "flat_style_no_shading")


# {
#   責務: [ LayerRow: 連鎖検査が追う作成済みレイヤー1枚（ID・role・可視性） ]
# }
@dataclass
class LayerRow:
    layer_id: str
    role: str
    visible: bool = True


# {
#   責務: [ ChainState: グループを順に検査するときの工程・レイヤー表・modelイベント数・終了状態 ]
# }
@dataclass
class ChainState:
    stage: str | None = None
    layers: list[LayerRow] = field(default_factory=list)
    model_events: int = 0
    finished: bool = False


# 期待するイベント1件: (payload不一致の規則名, actor, type, payload, 適用直後のstage_id)
Expected = tuple[str, str, str, dict[str, Any], str]


# {
#   責務: [ _fail: 規則名付きのEpisodeFormatErrorを送出する ]
# }
def _fail(rule: str, event: DrawingAction | None, message: str) -> EpisodeFormatError:
    where = "" if event is None else f" (seq {event.seq})"
    return EpisodeFormatError(f"{rule}{where}: {message}")


# {
#   責務: [ _system_layer_id: roleと既存のsys_レイヤー数だけから決まる期待ID（sys_<role>_<連番3桁>） ]
# }
def _system_layer_id(state: ChainState, role: str) -> str:
    prefix = f"{SYSTEM_LAYER_PREFIX}{role}_"
    count = sum(1 for layer in state.layers if layer.layer_id.startswith(prefix))
    return f"{prefix}{count + 1:03d}"


# {
#   責務: [ _entry_expected: 入場先の期待列（enter_stage、既定roleがあればcreate_layer、clear_active_layer） ]
# }
def _entry_expected(table: StageTable, state: ChainState, stage_id: str) -> list[Expected]:
    expected: list[Expected] = [
        ("R6", "system", "enter_stage", {"stage_id": stage_id}, stage_id),
    ]
    role = table.entry_default_roles[stage_id]
    if role is not None:
        payload = {
            "layer_id": _system_layer_id(state, role),
            "name": ROLE_LAYER_NAMES[role],
            "role": role,
        }
        expected.append(("R9", "system", "create_layer", payload, stage_id))
    expected.append(("R4", "system", "clear_active_layer", {}, stage_id))
    return expected


# {
#   責務: [ _transition_expected: 現在工程からの遷移またはfinalizeグループの期待列（先頭のadvance_stageを除く） ]
# }
def _transition_expected(table: StageTable, state: ChainState) -> list[Expected]:
    current = state.stage
    expected: list[Expected] = [("R4", "system", "exit_stage", {"stage_id": current}, current)]
    next_id = table.next_stage_id(current)
    if next_id is None:
        return [*expected, ("R4", "system", "finish", {}, FINISH_STAGE_ID)]
    hidden_roles = table.hide_on_enter.get(next_id, ())
    for layer in state.layers:
        if layer.visible and layer.role in hidden_roles:
            payload = {"layer_id": layer.layer_id, "visible": False}
            expected.append(("R9", "system", "set_layer_visibility", payload, current))
    return expected + _entry_expected(table, state, next_id)


# {
#   責務: [ _compare: グループの各イベントを期待列と順に突き合わせる（type→actor→stage_id→payloadの順） ]
#   エラー: [ EpisodeFormatError: 件数・type・actor・stage_id・payloadのいずれかが不一致 ]
# }
def _compare(rule: str, group: list[DrawingAction], expected: list[Expected]) -> None:
    if len(group) != len(expected):
        raise _fail(rule, group[0], f"group must have {len(expected)} events, got {len(group)}")
    for event, (payload_rule, actor, action_type, payload, stage_id) in zip(group, expected):
        if event.type != action_type:
            raise _fail(rule, event, f"expected {action_type}, got {event.type}")
        if event.actor != actor:
            raise _fail("R10", event, f"{action_type} must have actor {actor}, got {event.actor}")
        # 工程入場は、飛ばし・逆行をR5ではなくR6として報告するためpayloadを先に見る。
        if payload_rule == "R6" and event.payload != payload:
            raise _fail("R6", event, f"enter_stage must enter {stage_id!r}: {event.payload}")
        if event.stage_id != stage_id:
            raise _fail("R5", event, f"stage_id must be {stage_id!r}, got {event.stage_id!r}")
        if event.payload != payload:
            raise _fail(payload_rule, event, f"{action_type} payload mismatch: {event.payload}")


# {
#   責務: [ _apply_expected: 検査済みの期待列の効果（工程・レイヤー作成・非表示・終了）を連鎖状態へ反映する ]
# }
def _apply_expected(state: ChainState, expected: list[Expected]) -> None:
    for _rule, _actor, action_type, payload, stage_id in expected:
        if action_type == "enter_stage":
            state.stage = stage_id
        elif action_type == "finish":
            state.stage, state.finished = FINISH_STAGE_ID, True
        elif action_type == "create_layer":
            state.layers.append(LayerRow(payload["layer_id"], payload["role"]))
        elif action_type == "set_layer_visibility":
            next(row for row in state.layers if row.layer_id == payload["layer_id"]).visible = False


# {
#   責務: [ check_initial_group: group 0（rough入場・sys_rough_001作成・選択解除）を検査して連鎖を開始する ]
# }
def check_initial_group(table: StageTable, state: ChainState, group: list[DrawingAction]) -> None:
    expected = _entry_expected(table, state, STAGE_IDS[0])
    _compare("R3", group, expected)
    _apply_expected(state, expected)


# {
#   責務: [ _check_review_reason: advance_stageのreview_reasonが現在工程の許可コードで、StyleIntentの裏付けがあるか確認する ]
# }
def _check_review_reason(
    table: StageTable, state: ChainState, intent: dict[str, Any], event: DrawingAction
) -> None:
    reason = event.payload.get("review_reason")
    if reason is None:
        return
    if reason not in table.spec(state.stage).review_only_reason_codes:
        raise _fail("R12", event, f"{reason} is not allowed in {state.stage}")
    if reason in INTENT_REASONS and intent.get(reason) is not True:
        raise _fail("R12", event, f"style_intent.{reason} must be true for {reason}")


# {
#   責務: [ _check_action_group: modelの通常Action（1件）を検査し、create_layerならレイヤー表へ追加する ]
# }
def _check_action_group(table: StageTable, state: ChainState, event: DrawingAction) -> None:
    if event.type not in ACTION_TYPES:
        raise _fail("R10", event, f"{event.type} cannot be a model action group")
    if event.stage_id != state.stage:
        raise _fail("R5", event, f"stage_id must be {state.stage!r}, got {event.stage_id!r}")
    state.model_events += 1
    if event.type != "create_layer":
        return
    layer_id, role = event.payload["layer_id"], event.payload["role"]
    if layer_id.startswith(SYSTEM_LAYER_PREFIX):
        raise _fail("R9", event, f"model layer_id must not start with {SYSTEM_LAYER_PREFIX!r}")
    if role not in table.spec(state.stage).layer_roles:
        raise _fail("R9", event, f"role {role} cannot be created in {state.stage}")
    if any(row.layer_id == layer_id for row in state.layers):
        raise _fail("R9", event, f"duplicate layer_id: {layer_id}")
    state.layers.append(LayerRow(layer_id, role))


# {
#   責務: [ _check_advance_group: advance_stageで始まる遷移（またはfinalize）グループを検査して連鎖を進める ]
# }
def _check_advance_group(
    table: StageTable,
    state: ChainState,
    intent: dict[str, Any],
    group: list[DrawingAction],
) -> None:
    advance = group[0]
    if advance.actor != "model":
        raise _fail("R10", advance, "advance_stage must have actor model")
    if advance.stage_id != state.stage:
        raise _fail("R5", advance, f"stage_id must be {state.stage!r}, got {advance.stage_id!r}")
    _check_review_reason(table, state, intent, advance)
    expected = _transition_expected(table, state)
    head: Expected = ("R4", "model", "advance_stage", advance.payload, state.stage)
    _compare("R4", group, [head, *expected])
    state.model_events += 1
    _apply_expected(state, expected)


# {
#   責務: [ check_group: group 1以降の1グループを文法（action / transition / finalize）に照らして検査する ]
#   引数: [ table: 工程表, state: 連鎖状態（更新される）, intent: style_intent, group: 検査するグループ ]
#   エラー: [ EpisodeFormatError: 終了後のグループ、またはどの文法にも合わない ]
# }
def check_group(
    table: StageTable, state: ChainState, intent: dict[str, Any], group: list[DrawingAction]
) -> None:
    first = group[0]
    if state.finished:
        raise _fail("R7", first, "no event may follow finish")
    if first.type == "advance_stage":
        _check_advance_group(table, state, intent, group)
    elif len(group) == 1 and first.actor == "model":
        _check_action_group(table, state, first)
    else:
        raise _fail("R4", first, f"group must be action, transition or finalize; got {first.type}")
