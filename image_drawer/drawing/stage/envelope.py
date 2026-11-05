"""提案の基本の正当性（型・actor・Controllerが付ける項目）と、提案時点のコピー。"""

from __future__ import annotations

import copy

from ..models import DrawingAction


# {
#   責務: [ snapshot_action: 提案時点のコピーを作る。Actionでない値は"<invalid>"の代用Actionにする ]
# }
def snapshot_action(action: object) -> DrawingAction:
    if isinstance(action, DrawingAction):
        try:
            return copy.deepcopy(action)
        except Exception:
            pass
    return DrawingAction("<invalid>", {}, "model")


# {
#   責務: [ check_envelope: 提案の基本の正当性（型・actor・Controllerが付ける項目が空）を確認する ]
#   戻り値: [ reason: "invalid_envelope: ..."。正当ならNone ]
# }
def check_envelope(action: object) -> str | None:
    if not isinstance(action, DrawingAction):
        return "invalid_envelope: proposal must be a DrawingAction"
    if type(action.type) is not str or not isinstance(action.payload, dict):
        return "invalid_envelope: type must be a string and payload an object"
    if action.actor != "model":
        return f"invalid_envelope: actor must be 'model', got {action.actor!r}"
    if action.seq is not None or action.stage_id is not None or action.group is not None:
        return "invalid_envelope: seq, stage_id and group are assigned by the controller"
    return None
