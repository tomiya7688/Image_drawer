"""Policyが受け取れる観測。目標画像・目標特徴・Rewardを持つフィールドは作らない。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from image_drawer.drawing import Engine
from image_drawer.drawing.stage import Budget, StageController
from image_drawer.training.image_ops import box_downscale

# 契約: Observationのフィールド名の集合はこれと完全に一致する(設計書3.1・5.2節)。
OBSERVATION_KEYS: frozenset[str] = frozenset(
    {
        "text_condition",
        "stage_id",
        "canvas_summary",
        "layer_summary",
        "tool_state",
        "budget",
        "action_tail",
    }
)
# 直近Actionの種別ラベル。6種別+その他+なし(one-hotで8)。
ACTION_KINDS: tuple[str, ...] = (
    "select_layer",
    "select_color",
    "select_brush",
    "set_brush_params",
    "stroke_v2",
    "advance_stage",
)
TAIL_KIND_LABELS: tuple[str, ...] = (*ACTION_KINDS, "other", "none")
# 拒否理由の粗い分類(目標画像に依存しない)。
REJECT_CLASSES: tuple[str, ...] = ("illegal_stage", "no_layer", "invalid_value", "budget", "other")
_REJECT_PREFIX_CLASS = {
    "action_not_allowed": "illegal_stage",
    "stage_not_ready": "illegal_stage",
    "review_reason_not_allowed": "illegal_stage",
    "review_after_drawing": "illegal_stage",
    "review_intent_not_confirmed": "illegal_stage",
    "layer_not_selectable": "no_layer",
    "layer_not_writable": "no_layer",
    "role_not_creatable": "no_layer",
    "reserved_layer_id": "no_layer",
    "layer_capacity": "budget",
    "invalid_envelope": "invalid_value",
    "invalid_payload": "invalid_value",
    "engine_rejected": "invalid_value",
}


# {
#   責務: [ LayerBrief: 自分のレイヤー1枚の概要。画素そのものは持たない ]
#   フィールド: [ layer_id: Actionの宛先に使うID, role, visible, has_pixels, is_active ]
# }
@dataclass(frozen=True)
class LayerBrief:
    layer_id: str
    role: str | None
    visible: bool
    has_pixels: bool
    is_active: bool


# {
#   責務: [ ToolState: 色・ブラシ・サイズ・不透明度・レイヤーが選択済みかの真偽 ]
# }
@dataclass(frozen=True)
class ToolState:
    layer_selected: bool
    color_selected: bool
    brush_selected: bool
    size_set: bool
    opacity_set: bool


# {
#   責務: [ BudgetView: 残りAction/attemptの比率(0〜1) ]
# }
@dataclass(frozen=True)
class BudgetView:
    remaining_actions_ratio: float
    remaining_attempts_ratio: float


# {
#   責務: [ ActionBrief: 自分の直近Actionの種別・受理/拒否・粗い拒否分類だけ ]
#   フィールド: [ kind: TAIL_KIND_LABELSのいずれか, accepted, reject_class: 拒否時のみREJECT_CLASSES ]
# }
@dataclass(frozen=True)
class ActionBrief:
    kind: str
    accepted: bool
    reject_class: str | None


# {
#   責務: [ Observation: Policyの入力。フィールドはOBSERVATION_KEYSと一致する ]
# }
@dataclass(frozen=True)
class Observation:
    text_condition: NDArray[np.float32]
    stage_id: str
    canvas_summary: NDArray[np.float32]
    layer_summary: tuple[LayerBrief, ...]
    tool_state: ToolState
    budget: BudgetView
    action_tail: tuple[ActionBrief, ...]


# {
#   責務: [ classify_reject_reason: Controllerの拒否理由コードを粗い分類へ写す ]
#   引数: [ reason: "<code>"または"<code>: <detail>" ]
#   戻り値: [ reject_class: REJECT_CLASSESのいずれか ]
# }
def classify_reject_reason(reason: str | None) -> str:
    if not reason:
        return "other"
    return _REJECT_PREFIX_CLASS.get(reason.split(":", 1)[0].strip(), "other")


# {
#   責務: [
#     ObservationBuilder: text_condition・Controller・Engine・自分の試行ログだけから観測を作る
#   ]
#   フィールド: [ tail_length: 直近Actionの件数, summary_side: キャンバス縮小の一辺 ]
# }
class ObservationBuilder:
    def __init__(self, tail_length: int = 8, summary_side: int = 8) -> None:
        if tail_length < 1 or summary_side < 1:
            raise ValueError("tail_lengthとsummary_sideは1以上にしてください")
        self.tail_length = tail_length
        self.summary_side = summary_side

    # {
    #   責務: [ build: 自分のキャンバス・レイヤー・試行ログから観測を作る ]
    #   処理: [
    #     1: 合成画像を縮小したRGBサマリを作る 2: レイヤー・道具・予算の概要を集める
    #     3: 試行ログの直近からActionBriefを作る(拒否理由は粗い分類へ写す)
    #   ]
    #   引数: [ text_condition: TextEncoderの出力, controller, engine, budget: Controllerと同じBudget ]
    #   戻り値: [ observation: 目標に依存しない観測 ]
    # }
    def build(
        self,
        text_condition: NDArray[np.float32],
        controller: StageController,
        engine: Engine,
        budget: Budget,
    ) -> Observation:
        state = engine.state
        # 自分の合成キャンバスだけから作る。縮小サイズがキャンバスを超える場合は一辺に抑える。
        side = min(self.summary_side, state.width)
        rgba = np.frombuffer(engine.render().rgba, dtype=np.uint8)
        rgb = rgba.reshape(state.height, state.width, 4)[..., :3].astype(np.float64) / 255.0
        small = box_downscale(rgb, side) if state.width == state.height else rgb[:side, :side]
        summary = np.zeros((3, self.summary_side, self.summary_side), dtype=np.float32)
        summary[:, :side, :side] = np.moveaxis(small, 2, 0)

        layers = tuple(
            LayerBrief(
                layer.id,
                layer.role,
                layer.visible,
                any(layer.pixels.data[3::4]),
                layer.id == state.active_layer_id,
            )
            for layer in state.layers
        )
        tool = ToolState(
            state.active_layer_id is not None,
            state.selected_color_id is not None,
            state.selected_brush_id is not None,
            state.size_level_id is not None,
            state.opacity_level_id is not None,
        )
        assert budget.max_attempts is not None
        view = BudgetView(
            max(0.0, 1.0 - controller.accepted_action_count / budget.max_actions),
            max(0.0, 1.0 - controller.attempt_count / budget.max_attempts),
        )
        tail = tuple(
            ActionBrief(
                record.action.type if record.action.type in ACTION_KINDS else "other",
                record.accepted,
                None if record.accepted else classify_reject_reason(record.reason),
            )
            for record in controller.attempts()[-self.tail_length :]
        )
        return Observation(
            np.asarray(text_condition, dtype=np.float32),
            state.stage_id or "finish",
            summary,
            layers,
            tool,
            view,
            tail,
        )
