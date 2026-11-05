"""Observationを固定長の特徴ベクトルへ変換する(設計書4.2節)。目標を参照しない。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from image_drawer.training.config import STAGE_IDS
from image_drawer.training.observation import TAIL_KIND_LABELS, Observation

TOOL_FEATURES = 5
BUDGET_FEATURES = 2
TAIL_FEATURES = len(TAIL_KIND_LABELS) + 1  # 直近1件の種別one-hot + 直近のaccepted比率


# {
#   責務: [ feature_dim: 特徴次元D(text + 工程one-hot + キャンバス + 道具 + 予算 + 履歴 + バイアス) ]
# }
def feature_dim(text_dim: int, summary_side: int) -> int:
    return (
        text_dim + len(STAGE_IDS) + 3 * summary_side * summary_side
        + TOOL_FEATURES + BUDGET_FEATURES + TAIL_FEATURES + 1
    )  # fmt: skip


# {
#   責務: [ build_features: 観測を連結した特徴ベクトル(float64)にする ]
#   エラー: [ ValueError: 未知の工程 ]
# }
def build_features(observation: Observation) -> NDArray[np.float64]:
    if observation.stage_id not in STAGE_IDS:
        raise ValueError(f"未知のstage_id: {observation.stage_id}")
    stage = np.zeros(len(STAGE_IDS))
    stage[STAGE_IDS.index(observation.stage_id)] = 1.0
    tool = observation.tool_state
    tool_vector = np.array(
        [
            tool.layer_selected,
            tool.color_selected,
            tool.brush_selected,
            tool.size_set,
            tool.opacity_set,
        ],
        dtype=np.float64,
    )
    budget = observation.budget
    tail = observation.action_tail
    last_kind = np.zeros(len(TAIL_KIND_LABELS))
    last_kind[TAIL_KIND_LABELS.index(tail[-1].kind if tail else "none")] = 1.0
    accepted_ratio = float(np.mean([brief.accepted for brief in tail])) if tail else 0.0
    return np.concatenate(
        [
            observation.text_condition.astype(np.float64),
            stage,
            observation.canvas_summary.astype(np.float64).ravel(),
            tool_vector,
            [budget.remaining_actions_ratio, budget.remaining_attempts_ratio],
            last_kind,
            [accepted_ratio],
            [1.0],
        ]
    )
