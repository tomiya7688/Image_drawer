"""工程別Reward仕様(stage_reward_v2.jsonの必要部分)の検証付き読み込み。

wheelにdocsは含まれないため、必要部分をここへ複製し、テストで正本のJSONと一致を確認する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_drawer.drawing.catalog import canonical_sha256
from image_drawer.training.reward_features import FEATURE_NAMES

SCHEMA_VERSION = "idraw-stage-reward/v2"
EMBEDDED_SPEC: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "reward_profile_id": "image-only-stage-reward-baseline-v2",
    "stage_ids": ["rough", "sketch", "final_drawing", "base_color", "shading", "refine"],
    "stage_potential_weights": {
        "rough": {"edge_32": 1},
        "sketch": {"edge_32": 0.3, "edge_64": 0.7},
        "final_drawing": {"edge_64": 0.4, "edge_128": 0.6},
        "base_color": {"rgb_32": 0.7, "rgb_hist_16": 0.3},
        "shading": {"luma_64": 0.65, "rgb_32": 0.35},
        "refine": {"rgb_128": 0.55, "edge_128": 0.25, "rgb_32": 0.2},
    },
    "final_quality_weights": {"rgb_128": 0.6, "edge_128": 0.3, "rgb_hist_16": 0.1},
}


# {
#   責務: [ StageRewardSpec: 工程ごとのpotential重みとfinal_qualityの重み ]
#   フィールド: [ profile_id, stage_ids, potential_weights, final_weights, sha256: 内容のcanonical SHA-256 ]
# }
@dataclass(frozen=True)
class StageRewardSpec:
    profile_id: str
    stage_ids: tuple[str, ...]
    potential_weights: Mapping[str, Mapping[str, float]]
    final_weights: Mapping[str, float]
    sha256: str


# {
#   責務: [ _check_weights: 重みの特徴名が既知で、0以上、合計が1であることを確認する ]
#   エラー: [ ValueError: 未知の特徴名、負の重み、合計が1でない ]
# }
def _check_weights(name: str, weights: Mapping[str, Any]) -> dict[str, float]:
    unknown = set(weights) - FEATURE_NAMES
    if unknown:
        raise ValueError(f"{name}に未知の特徴名があります: {sorted(unknown)}")
    values = {key: float(value) for key, value in weights.items()}
    if not values or any(value < 0 for value in values.values()):
        raise ValueError(f"{name}の重みは1つ以上で0以上にしてください")
    if abs(sum(values.values()) - 1.0) > 1e-9:
        raise ValueError(f"{name}の重みの合計が1ではありません: {sum(values.values())}")
    return values


# {
#   責務: [ parse_stage_reward_spec: JSON辞書を検証してStageRewardSpecにする ]
#   エラー: [ ValueError: schema違い、工程の過不足、重みの不正 ]
# }
def parse_stage_reward_spec(document: Mapping[str, Any]) -> StageRewardSpec:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_versionが{SCHEMA_VERSION}ではありません")
    stage_ids = tuple(document.get("stage_ids", ()))
    weights = document.get("stage_potential_weights", {})
    if not stage_ids or set(weights) != set(stage_ids):
        raise ValueError("stage_idsとstage_potential_weightsの工程が一致しません")
    return StageRewardSpec(
        profile_id=str(document["reward_profile_id"]),
        stage_ids=stage_ids,
        potential_weights={
            stage: _check_weights(f"stage_potential_weights[{stage}]", weights[stage])
            for stage in stage_ids
        },
        final_weights=_check_weights("final_quality_weights", document["final_quality_weights"]),
        sha256=canonical_sha256({key: document[key] for key in EMBEDDED_SPEC if key in document}),
    )


# {
#   責務: [ load_stage_reward_spec: stage_reward_v2.jsonを読んで検証する ]
# }
def load_stage_reward_spec(path: Path) -> StageRewardSpec:
    return parse_stage_reward_spec(json.loads(path.read_text(encoding="utf-8")))


# {
#   責務: [ default_stage_reward_spec: 同梱の複製から仕様を作る ]
# }
def default_stage_reward_spec() -> StageRewardSpec:
    return parse_stage_reward_spec(EMBEDDED_SPEC)
