"""複製したStage JSONから読む工程表（StageSpec・StageTable）。順序・role・権限を一箇所に持つ。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..catalog import DATA_DIR, canonical_sha256
from ..models import FINISH_STAGE_ID, LAYER_ROLES_Z_ORDER, REVIEW_REASONS, STAGE_IDS
from .errors import StageTableError

STAGE_JSON_NAME = "drawing_stage_workflow_v2.json"
ROLE_LAYER_NAMES = {
    "rough": "ラフ", "sketch": "下書き", "final_drawing": "本描き", "base_color": "基本色",
    "shade": "陰影", "highlight": "ハイライト", "correction": "修正",
}  # fmt: skip


# {
#   責務: [ StageSpec: 工程1つの権限と退出条件（Stage JSONの同名項目） ]
#   フィールド: [
#     id, name_ja: 工程, layer_roles: 作成できるrole, write_layer_roles: 書込・選択できるrole,
#     min_effectful_strokes: 通常退出に必要な有効筆跡数, review_only_reason_codes: レビュー退出の許可コード
#   ]
# }
@dataclass(frozen=True)
class StageSpec:
    id: str
    name_ja: str
    layer_roles: tuple[str, ...]
    write_layer_roles: tuple[str, ...]
    min_effectful_strokes: int
    review_only_reason_codes: tuple[str, ...]


# {
#   責務: [ StageTable: 6工程の固定順・入場時の既定role・非表示規則・提案可能な種別を保持する ]
#   フィールド: [
#     workflow_id/workflow_sha256: 識別子とcanonical JSONのSHA-256, stages: 固定順の工程,
#     entry_default_roles: 工程ID→入場時に作るrole（なければNone）,
#     hide_on_enter: 工程ID→入場時に非表示にするrole, allowed_model_actions/controller_only_actions: 種別
#   ]
# }
@dataclass(frozen=True)
class StageTable:
    workflow_id: str
    workflow_sha256: str
    stages: tuple[StageSpec, ...]
    entry_default_roles: dict[str, str | None]
    hide_on_enter: dict[str, tuple[str, ...]]
    allowed_model_actions: tuple[str, ...]
    controller_only_actions: tuple[str, ...]

    # {
    #   責務: [ spec: 工程IDからStageSpecを返す ]
    # }
    def spec(self, stage_id: str) -> StageSpec:
        return self.stages[STAGE_IDS.index(stage_id)]

    # {
    #   責務: [ next_stage_id: 次の工程IDを返す。最終工程ではNone ]
    # }
    def next_stage_id(self, stage_id: str) -> str | None:
        index = STAGE_IDS.index(stage_id)
        return STAGE_IDS[index + 1] if index + 1 < len(STAGE_IDS) else None


# {
#   責務: [ _string_tuple: 文字列だけのリストをタプルにする ]
#   エラー: [ StageTableError: リストでない、または文字列以外を含む ]
# }
def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise StageTableError(f"{name} must be a list of strings")
    return tuple(value)


# {
#   責務: [ _build_spec: 工程1つを検証してStageSpecにする ]
#   エラー: [ StageTableError: role・コード・上限が不正 ]
# }
def _build_spec(raw: Any) -> StageSpec:
    if not isinstance(raw, dict):
        raise StageTableError("stage must be an object")
    spec = StageSpec(
        id=raw.get("id"),
        name_ja=raw.get("name_ja"),
        layer_roles=_string_tuple(raw.get("layer_roles"), "layer_roles"),
        write_layer_roles=_string_tuple(raw.get("write_layer_roles"), "write_layer_roles"),
        min_effectful_strokes=raw.get("min_effectful_strokes"),
        review_only_reason_codes=_string_tuple(
            raw.get("review_only_reason_codes"), "review_only_reason_codes"
        ),
    )
    roles = spec.layer_roles + spec.write_layer_roles
    if any(role not in LAYER_ROLES_Z_ORDER for role in roles):
        raise StageTableError(f"stage {spec.id!r} uses a role missing from the Engine z-order")
    if type(spec.min_effectful_strokes) is not int or spec.min_effectful_strokes < 0:
        raise StageTableError("min_effectful_strokes must be a non-negative integer")
    if any(code not in REVIEW_REASONS for code in spec.review_only_reason_codes):
        raise StageTableError(f"stage {spec.id!r} has an unknown review reason code")
    return spec


# {
#   責務: [ parse_stage_table: Stage JSONの辞書を検証してStageTableにする ]
#   処理: [
#     1: 工程IDがSTAGE_IDSと同順か 2: z順がEngineのLAYER_ROLES_Z_ORDERと一致するか
#     3: 入場時の既定role・非表示規則・種別一覧を読む 4: canonical SHA-256を求める
#   ]
#   引数: [ document: Stage JSONを読んだ辞書 ]
#   戻り値: [ table: 検証済みの工程表 ]
#   エラー: [ StageTableError: 不整合 ]
# }
def parse_stage_table(document: Any) -> StageTable:
    if not isinstance(document, dict) or not isinstance(document.get("stages"), list):
        raise StageTableError("stage workflow must be an object with stages")
    stages = tuple(_build_spec(raw) for raw in document["stages"])
    if tuple(spec.id for spec in stages) != STAGE_IDS:
        raise StageTableError(f"stage ids must be {list(STAGE_IDS)} in this order")
    if tuple(document.get("layer_z_order_bottom_to_top", ())) != LAYER_ROLES_Z_ORDER:
        raise StageTableError("layer_z_order_bottom_to_top must equal the Engine z-order")
    if document.get("terminal") != FINISH_STAGE_ID or document.get("initial_stage") != STAGE_IDS[0]:
        raise StageTableError("terminal/initial_stage do not match the fixed workflow")
    defaults = document.get("stage_entry_default_layer_roles")
    if not isinstance(defaults, dict) or set(defaults) != set(STAGE_IDS):
        raise StageTableError("stage_entry_default_layer_roles must list every stage")
    if any(role is not None and role not in LAYER_ROLES_Z_ORDER for role in defaults.values()):
        raise StageTableError("stage_entry_default_layer_roles has an unknown role")
    hide: dict[str, tuple[str, ...]] = {}
    for rule in document.get("layer_visibility_transitions", []):
        stage_id = rule.get("on_enter_stage") if isinstance(rule, dict) else None
        if stage_id not in STAGE_IDS:
            raise StageTableError("layer_visibility_transitions has an unknown stage")
        hide[stage_id] = _string_tuple(rule.get("hide_layer_roles"), "hide_layer_roles")
    workflow_id = document.get("workflow_id")
    if type(workflow_id) is not str:
        raise StageTableError("workflow_id must be a string")
    return StageTable(
        workflow_id=workflow_id,
        workflow_sha256=canonical_sha256(document),
        stages=stages,
        entry_default_roles=dict(defaults),
        hide_on_enter=hide,
        allowed_model_actions=_string_tuple(document.get("allowed_model_actions"), "allowed"),
        controller_only_actions=_string_tuple(document.get("controller_only_actions"), "only"),
    )


# {
#   責務: [ load_stage_table: 複製したStage JSONを読んでStageTableにする。パス省略時は同梱の複製 ]
#   引数: [ path: Stage JSONのパス（省略可） ]
#   戻り値: [ table: 検証済みの工程表 ]
#   エラー: [ StageTableError: 読めない、またはJSONとして不正 ]
# }
def load_stage_table(path: Path | None = None) -> StageTable:
    target = path if path is not None else DATA_DIR / STAGE_JSON_NAME
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageTableError(f"cannot read stage workflow {target.name}: {exc}") from exc
    return parse_stage_table(document)


# {
#   責務: [ default_stage_table: 同梱のStageTableを1回だけ読んで共有する（不変） ]
# }
@lru_cache(maxsize=1)
def default_stage_table() -> StageTable:
    return load_stage_table()
