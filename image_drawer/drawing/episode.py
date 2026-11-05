"""DrawingEpisodeのデータ型と厳密なJSON読取。保存・読込はepisode_io、再生はreplayが担当する。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .catalog import BRUSH_SET_ID, CONTROL_SET_ID, default_catalog
from .engine import RENDERER_V1, RENDERER_V2
from .models import MAX_EVENTS, DrawingAction
from .palette import PALETTE_ID
from .validation import DrawingError, require_exact_keys, require_int

V1_KEYS = {
    "schema_version", "episode_id", "origin", "prompt", "canvas", "palette_id",
    "brush_set_id", "renderer_version", "events", "metadata",
}  # fmt: skip
V2_EXTRA_KEYS = {"seed", "brush_manifest_sha256", "control_set_id", "control_sha256"}
V2_WORKFLOW_KEYS = {
    "workflow_id", "workflow_sha256", "style_intent", "budget", "status", "termination_reason",
}  # fmt: skip
STYLE_INTENT_KEYS = {"version", "monochrome_line_art_only", "flat_style_no_shading"}
SCHEMA_BY_RENDERER = {RENDERER_V1: "drawing-episode/v1", RENDERER_V2: "drawing-episode/v2"}
SINGLE_ORIGINS = ("human", "model", "synthetic", "derived")
V2_ORIGIN = "mixed"
STATUSES = ("complete", "draft")
DRAFT_REASONS = (
    "action_budget_exhausted", "attempt_budget_exhausted", "in_progress", "recovered_truncated",
    "internal_error",
)  # fmt: skip
MAX_ACTIONS_LIMIT = 19_900  # stage.budget.MAX_ACTIONS_LIMITと同じ値（循環importを避けて複製）
MAX_ATTEMPTS_LIMIT = 99_500


# {
#   責務: [ DrawingEpisode: 1枚を描く過程のヘッダとAction列。画像バイトは含まない ]
#   フィールド: [
#     renderer_version: raster-v1またはbrush-raster-v2, events: DrawingAction列,
#     seed: v2のgrain seed（v1は0）, 他はdrawing_process_v1.md §6の同名キー,
#     workflow_id/workflow_sha256/style_intent/budget/status/termination_reason: v2のheader
#       （工程Controllerが付ける。v1では使わない）
#   ]
# }
@dataclass
class DrawingEpisode:
    episode_id: str
    origin: str
    prompt: str
    width: int
    height: int
    renderer_version: str
    events: list[DrawingAction]
    metadata: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    workflow_id: str | None = None
    workflow_sha256: str | None = None
    style_intent: dict[str, Any] | None = None
    budget: dict[str, int] | None = None
    status: str = "complete"
    termination_reason: str | None = "finished"

    # {
    #   責務: [ to_dict: Episode JSON（v1は10キー、v2は+10キー）の辞書にする ]
    #   エラー: [ DrawingError: v2のheader項目が未設定、またはeventの注釈が未設定 ]
    # }
    def to_dict(self) -> dict[str, Any]:
        is_v2 = self.renderer_version == RENDERER_V2
        data: dict[str, Any] = {
            "schema_version": SCHEMA_BY_RENDERER[self.renderer_version],
            "episode_id": self.episode_id,
            "origin": self.origin,
            "prompt": self.prompt,
            "canvas": {"width": self.width, "height": self.height},
            "palette_id": PALETTE_ID,
            "brush_set_id": BRUSH_SET_ID if is_v2 else "brush-v1",
            "renderer_version": self.renderer_version,
            "events": [event.to_dict(annotated=is_v2) for event in self.events],
            "metadata": self.metadata,
        }
        if is_v2:
            catalog = default_catalog()
            data.update(
                seed=self.seed,
                brush_manifest_sha256=catalog.brush_sha256,
                control_set_id=CONTROL_SET_ID,
                control_sha256=catalog.control_sha256,
                **self._workflow_header(),
            )
        return data

    # {
    #   責務: [ _workflow_header: v2のworkflow・style_intent・budget・status・termination_reasonを返す ]
    #   エラー: [ DrawingError: 未設定の項目がある ]
    # }
    def _workflow_header(self) -> dict[str, Any]:
        header = {key: getattr(self, key) for key in sorted(V2_WORKFLOW_KEYS)}
        if any(value is None for value in header.values()):
            raise DrawingError("a v2 episode requires workflow_id, style_intent, budget and status")
        return header

    # {
    #   責務: [ from_dict: ヘッダを厳密に検証してEpisodeを作る ]
    #   処理: [
    #     1: キー集合と識別子（palette・brush set・hash・v2はworkflow）がこの実装と一致するか
    #     2: seqの連続・originの整合を確認。v1はactor・終端finishも確認する
    #        （v2のグループ・工程連鎖はepisode_groups、描画内容の検証はEngineが行う）
    #   ]
    #   エラー: [ DrawingError: 不一致 ]
    # }
    @classmethod
    def from_dict(cls, raw: Any) -> DrawingEpisode:
        renderer = raw.get("renderer_version") if isinstance(raw, dict) else None
        if renderer not in SCHEMA_BY_RENDERER:
            raise DrawingError(f"unsupported renderer_version: {renderer!r}")
        is_v2 = renderer == RENDERER_V2
        keys = V1_KEYS | ((V2_EXTRA_KEYS | V2_WORKFLOW_KEYS) if is_v2 else set())
        data = require_exact_keys(raw, keys, "episode")
        _check_identifiers(data, is_v2)
        canvas = require_exact_keys(data["canvas"], {"width", "height"}, "canvas")
        if type(data["episode_id"]) is not str or not data["episode_id"]:
            raise DrawingError("episode_id must be a non-empty string")
        if type(data["prompt"]) is not str or not isinstance(data["metadata"], dict):
            raise DrawingError("prompt must be a string and metadata an object")
        events_raw = data["events"]
        if not isinstance(events_raw, list) or not 1 <= len(events_raw) <= MAX_EVENTS:
            raise DrawingError(f"events must be a list of 1..{MAX_EVENTS} items")
        events = [DrawingAction.from_dict(item, annotated=is_v2) for item in events_raw]
        _check_events(events, data["origin"], is_v2)
        episode = cls(
            data["episode_id"], data["origin"], data["prompt"], canvas["width"], canvas["height"],
            renderer, events, data["metadata"], data["seed"] if is_v2 else 0,
        )  # fmt: skip
        if is_v2:
            _apply_workflow_header(episode, data)
        return episode


# {
#   責務: [ _check_identifiers: schema・palette・brush set・hashが実装と一致することを確認する ]
# }
def _check_identifiers(data: Any, is_v2: bool) -> None:
    renderer = data["renderer_version"]
    expected = {
        "schema_version": SCHEMA_BY_RENDERER[renderer],
        "palette_id": PALETTE_ID,
        "brush_set_id": BRUSH_SET_ID if is_v2 else "brush-v1",
    }
    for key, value in expected.items():
        if data[key] != value:
            raise DrawingError(f"{key} must be {value!r}, got {data[key]!r}")
    if is_v2:
        catalog = default_catalog()
        if (
            data["control_set_id"] != CONTROL_SET_ID
            or data["brush_manifest_sha256"] != catalog.brush_sha256
            or data["control_sha256"] != catalog.control_sha256
        ):
            raise DrawingError("brush/control set identity or checksum mismatch")
        require_int(data["seed"], "seed", 0, 0xFFFFFFFF)


# {
#   責務: [ check_workflow_identity: workflow_idとworkflow_sha256が同梱のStage JSONと一致することを確認する ]
#   エラー: [ DrawingError: 不一致 ]
# }
def check_workflow_identity(workflow_id: Any, workflow_sha256: Any) -> None:
    # stage.tableはstageパッケージ経由でこのモジュールを読み込むため、循環を避けて関数内で読む。
    from .stage.table import default_stage_table

    table = default_stage_table()
    if workflow_id != table.workflow_id or workflow_sha256 != table.workflow_sha256:
        raise DrawingError("workflow identity or checksum mismatch")


# {
#   責務: [ _apply_workflow_header: v2のworkflow・style_intent・budget・status・理由を検証してEpisodeへ設定する ]
#   エラー: [ DrawingError: キー集合・型・値域・workflow識別の不一致 ]
# }
def _apply_workflow_header(episode: DrawingEpisode, data: Any) -> None:
    check_workflow_identity(data["workflow_id"], data["workflow_sha256"])
    intent = require_exact_keys(data["style_intent"], STYLE_INTENT_KEYS, "style_intent")
    version = intent["version"]
    if type(version) is not str or not 1 <= len(version) <= 64:
        raise DrawingError("style_intent.version must be a string of 1..64 characters")
    for key in ("monochrome_line_art_only", "flat_style_no_shading"):
        if type(intent[key]) is not bool:
            raise DrawingError(f"style_intent.{key} must be a boolean")
    budget = require_exact_keys(data["budget"], {"max_actions", "max_attempts"}, "budget")
    require_int(budget["max_actions"], "budget.max_actions", 1, MAX_ACTIONS_LIMIT)
    require_int(budget["max_attempts"], "budget.max_attempts", 1, MAX_ATTEMPTS_LIMIT)
    reason = data["termination_reason"]
    if data["status"] not in STATUSES or reason not in ("finished", *DRAFT_REASONS):
        raise DrawingError("status or termination_reason is unknown")
    episode.workflow_id, episode.workflow_sha256 = data["workflow_id"], data["workflow_sha256"]
    episode.style_intent, episode.budget = dict(intent), dict(budget)
    episode.status, episode.termination_reason = data["status"], reason


# {
#   責務: [ _check_events: seq連続・origin整合を確認する。v1はactor整合・終端finishも確認する ]
#   処理: [ v2はoriginを常にmixedとし、systemのみ・finishなしのdraftを許す（構造はepisode_groupsが検査） ]
# }
def _check_events(events: list[DrawingAction], origin: str, is_v2: bool = False) -> None:
    for index, event in enumerate(events):
        if event.seq != index:
            raise DrawingError(f"event seq must be {index}, got {event.seq!r}")
    if is_v2:
        if origin != V2_ORIGIN:
            raise DrawingError(f"origin must be {V2_ORIGIN!r} in v2, got {origin!r}")
        return
    actors = {event.actor for event in events}
    if actors == {"system"}:
        raise DrawingError("an episode made only of system actions is rejected")
    expected = next(iter(actors)) if len(actors) == 1 else "mixed"
    if origin != expected:
        raise DrawingError(f"origin must be {expected!r} for the given actors, got {origin!r}")
    if events[-1].type != "finish" or any(e.type == "finish" for e in events[:-1]):
        raise DrawingError("finish must appear exactly once, as the last event")


def _reject_decimal(value: str) -> Any:
    raise DrawingError(f"integer required, got {value}")


def _reject_constant(value: str) -> Any:
    raise DrawingError(f"non-finite number is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise DrawingError(f"duplicate key: {key}")
        data[key] = value
    return data


# {
#   責務: [ loads_strict: 小数・指数表記・NaN・重複キーを拒否するJSON読取 ]
#   引数: [ text: JSON文字列 ]
#   戻り値: [ document: 解析済みの値 ]
#   エラー: [ DrawingError: 曖昧な数値・重複キー ]
# }
def loads_strict(text: str) -> Any:
    return json.loads(
        text,
        parse_float=_reject_decimal,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
