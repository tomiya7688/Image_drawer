"""DrawingEpisodeのデータ型と厳密なJSON読取。永続化・Replayerは別Issueの範囲。"""

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
SCHEMA_BY_RENDERER = {RENDERER_V1: "drawing-episode/v1", RENDERER_V2: "drawing-episode/v2"}
SINGLE_ORIGINS = ("human", "model", "synthetic", "derived")


# {
#   責務: [ DrawingEpisode: 1枚を描く過程のヘッダとAction列。画像バイトは含まない ]
#   フィールド: [
#     renderer_version: raster-v1またはbrush-raster-v2, events: DrawingAction列,
#     seed: v2のgrain seed（v1は0）, 他はdrawing_process_v1.md §6の同名キー,
#     workflow_id/workflow_sha256/style_intent/budget/status/termination_reason: 工程Controllerが付ける
#       記録（S1bではメモリ上のみ。to_dict/from_dictは変えず、保存はS1cで拡張する）
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
    #   責務: [ to_dict: Episode JSON（v1は10キー、v2は+4キー）の辞書にする ]
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
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }
        if is_v2:
            catalog = default_catalog()
            data.update(
                seed=self.seed,
                brush_manifest_sha256=catalog.brush_sha256,
                control_set_id=CONTROL_SET_ID,
                control_sha256=catalog.control_sha256,
            )
        return data

    # {
    #   責務: [ from_dict: ヘッダを厳密に検証してEpisodeを作る ]
    #   処理: [
    #     1: キー集合と識別子（palette・brush set・hash）がこの実装と一致するか
    #     2: seqの連続・actor・originの整合・終端finishを確認（描画内容の検証はEngineが行う）
    #   ]
    #   エラー: [ DrawingError: 不一致 ]
    # }
    @classmethod
    def from_dict(cls, raw: Any) -> DrawingEpisode:
        renderer = raw.get("renderer_version") if isinstance(raw, dict) else None
        if renderer not in SCHEMA_BY_RENDERER:
            raise DrawingError(f"unsupported renderer_version: {renderer!r}")
        is_v2 = renderer == RENDERER_V2
        data = require_exact_keys(raw, V1_KEYS | (V2_EXTRA_KEYS if is_v2 else set()), "episode")
        _check_identifiers(data, is_v2)
        canvas = require_exact_keys(data["canvas"], {"width", "height"}, "canvas")
        if type(data["episode_id"]) is not str or not data["episode_id"]:
            raise DrawingError("episode_id must be a non-empty string")
        if type(data["prompt"]) is not str or not isinstance(data["metadata"], dict):
            raise DrawingError("prompt must be a string and metadata an object")
        events_raw = data["events"]
        if not isinstance(events_raw, list) or not 1 <= len(events_raw) <= MAX_EVENTS:
            raise DrawingError(f"events must be a list of 1..{MAX_EVENTS} items")
        events = [DrawingAction.from_dict(item) for item in events_raw]
        _check_events(events, data["origin"])
        return cls(
            data["episode_id"], data["origin"], data["prompt"], canvas["width"], canvas["height"],
            renderer, events, data["metadata"], data["seed"] if is_v2 else 0,
        )  # fmt: skip


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
#   責務: [ _check_events: seq連続・actor・origin整合・終端finishを確認する ]
# }
def _check_events(events: list[DrawingAction], origin: str) -> None:
    for index, event in enumerate(events):
        if event.seq != index:
            raise DrawingError(f"event seq must be {index}, got {event.seq!r}")
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
