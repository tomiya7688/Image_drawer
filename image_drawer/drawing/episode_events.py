"""v2 Episodeのevent 1件ごとの型・値域・payloadのキー集合の検査（JSON Schemaと同じ契約）。

jsonschemaは開発用依存のため、製品コードでは同じ契約を純Pythonで実装する。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import (
    LAYER_ROLES_Z_ORDER,
    MAX_SEQ,
    MAX_TIME_MS,
    REVIEW_REASONS,
    STAGE_IDS,
    DrawingAction,
)
from .validation import (
    IDENTIFIER_PATTERN,
    DrawingError,
    EpisodeFormatError,
    require_exact_keys,
    require_int,
)

BRUSH_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
MAX_SAMPLES = 4096
MAX_SAMPLE_Q8 = 261888
MAX_SAMPLE_TIME_US = 60_000_000
SAMPLE_KEYS = {"x_q8", "y_q8", "time_us", "pressure_level_id", "tilt_x_level_id", "tilt_y_level_id"}
MODEL_TYPES = (
    "create_layer", "select_layer", "select_color", "select_brush", "set_brush_params",
    "stroke_v2", "erase_v2", "advance_stage",
)  # fmt: skip
SYSTEM_TYPES = (
    "create_layer", "set_layer_visibility", "enter_stage", "exit_stage", "clear_active_layer",
    "finish",
)  # fmt: skip
ACTORS_BY_TYPE: dict[str, tuple[str, ...]] = {
    **{name: ("model",) for name in MODEL_TYPES},
    **{name: ("system",) for name in SYSTEM_TYPES},
    "create_layer": ("model", "system"),
}


# {
#   責務: [ _identifier: レイヤーIDの形式を確認する ]
# }
def _identifier(value: Any, name: str) -> None:
    if type(value) is not str or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise DrawingError(f"{name} must be a layer identifier, got {value!r}")


# {
#   責務: [ _create_layer: create_layerのpayload（layer_id・name・role）を確認する ]
# }
def _create_layer(payload: Any) -> None:
    data = require_exact_keys(payload, {"layer_id", "name", "role"}, "payload")
    _identifier(data["layer_id"], "layer_id")
    name = data["name"]
    if type(name) is not str or not 1 <= len(name) <= 128:
        raise DrawingError("layer name must be a string of 1..128 characters")
    if data["role"] not in LAYER_ROLES_Z_ORDER:
        raise DrawingError(f"unknown layer role: {data['role']!r}")


# {
#   責務: [ _select_layer: select_layerのpayloadを確認する ]
# }
def _select_layer(payload: Any) -> None:
    _identifier(require_exact_keys(payload, {"layer_id"}, "payload")["layer_id"], "layer_id")


# {
#   責務: [ _select_color: select_colorのpayload（0..255）を確認する ]
# }
def _select_color(payload: Any) -> None:
    require_int(
        require_exact_keys(payload, {"color_id"}, "payload")["color_id"], "color_id", 0, 255
    )


# {
#   責務: [ _select_brush: select_brushのpayload（小文字・数字・_のID）を確認する ]
# }
def _select_brush(payload: Any) -> None:
    brush_id = require_exact_keys(payload, {"brush_id"}, "payload")["brush_id"]
    valid = (
        type(brush_id) is str
        and 1 <= len(brush_id) <= 64
        and "a" <= brush_id[0] <= "z"
        and set(brush_id) <= BRUSH_ID_CHARS
    )
    if not valid:
        raise DrawingError(f"brush_id has an invalid form: {brush_id!r}")


# {
#   責務: [ _set_brush_params: set_brush_paramsのpayload（段階番号）を確認する ]
# }
def _set_brush_params(payload: Any) -> None:
    data = require_exact_keys(payload, {"size_level_id", "opacity_level_id"}, "payload")
    require_int(data["size_level_id"], "size_level_id", 0, 18)
    require_int(data["opacity_level_id"], "opacity_level_id", 0, 10)


# {
#   責務: [ _optional_level: nullまたは0..maximumの段階番号を確認する ]
# }
def _optional_level(value: Any, name: str, maximum: int) -> None:
    if value is not None:
        require_int(value, name, 0, maximum)


# {
#   責務: [ _samples: stroke_v2/erase_v2のsamples（1〜4096件、各6キー）を確認する ]
# }
def _samples(payload: Any) -> None:
    samples = require_exact_keys(payload, {"samples"}, "payload")["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise DrawingError(f"samples must be a list of 1..{MAX_SAMPLES} items")
    for item in samples:
        sample = require_exact_keys(item, SAMPLE_KEYS, "sample")
        require_int(sample["x_q8"], "x_q8", 0, MAX_SAMPLE_Q8)
        require_int(sample["y_q8"], "y_q8", 0, MAX_SAMPLE_Q8)
        require_int(sample["time_us"], "time_us", 0, MAX_SAMPLE_TIME_US)
        _optional_level(sample["pressure_level_id"], "pressure_level_id", 10)
        _optional_level(sample["tilt_x_level_id"], "tilt_x_level_id", 12)
        _optional_level(sample["tilt_y_level_id"], "tilt_y_level_id", 12)


# {
#   責務: [ _advance_stage: advance_stageのpayload（{}またはreview_reason）を確認する ]
# }
def _advance_stage(payload: Any) -> None:
    if isinstance(payload, dict) and payload:
        reason = require_exact_keys(payload, {"review_reason"}, "payload")["review_reason"]
        if reason not in REVIEW_REASONS:
            raise DrawingError(f"unknown review_reason: {reason!r}")
    else:
        require_exact_keys(payload, set(), "payload")


# {
#   責務: [ _set_layer_visibility: set_layer_visibilityのpayloadを確認する ]
# }
def _set_layer_visibility(payload: Any) -> None:
    data = require_exact_keys(payload, {"layer_id", "visible"}, "payload")
    _identifier(data["layer_id"], "layer_id")
    if type(data["visible"]) is not bool:
        raise DrawingError("visible must be a boolean")


# {
#   責務: [ _stage_payload: enter_stage/exit_stageのpayload（6工程のいずれか）を確認する ]
# }
def _stage_payload(payload: Any) -> None:
    stage_id = require_exact_keys(payload, {"stage_id"}, "payload")["stage_id"]
    if stage_id not in STAGE_IDS:
        raise DrawingError(f"unknown stage_id: {stage_id!r}")


# {
#   責務: [ _empty: 空payloadを確認する ]
# }
def _empty(payload: Any) -> None:
    require_exact_keys(payload, set(), "payload")


PAYLOAD_CHECKS: dict[str, Callable[[Any], None]] = {
    "create_layer": _create_layer,
    "select_layer": _select_layer,
    "select_color": _select_color,
    "select_brush": _select_brush,
    "set_brush_params": _set_brush_params,
    "stroke_v2": _samples,
    "erase_v2": _samples,
    "advance_stage": _advance_stage,
    "set_layer_visibility": _set_layer_visibility,
    "enter_stage": _stage_payload,
    "exit_stage": _stage_payload,
    "clear_active_layer": _empty,
    "finish": _empty,
}


# {
#   責務: [ check_event_shape: event 1件のseq・time_ms・actor・type・payloadを確認する ]
#   エラー: [ DrawingError: 型・値域・キー集合・actorとtypeの組合せの違反 ]
# }
def check_event_shape(event: DrawingAction) -> None:
    require_int(event.seq, "seq", 0, MAX_SEQ)
    require_int(event.time_ms, "time_ms", 0, MAX_TIME_MS)
    allowed = ACTORS_BY_TYPE.get(event.type)
    if allowed is None:
        raise DrawingError(f"unknown event type: {event.type!r}")
    if event.actor not in allowed:
        raise DrawingError(f"{event.type} cannot be recorded with actor {event.actor!r}")
    PAYLOAD_CHECKS[event.type](event.payload)


# {
#   責務: [ check_event_shapes: 全eventの形を確認し、違反を位置付きのEpisodeFormatErrorにする ]
#   エラー: [ EpisodeFormatError: いずれかのeventが契約に反する ]
# }
def check_event_shapes(events: list[DrawingAction]) -> None:
    for event in events:
        try:
            check_event_shape(event)
        except DrawingError as exc:
            raise EpisodeFormatError(f"event {event.seq!r}: {exc}") from exc
