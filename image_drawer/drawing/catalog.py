"""ブラシ12種v2と段階表（サイズ・不透明度・筆圧・傾き）の読込と検証。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .validation import DrawingError, require_int

DATA_DIR = Path(__file__).parent / "data"
BRUSH_SET_ID = "brush-v2-core12"
CONTROL_SET_ID = "brush-controls-v2-levels-20261103"
TIPS = ("round", "flat", "soft_round", "grain_round")
MODES = ("stroke", "erase")
U16_MAX = 65535
PRESET_FIELDS = {
    "id", "name_ja", "mode", "tip", "family", "default_size_level_id",
    "default_opacity_level_id", "spacing_q16", "hardness_u16", "flow_u16", "grain_u16",
    "aspect_u16", "angle_cdeg", "pressure_min_size_u16", "pressure_min_opacity_u16",
}  # fmt: skip


# {
#   責務: [
#     BrushPreset: プリセット1種の固定特性。値は全て整数（JSONの値そのまま）
#   ]
#   フィールド: [
#     id, mode, tip, spacing_q16, hardness_u16, flow_u16, grain_u16, aspect_u16,
#     angle_cdeg, pressure_min_size_u16, pressure_min_opacity_u16,
#     default_size_level_id, default_opacity_level_id: manifestと同名
#   ]
# }
@dataclass(frozen=True)
class BrushPreset:
    id: str
    mode: str
    tip: str
    family: str
    default_size_level_id: int
    default_opacity_level_id: int
    spacing_q16: int
    hardness_u16: int
    flow_u16: int
    grain_u16: int
    aspect_u16: int
    angle_cdeg: int
    pressure_min_size_u16: int
    pressure_min_opacity_u16: int


# {
#   責務: [
#     BrushCatalog: 12プリセットと段階表をまとめた読み取り専用の辞書
#   ]
#   フィールド: [
#     presets: JSON配列順のプリセット, size_px/opacity_percent/pressure_percent: 段階表,
#     brush_sha256/control_sha256: canonical JSONのSHA-256
#   ]
# }
@dataclass(frozen=True)
class BrushCatalog:
    presets: tuple[BrushPreset, ...]
    size_px: tuple[int, ...]
    opacity_percent: tuple[int, ...]
    pressure_percent: tuple[int, ...]
    tilt_cdeg: tuple[int, ...]
    pattern_ids: tuple[str, ...]
    brush_sha256: str
    control_sha256: str

    # {
    #   責務: [ get: ブラシIDからプリセットを返す ]
    #   エラー: [ DrawingError: 未知のID ]
    # }
    def get(self, brush_id: Any) -> BrushPreset:
        for preset in self.presets:
            if preset.id == brush_id:
                return preset
        raise DrawingError(f"unknown brush_id: {brush_id!r}")

    # {
    #   責務: [ size_q8: サイズ段階IDを筆先径(q8 = px*256)へ展開する ]
    # }
    def size_q8(self, size_level_id: int) -> int:
        return self.size_px[size_level_id] * 256

    # {
    #   責務: [ opacity_u16: 不透明度段階IDを u16 = floor((percent*65535+50)/100) へ展開する ]
    # }
    def opacity_u16(self, opacity_level_id: int) -> int:
        return (self.opacity_percent[opacity_level_id] * U16_MAX + 50) // 100

    # {
    #   責務: [ pressure_u16: 筆圧段階ID（nullは100%）を u16 へ展開する ]
    # }
    def pressure_u16(self, pressure_level_id: int | None) -> int:
        if pressure_level_id is None:
            return U16_MAX
        return (self.pressure_percent[pressure_level_id] * U16_MAX + 50) // 100


# {
#   責務: [
#     canonical_sha256: 設計書のcanonical JSON規則でSHA-256を求める
#   ]
#   処理: [
#     sort_keys・区切り文字(",",":")・非ASCII保持・NaN禁止のUTF-8を対象にする
#   ]
# }
def canonical_sha256(document: Any) -> str:
    encoded = json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# {
#   責務: [
#     _build_preset: manifestの1要素を検証してBrushPresetにする
#   ]
#   エラー: [ DrawingError: キー集合・値域・列挙値の違反 ]
# }
def _build_preset(raw: dict[str, Any], levels: dict[str, Any]) -> BrushPreset:
    if set(raw) != PRESET_FIELDS:
        raise DrawingError(f"preset keys mismatch: {sorted(set(raw) ^ PRESET_FIELDS)}")
    if raw["mode"] not in MODES or raw["tip"] not in TIPS:
        raise DrawingError(f"invalid mode/tip in preset {raw['id']!r}")
    values: dict[str, Any] = {"id": raw["id"], "mode": raw["mode"], "tip": raw["tip"]}
    values["family"] = raw["family"]
    values["default_size_level_id"] = require_int(
        raw["default_size_level_id"], "default_size_level_id", 0, len(levels["size_px"]) - 1
    )
    values["default_opacity_level_id"] = require_int(
        raw["default_opacity_level_id"],
        "default_opacity_level_id",
        0,
        len(levels["opacity_percent"]) - 1,
    )
    for name, low, high in (
        ("spacing_q16", 1, U16_MAX), ("hardness_u16", 0, U16_MAX), ("flow_u16", 1, U16_MAX),
        ("grain_u16", 0, U16_MAX), ("aspect_u16", 1, U16_MAX), ("angle_cdeg", 0, 35999),
        ("pressure_min_size_u16", 1, U16_MAX), ("pressure_min_opacity_u16", 1, U16_MAX),
    ):  # fmt: skip
        values[name] = require_int(raw[name], name, low, high)
    return BrushPreset(**values)


# {
#   責務: [
#     load_catalog: データJSONを読み、12種・19/11/11/13段階・5パターンを検証して返す
#   ]
#   処理: [
#     1: JSONを読む 2: ID・段階数・一意性・control_set_id一致を確認する
#     3: canonical SHA-256を計算して保持する
#   ]
#   引数: [
#     data_dir: brush_presets_v2.jsonとbrush_control_levels_v2.jsonのあるフォルダ
#   ]
#   戻り値: [
#     catalog: 検証済みのBrushCatalog
#   ]
#   エラー: [
#     DrawingError: 設計と一致しないデータ
#   ]
# }
def load_catalog(data_dir: Path = DATA_DIR) -> BrushCatalog:
    manifest = json.loads((data_dir / "brush_presets_v2.json").read_text(encoding="utf-8"))
    levels = json.loads((data_dir / "brush_control_levels_v2.json").read_text(encoding="utf-8"))
    if manifest["brush_set_id"] != BRUSH_SET_ID or manifest["control_set_id"] != CONTROL_SET_ID:
        raise DrawingError("brush_set_id/control_set_id mismatch")
    if levels["control_set_id"] != CONTROL_SET_ID:
        raise DrawingError("control levels control_set_id mismatch")
    counts = (
        len(manifest["presets"]) == 12,
        len(levels["size_px"]) == 19,
        len(levels["opacity_percent"]) == 11,
        len(levels["pressure_percent"]) == 11,
        len(levels["tilt_cdeg"]) == 13,
        len(levels["pressure_patterns"]) == 5,
    )
    if not all(counts):
        raise DrawingError("catalog level/preset counts do not match the design")
    presets = tuple(_build_preset(item, levels) for item in manifest["presets"])
    if len({preset.id for preset in presets}) != len(presets):
        raise DrawingError("duplicate brush id")
    return BrushCatalog(
        presets=presets,
        size_px=tuple(levels["size_px"]),
        opacity_percent=tuple(levels["opacity_percent"]),
        pressure_percent=tuple(levels["pressure_percent"]),
        tilt_cdeg=tuple(levels["tilt_cdeg"]),
        pattern_ids=tuple(item["id"] for item in levels["pressure_patterns"]),
        brush_sha256=canonical_sha256(manifest),
        control_sha256=canonical_sha256(levels),
    )


# {
#   責務: [ default_catalog: 同梱データのカタログを一度だけ読み込んで共有する ]
# }
@lru_cache(maxsize=1)
def default_catalog() -> BrushCatalog:
    return load_catalog()
