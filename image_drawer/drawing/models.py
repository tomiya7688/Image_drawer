"""DrawingAction・Layer・DrawingStateのシリアライズ可能なデータ型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pixels import PixelBuffer
from .validation import DrawingError, require_exact_keys, require_int

ACTORS = ("human", "model", "system", "synthetic", "derived")
MAX_SEQ = 19999
MAX_TIME_MS = 604_800_000
MAX_EVENTS = 20000
MAX_LAYERS = 16
MAX_CANVAS_SIDE = 1024
MAX_LAYER_AREA = 4_194_304
LAYER_ROLES_Z_ORDER = (
    "rough", "sketch", "base_color", "shade", "final_drawing", "highlight", "correction",
)  # fmt: skip
STAGE_IDS = ("rough", "sketch", "final_drawing", "base_color", "shading", "refine")
FINISH_STAGE_ID = "finish"
REVIEW_REASONS = ("monochrome_line_art_only", "flat_style_no_shading", "no_correction_needed")


# {
#   責務: [
#     EventGroup: 原子的に確定する記録グループ内での位置（グループ番号・順番・件数）
#   ]
#   フィールド: [
#     id: 0始まりのグループ番号, index: グループ内の0始まりの順番, size: グループの件数
#   ]
# }
@dataclass(frozen=True)
class EventGroup:
    id: int
    index: int
    size: int


# {
#   責務: [
#     DrawingAction: 順序付きの1操作（seq・時刻・実行主体・種別・payload）
#   ]
#   フィールド: [
#     type: 操作種別, payload: 操作の引数, actor: 実行主体, seq/time_ms: 省略時はEngineが次の値を補う,
#     stage_id/group: 工程Controllerが付ける注釈（Engineは検査せず保持するだけ。to_dictには含めない）
#   ]
#   処理: [ to_dict/from_dict でEpisodeのevent形式と相互変換する ]
# }
@dataclass(frozen=True)
class DrawingAction:
    type: str
    payload: dict[str, Any]
    actor: str = "model"
    seq: int | None = None
    time_ms: int | None = None
    stage_id: str | None = None
    group: EventGroup | None = None

    # {
    #   責務: [ to_dict: Episodeのevent辞書（5キー）へ変換する ]
    #   エラー: [ DrawingError: seqまたはtime_msが未確定 ]
    # }
    def to_dict(self) -> dict[str, Any]:
        if self.seq is None or self.time_ms is None:
            raise DrawingError("seq and time_ms must be set before serialization")
        return {
            "seq": self.seq,
            "time_ms": self.time_ms,
            "actor": self.actor,
            "type": self.type,
            "payload": self.payload,
        }

    # {
    #   責務: [ from_dict: Episodeのevent辞書を厳密なキー集合で読み込む ]
    #   エラー: [ DrawingError: キー集合・型の不一致 ]
    # }
    @classmethod
    def from_dict(cls, raw: Any) -> DrawingAction:
        data = require_exact_keys(raw, {"seq", "time_ms", "actor", "type", "payload"}, "event")
        if type(data["type"]) is not str or not isinstance(data["payload"], dict):
            raise DrawingError("event type must be a string and payload an object")
        return cls(data["type"], data["payload"], data["actor"], data["seq"], data["time_ms"])


# {
#   責務: [
#     Layer: 名前・役割・可視性・画素を持つ透明RGBAレイヤー1枚
#   ]
#   フィールド: [
#     id, name: 識別子と表示名, role: v2の工程役割（v1はNone）, visible: 合成に含めるか, pixels: 画素
#   ]
# }
@dataclass
class Layer:
    id: str
    name: str
    role: str | None
    pixels: PixelBuffer
    visible: bool = True

    # {
    #   責務: [ to_dict: レイヤーを辞書にする。画素は必要なときだけ含める ]
    # }
    def to_dict(self, include_pixels: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "visible": self.visible,
        }
        if include_pixels:
            data["rgba_rows"] = self.pixels.rows()
        return data


# {
#   責務: [
#     DrawingState: 制作途中の全状態。Engine以外が直接書き換えない
#   ]
#   フィールド: [
#     width/height: 不変のキャンバス寸法, profile: renderer_version, seed: grain用の固定seed,
#     layers: 作成順のレイヤー, active_layer_id/selected_color_id/selected_brush_id/phase: 選択状態,
#     size_level_id/opacity_level_id: v2のブラシ調整値, stage_id: v2の現在の工程（未入場はNone）,
#     finished: 終了済みか, history: 適用済みAction,
#     points_used/area_used: 累積の計算量予算
#   ]
# }
@dataclass
class DrawingState:
    width: int
    height: int
    profile: str
    seed: int
    layers: list[Layer] = field(default_factory=list)
    active_layer_id: str | None = None
    selected_color_id: int | None = None
    selected_brush_id: str | None = None
    size_level_id: int | None = None
    opacity_level_id: int | None = None
    phase: str | None = None
    stage_id: str | None = None
    finished: bool = False
    history: list[DrawingAction] = field(default_factory=list)
    points_used: int = 0
    area_used: int = 0

    # {
    #   責務: [ find_layer: IDからレイヤーを返す。なければ例外 ]
    #   エラー: [ DrawingError: 存在しないレイヤー ]
    # }
    def find_layer(self, layer_id: str) -> Layer:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise DrawingError(f"unknown layer_id: {layer_id!r}")

    # {
    #   責務: [ to_dict: 状態を辞書にする（history・選択状態・レイヤー）。画素は任意 ]
    # }
    def to_dict(self, include_pixels: bool = False) -> dict[str, Any]:
        return {
            "canvas": {"width": self.width, "height": self.height},
            "renderer_version": self.profile,
            "seed": self.seed,
            "layers": [layer.to_dict(include_pixels) for layer in self.layers],
            "active_layer_id": self.active_layer_id,
            "selected_color_id": self.selected_color_id,
            "selected_brush_id": self.selected_brush_id,
            "size_level_id": self.size_level_id,
            "opacity_level_id": self.opacity_level_id,
            "phase": self.phase,
            "finished": self.finished,
            "history": [action.to_dict() for action in self.history],
        }


# {
#   責務: [ validate_canvas: キャンバス寸法とseedを検査する ]
#   エラー: [ DrawingError: 範囲外 ]
# }
def validate_canvas(width: Any, height: Any, seed: Any) -> None:
    require_int(width, "width", 1, MAX_CANVAS_SIDE)
    require_int(height, "height", 1, MAX_CANVAS_SIDE)
    require_int(seed, "seed", 0, 0xFFFFFFFF)
