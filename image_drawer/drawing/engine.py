"""決定的なDrawingEngine。Actionを1件ずつ適用し、画素を描画する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import handlers as common
from . import handlers_strokes as strokes
from .catalog import BrushCatalog, default_catalog
from .models import (
    ACTORS,
    LAYER_ROLES_Z_ORDER,
    MAX_EVENTS,
    MAX_SEQ,
    MAX_TIME_MS,
    DrawingAction,
    DrawingState,
    validate_canvas,
)
from .pixels import composite_on_white
from .validation import DrawingError, require_int

RENDERER_V1 = "raster-v1"
RENDERER_V2 = "brush-raster-v2"
PROFILES = (RENDERER_V1, RENDERER_V2)


# {
#   責務: [ RenderedImage: 合成済みの不変なRGBA8画像（alphaは常に255） ]
#   フィールド: [ width, height: 寸法, rgba: 行優先のRGBAバイト列 ]
# }
@dataclass(frozen=True)
class RenderedImage:
    width: int
    height: int
    rgba: bytes

    # {
    #   責務: [ pixel: 画素1個の[r,g,b,a]を返す ]
    # }
    def pixel(self, x: int, y: int) -> list[int]:
        offset = (y * self.width + x) * 4
        return list(self.rgba[offset : offset + 4])

    # {
    #   責務: [ rows: [行][列][r,g,b,a] の入れ子リストで返す ]
    # }
    def rows(self) -> list[list[list[int]]]:
        return [[self.pixel(x, y) for x in range(self.width)] for y in range(self.height)]


# {
#   責務: [
#     Engine: DrawingStateを所有し、Actionの検証・適用・合成を行う唯一の実装
#   ]
#   フィールド: [
#     state: DrawingState, catalog: v2のブラシ・段階表
#   ]
#   処理: [
#     1: apply(action) が1件ずつ検証してから状態を変更する（失敗時は状態不変）
#     2: render() が現在の合成画像を返す。同じAction列は必ず同じ画素になる
#   ]
# }
class Engine:
    def __init__(
        self,
        width: int,
        height: int,
        profile: str = RENDERER_V2,
        seed: int = 0,
        catalog: BrushCatalog | None = None,
    ) -> None:
        validate_canvas(width, height, seed)
        if profile not in PROFILES:
            raise DrawingError(f"unknown renderer profile: {profile!r}")
        self.catalog = catalog if catalog is not None else default_catalog()
        self.state = DrawingState(width, height, profile, seed)

    # {
    #   責務: [ apply: Action1件を検証して適用する ]
    #   処理: [
    #     1: 終了後・件数上限・seq・時刻・actor・typeを確認
    #     2: 種別ごとの処理を実行（処理内で検証後に変更）
    #     3: 成功した場合だけseqと時刻を確定してhistoryへ追加
    #   ]
    #   引数: [ action: 適用するDrawingAction（seq/time_ms省略時は次の値） ]
    #   戻り値: [ applied: seq・time_msを確定したAction ]
    #   エラー: [ DrawingError: 不正なAction（状態は変わらない） ]
    # }
    def apply(self, action: DrawingAction) -> DrawingAction:
        state = self.state
        if state.finished:
            raise DrawingError("episode is finished; no further actions are accepted")
        if len(state.history) >= MAX_EVENTS:
            raise DrawingError(f"event limit exceeded: {MAX_EVENTS}")
        applied = self._resolve_envelope(action)
        self._dispatch(applied.type, applied.payload)
        state.history.append(applied)
        return applied

    # {
    #   責務: [ apply_event: Episodeのevent辞書を読み込んで適用する ]
    # }
    def apply_event(self, event: Any) -> DrawingAction:
        return self.apply(DrawingAction.from_dict(event))

    # {
    #   責務: [ render: 可視レイヤーを白背景へ合成した画像を返す ]
    #   処理: [ v1は作成順、v2はroleの固定z順（同roleは作成順）で下から上へ重ねる ]
    # }
    def render(self) -> RenderedImage:
        layers = [layer for layer in self.state.layers if layer.visible]
        if self.state.profile == RENDERER_V2:
            layers.sort(key=lambda layer: LAYER_ROLES_Z_ORDER.index(layer.role))
        rgba = composite_on_white(
            [layer.pixels for layer in layers], self.state.width, self.state.height
        )
        return RenderedImage(self.state.width, self.state.height, rgba)

    # {
    #   責務: [ layer_rows: レイヤー単体のRGBA画素を入れ子リストで返す ]
    # }
    def layer_rows(self, layer_id: str) -> list[list[list[int]]]:
        return self.state.find_layer(layer_id).pixels.rows()

    # {
    #   責務: [ _resolve_envelope: seq・時刻・actor・typeを検査し、確定したActionを作る ]
    # }
    def _resolve_envelope(self, action: DrawingAction) -> DrawingAction:
        history = self.state.history
        expected_seq = len(history)
        seq = expected_seq if action.seq is None else action.seq
        require_int(seq, "seq", 0, MAX_SEQ)
        if seq != expected_seq:
            raise DrawingError(f"seq must be {expected_seq}, got {seq}")
        previous_time = history[-1].time_ms if history else 0
        time_ms = previous_time if action.time_ms is None else action.time_ms
        require_int(time_ms, "time_ms", 0, MAX_TIME_MS)
        if time_ms < previous_time:
            raise DrawingError("time_ms must be non-decreasing")
        if action.actor not in ACTORS:
            raise DrawingError(f"unknown actor: {action.actor!r}")
        if type(action.type) is not str or not isinstance(action.payload, dict):
            raise DrawingError("action type must be a string and payload an object")
        return DrawingAction(action.type, action.payload, action.actor, seq, time_ms)

    # {
    #   責務: [ _dispatch: 種別を今のprofileで許されるハンドラへ振り分ける ]
    #   エラー: [ DrawingError: 未知の種別、またはそのprofileで使えない種別 ]
    # }
    def _dispatch(self, action_type: str, payload: dict[str, Any]) -> None:
        state, catalog = self.state, self.catalog
        is_v2 = state.profile == RENDERER_V2
        shared = {
            "select_layer": lambda: common.handle_select_layer(state, payload),
            "select_color": lambda: common.handle_select_color(state, payload),
            "finish": lambda: common.handle_finish(state, payload),
            "create_layer": lambda: common.handle_create_layer(state, payload, is_v2),
        }
        v1_only = {
            "set_phase": lambda: common.handle_set_phase(state, payload),
            "select_brush": lambda: strokes.handle_select_brush_v1(state, payload),
            "stroke": lambda: strokes.handle_stroke_v1(state, payload, False),
            "erase": lambda: strokes.handle_stroke_v1(state, payload, True),
        }
        v2_only = {
            "select_brush": lambda: strokes.handle_select_brush_v2(state, payload, catalog),
            "set_brush_params": lambda: strokes.handle_set_brush_params(state, payload, catalog),
            "stroke_v2": lambda: strokes.handle_stroke_v2(state, payload, catalog, False),
            "erase_v2": lambda: strokes.handle_stroke_v2(state, payload, catalog, True),
            "set_layer_visibility": lambda: common.handle_set_layer_visibility(state, payload),
        }
        table = {**shared, **(v2_only if is_v2 else v1_only)}
        handler = table.get(action_type)
        if handler is None:
            raise DrawingError(f"action type {action_type!r} is not accepted by {state.profile}")
        handler()
