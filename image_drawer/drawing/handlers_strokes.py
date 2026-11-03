"""ブラシ選択・調整とstroke/eraseのAction処理（v1互換とv2）。"""

from __future__ import annotations

from typing import Any

from . import raster_v1, raster_v2
from .catalog import BrushCatalog
from .handlers import charge_budget, writable_layer
from .models import DrawingState
from .palette import palette_rgb
from .validation import DrawingError, require_exact_keys, require_int

V1_BRUSH_ID = "hard_round_v1"
SAMPLE_KEYS = {"x_q8", "y_q8", "time_us", "pressure_level_id", "tilt_x_level_id", "tilt_y_level_id"}
MAX_SAMPLES = 4096
MAX_TIME_US = 60_000_000


# {
#   責務: [ handle_select_brush_v1: 旧v1の唯一のブラシ hard_round_v1 を選ぶ ]
# }
def handle_select_brush_v1(state: DrawingState, payload: Any) -> None:
    data = require_exact_keys(payload, {"brush_id"}, "select_brush.payload")
    if data["brush_id"] != V1_BRUSH_ID:
        raise DrawingError(f"unknown v1 brush_id: {data['brush_id']!r}")
    state.selected_brush_id = V1_BRUSH_ID


# {
#   責務: [ handle_select_brush_v2: v2プリセットを選び、サイズ・不透明度をそのdefaultへ戻す ]
# }
def handle_select_brush_v2(state: DrawingState, payload: Any, catalog: BrushCatalog) -> None:
    data = require_exact_keys(payload, {"brush_id"}, "select_brush.payload")
    preset = catalog.get(data["brush_id"])
    state.selected_brush_id = preset.id
    state.size_level_id = preset.default_size_level_id
    state.opacity_level_id = preset.default_opacity_level_id


# {
#   責務: [ handle_set_brush_params: 選択済みブラシのサイズ・不透明度段階を同時に更新する ]
#   エラー: [ DrawingError: ブラシ未選択・段階ID範囲外 ]
# }
def handle_set_brush_params(state: DrawingState, payload: Any, catalog: BrushCatalog) -> None:
    data = require_exact_keys(
        payload, {"size_level_id", "opacity_level_id"}, "set_brush_params.payload"
    )
    if state.selected_brush_id is None:
        raise DrawingError("select_brush is required before set_brush_params")
    size = require_int(data["size_level_id"], "size_level_id", 0, len(catalog.size_px) - 1)
    opacity = require_int(
        data["opacity_level_id"], "opacity_level_id", 0, len(catalog.opacity_percent) - 1
    )
    state.size_level_id, state.opacity_level_id = size, opacity


# {
#   責務: [ parse_points_v1: v1のpoints配列を検証して返す ]
# }
def parse_points_v1(raw: Any, width: int, height: int) -> list[list[int]]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SAMPLES:
        raise DrawingError(f"points must be a list of 1..{MAX_SAMPLES} items")
    points = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise DrawingError("each point must be [x, y]")
        points.append(
            [require_int(item[0], "x", 0, width - 1), require_int(item[1], "y", 0, height - 1)]
        )
    return points


# {
#   責務: [ handle_stroke_v1: 旧v1のstroke/eraseを描く。重なりは1画素1回だけ合成する ]
#   処理: [ 1: 前提（層・色・ブラシ・工程） 2: payload検証 3: 予算確認 4: 画素へ適用 ]
#   引数: [ state: 状態, payload: points/width_px(/opacity_u8), is_erase: 消去か ]
# }
def handle_stroke_v1(state: DrawingState, payload: Any, is_erase: bool) -> None:
    name = "erase" if is_erase else "stroke"
    keys = {"points", "width_px"} if is_erase else {"points", "width_px", "opacity_u8"}
    data = require_exact_keys(payload, keys, f"{name}.payload")
    layer = writable_layer(state)
    if state.phase is None or state.selected_brush_id is None:
        raise DrawingError(f"{name} requires phase and brush")
    if not is_erase and state.selected_color_id is None:
        raise DrawingError("stroke requires a selected color")
    points = parse_points_v1(data["points"], state.width, state.height)
    width_px = require_int(data["width_px"], "width_px", 1, 64)
    opacity = 0 if is_erase else require_int(data["opacity_u8"], "opacity_u8", 1, 255)
    segments = list(zip(points, points[1:])) if len(points) > 1 else [(points[0], points[0])]
    area = sum(
        raster_v1.rect_area(raster_v1.segment_rect(a, b, width_px, state.width, state.height))
        for a, b in segments
    )
    # 色の解決も含め、失敗し得る処理を全て予算の課金より前に済ませる(失敗時に状態を変えない)。
    rgb = None if is_erase else palette_rgb(state.selected_color_id)
    charge_budget(state, len(points), area)
    pixels = raster_v1.covered_pixels(points, width_px, state.width, state.height)
    if is_erase:
        for x, y in pixels:
            layer.pixels.clear_pixel(x, y)
        return
    for x, y in pixels:
        layer.pixels.paint_over(x, y, rgb, opacity)


# {
#   責務: [ parse_samples_v2: stroke_v2のsamplesを検証して正規化する ]
#   処理: [ 座標範囲・時間の非減少・筆圧ID・傾き（両方nullか両方0..12）を確認する ]
# }
def parse_samples_v2(raw: Any, state: DrawingState, catalog: BrushCatalog) -> list[dict]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SAMPLES:
        raise DrawingError(f"samples must be a list of 1..{MAX_SAMPLES} items")
    samples, previous_time = [], 0
    for index, item in enumerate(raw):
        sample = dict(require_exact_keys(item, SAMPLE_KEYS, "sample"))
        require_int(sample["x_q8"], "x_q8", 0, (state.width - 1) * 256)
        require_int(sample["y_q8"], "y_q8", 0, (state.height - 1) * 256)
        time_us = require_int(sample["time_us"], "time_us", 0, MAX_TIME_US)
        if (index == 0 and time_us != 0) or time_us < previous_time:
            raise DrawingError("time_us must start at 0 and be non-decreasing")
        previous_time = time_us
        if sample["pressure_level_id"] is not None:
            require_int(sample["pressure_level_id"], "pressure_level_id", 0, 10)
        tilts = (sample["tilt_x_level_id"], sample["tilt_y_level_id"])
        if (tilts[0] is None) != (tilts[1] is None):
            raise DrawingError("tilt_x_level_id and tilt_y_level_id must both be null or set")
        for tilt in tilts:
            if tilt is not None:
                require_int(tilt, "tilt_level_id", 0, len(catalog.tilt_cdeg) - 1)
        samples.append(sample)
    return samples


# {
#   責務: [ handle_stroke_v2: stroke_v2/erase_v2をスタンプ列として描く ]
#   処理: [
#     1: 前提（層・ブラシmode・色）を確認 2: samples検証 3: 全スタンプと予算を描画前に計算
#     4: スタンプ順にsource-over（消去はdestination-out）で適用
#   ]
#   引数: [ state: 状態, payload: samples, catalog: 段階表とプリセット, is_erase: 消去か ]
# }
def handle_stroke_v2(
    state: DrawingState, payload: Any, catalog: BrushCatalog, is_erase: bool
) -> None:
    name = "erase_v2" if is_erase else "stroke_v2"
    data = require_exact_keys(payload, {"samples"}, f"{name}.payload")
    layer = writable_layer(state)
    if state.selected_brush_id is None:
        raise DrawingError(f"{name} requires a selected brush")
    preset = catalog.get(state.selected_brush_id)
    if (preset.mode == "erase") != is_erase:
        raise DrawingError(f"brush {preset.id!r} (mode={preset.mode}) cannot be used by {name}")
    if not is_erase and state.selected_color_id is None:
        raise DrawingError("stroke_v2 requires a selected color")
    samples = parse_samples_v2(data["samples"], state, catalog)
    stamps = raster_v2.plan_stamps(
        samples, preset, catalog, state.size_level_id, state.opacity_level_id,
        state.width, state.height,
    )  # fmt: skip
    area = sum(raster_v1.rect_area(stamp.rect) for stamp in stamps)
    # 色の解決も含め、失敗し得る処理を全て予算の課金より前に済ませる(失敗時に状態を変えない)。
    rgb = None if is_erase else palette_rgb(state.selected_color_id)
    charge_budget(state, len(samples), area)
    if is_erase:
        raster_v2.paint_stamps(stamps, preset, state.seed, layer.pixels.erase_alpha)
        return
    raster_v2.paint_stamps(
        stamps, preset, state.seed, lambda x, y, alpha: layer.pixels.paint_over(x, y, rgb, alpha)
    )
