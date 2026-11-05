"""Policyの選択結果をDrawingActionへ変換する。目標画像を知らない固定の写像。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from image_drawer.drawing import PALETTE, DrawingAction, default_catalog
from image_drawer.drawing.stage import default_stage_table
from image_drawer.training.observation import ACTION_KINDS, Observation
from image_drawer.training.policy_params import HEAD_SIZES
from image_drawer.training.stroke_geometry import (
    GridPoint,
    StrokeShape,
    StrokeWalk,
    replay_pick,
    start_mask,
    walk_stroke,
)

BRUSH_IDS: tuple[str, ...] = tuple(preset.id for preset in default_catalog().presets)
SIZE_LEVELS = len(default_catalog().size_px)
OPACITY_LEVELS = len(default_catalog().opacity_percent)
MAX_LAYERS = HEAD_SIZES["layer"]
MAX_PRESSURE_LEVEL = 10  # pressure_level_idの上限(Engineの契約)
STROKE_POINT_STEP_US = 10_000  # サンプル間の時間。time_usは0から非減少にする契約。
assert len(PALETTE) == HEAD_SIZES["color"] and len(BRUSH_IDS) == HEAD_SIZES["brush"]
assert SIZE_LEVELS == HEAD_SIZES["size"] and OPACITY_LEVELS == HEAD_SIZES["opacity"]


# {
#   責務: [
#     ActionChoice: Policyが選んだ種別と、その種別が使うheadの選択結果
#   ]
#   フィールド: [
#     kind_index: ACTION_KINDSの添字, layer_index/color_id/brush_index/size_level/opacity_level:
#     使う種別でだけ設定
#     stroke_mode: 0=new/1=continue, stroke_start: newの始点マス添字(gy*G+gx),
#     stroke_segments: segments headで選んだ線分数(1〜K),
#     stroke_dirs: 実際に引いた線分の方向(1本目は絶対、2本目以降は直前に対する相対),
#     stroke_lens: 実際に引いた線分の長さ段階の添字(len(dirs)==len(lens)<=segments)
#   ]
# }
@dataclass(frozen=True)
class ActionChoice:
    kind_index: int
    layer_index: int | None = None
    color_id: int | None = None
    brush_index: int | None = None
    size_level: int | None = None
    opacity_level: int | None = None
    stroke_mode: int | None = None
    stroke_start: int | None = None
    stroke_segments: int | None = None
    stroke_dirs: tuple[int, ...] | None = None
    stroke_lens: tuple[int, ...] | None = None


# {
#   責務: [ legal_kinds: 工程で選べるAction種別。固定工程表だけで決まり、目標に依存しない ]
#   引数: [ stage_id: 現在の工程 ]
#   戻り値: [ kinds: 選べる種別。finishでは空 ]
# }
def legal_kinds(stage_id: str) -> tuple[str, ...]:
    table = default_stage_table()
    if stage_id not in [spec.id for spec in table.stages]:
        return ()
    allowed = set(table.allowed_model_actions)
    return tuple(kind for kind in ACTION_KINDS if kind in allowed)


# {
#   責務: [ layer_mask: 現在の工程で選択できる(可視かつ書込role)レイヤーの真偽(長さMAX_LAYERS) ]
# }
def layer_mask(observation: Observation) -> NDArray[np.bool_]:
    mask = np.zeros(MAX_LAYERS, dtype=bool)
    table = default_stage_table()
    if observation.stage_id not in [spec.id for spec in table.stages]:
        return mask
    writable = table.spec(observation.stage_id).write_layer_roles
    for index, layer in enumerate(observation.layer_summary[:MAX_LAYERS]):
        mask[index] = layer.visible and layer.role in writable
    return mask


# {
#   責務: [ brush_mask: stroke_v2で使えるブラシ(mode=stroke)の真偽(長さ12)。観測に依らない定数 ]
#   処理: [ 消去ブラシはstroke_v2で必ず拒否されるため選ばせない(erase_v2はv0の対象外) ]
# }
def brush_mask() -> NDArray[np.bool_]:
    return np.array([preset.mode == "stroke" for preset in default_catalog().presets], dtype=bool)


# {
#   責務: [
#     _has_stage_paint: 現在工程の役割レイヤーに、可視で画素が残っているか(観測のlayer_summaryだけ)
#   ]
#   処理: [ 工程入場時に作られる専用レイヤーは空なので、画素があれば受理済みの有効な筆跡がある ]
# }
def _has_stage_paint(observation: Observation, roles: Sequence[str]) -> bool:
    return any(
        layer.visible and layer.has_pixels and layer.role in roles
        for layer in observation.layer_summary
    )


# {
#   責務: [
#     _stroke_since_stage_entry: 直近の受理済みadvance_stage以降に、受理済みのstroke_v2があるか
#   ]
#   処理: [ 工程入場のsystem eventは試行ではないので、前工程を閉じた受理済みadvance_stageで区切る ]
# }
def _stroke_since_stage_entry(observation: Observation) -> bool:
    for brief in reversed(observation.action_tail):
        if brief.accepted and brief.kind == "advance_stage":
            return False
        if brief.accepted and brief.kind == "stroke_v2":
            return True
    return False


# {
#   責務: [
#     can_advance: 現在工程を通常退出できると観測から言えるか。Controllerの退出条件(有効な筆跡と
#     残存画素)のうち、観測にある情報で判定できる範囲だけを見る。必須描画のない工程(refine)は
#     工程入場後の受理済みstrokeを数える
#   ]
# }
def can_advance(observation: Observation) -> bool:
    table = default_stage_table()
    if observation.stage_id not in [spec.id for spec in table.stages]:
        return False
    spec = table.spec(observation.stage_id)
    if spec.min_effectful_strokes == 0:
        return _stroke_since_stage_entry(observation)
    return _has_stage_paint(observation, spec.layer_roles)


# {
#   責務: [
#     kind_mask: 観測だけから合法と分かる種別の真偽(長さ6)。工程表の許可種別に加え、
#     選択可能なレイヤーが無ければselect_layer、ブラシ未選択ならset_brush_params、
#     レイヤー・色・ブラシのいずれかが未選択、または線分を1本も引けない構成ならstroke_v2、
#     工程を抜ける筆跡がまだ無ければadvance_stageを不可にする。
#     select_color・select_brushは常に残るので、全候補が消えることはない
#   ]
# }
def kind_mask(
    observation: Observation, stroke_shape: StrokeShape | None = None
) -> NDArray[np.bool_]:
    legal = set(legal_kinds(observation.stage_id))
    mask = np.array([kind in legal for kind in ACTION_KINDS], dtype=bool)
    tool = observation.tool_state
    if not layer_mask(observation).any():
        mask[ACTION_KINDS.index("select_layer")] = False
    if not tool.brush_selected:
        mask[ACTION_KINDS.index("set_brush_params")] = False
    ready = tool.layer_selected and tool.color_selected and tool.brush_selected
    if not ready or not start_mask(stroke_shape or StrokeShape()).any():
        mask[ACTION_KINDS.index("stroke_v2")] = False
    if mask[ACTION_KINDS.index("advance_stage")] and not can_advance(observation):
        mask[ACTION_KINDS.index("advance_stage")] = False
    return mask


# {
#   責務: [ _require_range: 添字が範囲内の整数であることを確認する ]
#   エラー: [ ValueError: Noneまたは範囲外 ]
# }
def _require_range(name: str, value: int | None, size: int) -> int:
    if value is None or isinstance(value, bool) or not 0 <= value < size:
        raise ValueError(f"{name}は0〜{size - 1}の整数にしてください: {value!r}")
    return int(value)


# {
#   責務: [ grid_to_q8: 格子点(gx,gy)をマス中心のキャンバス座標(q8の実数)へ写す ]
#   処理: [ x_q8 = (gx+0.5)/G x (canvas_side-1) x 256 ]
# }
def grid_to_q8(point: GridPoint, grid: int, canvas_side: int) -> tuple[float, float]:
    limit = (canvas_side - 1) * 256
    return (point[0] + 0.5) / grid * limit, (point[1] + 0.5) / grid * limit


# {
#   責務: [ _round_half_up: 座標を0.5切り上げで整数にする(環境に依存しない丸め) ]
# }
def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


# {
#   責務: [ _extra_points: 余りの点を長い線分から順に巡回して配る(同じ長さは先の線分から) ]
#   戻り値: [ counts: 線分ごとに追加する内部点の数 ]
# }
def _extra_points(path: Sequence[GridPoint], extra: int) -> list[int]:
    count = len(path) - 1
    lengths = [math.dist(path[i], path[i + 1]) for i in range(count)]
    order = sorted(range(count), key=lambda i: (-lengths[i], i))
    counts = [0] * count
    for n in range(extra):
        counts[order[n % count]] += 1
    return counts


# {
#   責務: [ build_stroke_samples: 格子点の経路を等間隔に補間してstroke_v2のsamplesを作る ]
#   処理: [
#     各点をマス中心のq8座標へ写し、線分ごとに端点間を等間隔に補間する。連続する線分の
#     共有端点は1点にまとめ、総数をsample_countにそろえる(余りは長い線分から配る)
#   ]
#   引数: [ path: 始点と各線分の終点(2点以上), shape, canvas_side, sample_count, pressure_level ]
#   エラー: [ ValueError: 経路が短い・格子外・sample_countが線分数+1未満・引数の範囲外 ]
# }
def build_stroke_samples(
    path: Sequence[GridPoint],
    shape: StrokeShape,
    canvas_side: int,
    sample_count: int,
    pressure_level: int,
) -> list[dict[str, int | None]]:
    if canvas_side < 1:
        raise ValueError(f"canvas_sideは1以上にしてください: {canvas_side}")
    if not 0 <= pressure_level <= MAX_PRESSURE_LEVEL:
        raise ValueError(f"pressure_levelは0〜{MAX_PRESSURE_LEVEL}にしてください: {pressure_level}")
    if len(path) < 2 or len(path) - 1 > shape.max_segments:
        raise ValueError(f"経路は2点以上、線分数{shape.max_segments}以下にしてください")
    if any(not (0 <= gx < shape.grid and 0 <= gy < shape.grid) for gx, gy in path):
        raise ValueError("経路が格子の外にあります")
    if sample_count < len(path):
        raise ValueError(f"sample_countは線分数+1({len(path)})以上にしてください: {sample_count}")
    limit = (canvas_side - 1) * 256
    vertices = [grid_to_q8(point, shape.grid, canvas_side) for point in path]
    inner = _extra_points(path, sample_count - len(path))
    points: list[tuple[float, float]] = [vertices[0]]
    for index, (a, b) in enumerate(zip(vertices, vertices[1:])):
        steps = inner[index] + 1
        for n in range(1, steps + 1):
            t = n / steps
            points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return [
        {
            "x_q8": min(max(_round_half_up(x), 0), limit),
            "y_q8": min(max(_round_half_up(y), 0), limit),
            "time_us": index * STROKE_POINT_STEP_US,
            "pressure_level_id": pressure_level,
            "tilt_x_level_id": None,
            "tilt_y_level_id": None,
        }
        for index, (x, y) in enumerate(points)
    ]


# {
#   責務: [ resolve_stroke: ActionChoiceのストローク選択を逐次マスで検証し、経路を再現する ]
#   エラー: [ ValueError: 欠落・範囲外・マスされた選択・線分数の不一致・continueで直前の終点なし ]
# }
def resolve_stroke(
    choice: ActionChoice, shape: StrokeShape, prev_end: GridPoint | None
) -> StrokeWalk:
    if (
        choice.stroke_mode is None
        or choice.stroke_segments is None
        or choice.stroke_dirs is None
        or choice.stroke_lens is None
    ):
        raise ValueError("ストロークの選択が足りません")
    if len(choice.stroke_dirs) != len(choice.stroke_lens):
        raise ValueError("stroke_dirsとstroke_lensの長さが違います")
    if not 1 <= choice.stroke_segments <= shape.max_segments:
        raise ValueError(f"stroke_segmentsは1〜{shape.max_segments}にしてください")
    pick = replay_pick(
        choice.stroke_mode,
        choice.stroke_start,
        choice.stroke_segments,
        tuple(choice.stroke_dirs),
        tuple(choice.stroke_lens),
    )
    walked = walk_stroke(shape, prev_end, pick)
    if walked.dirs != tuple(choice.stroke_dirs) or walked.lens != tuple(choice.stroke_lens):
        raise ValueError("線分の選択が逐次マスの結果と一致しません")
    return walked


# {
#   責務: [ decode_action: 選択結果をactor=modelのDrawingActionへ変換する ]
#   処理: [ 種別ごとに必要なheadの値を範囲検査し、固定のpayloadを作る ]
#   引数: [
#     choice, observation: レイヤーIDの解決に使う, canvas_side, stroke_samples, pressure_level,
#     stroke_shape: ストローク構成(既定は標準構成), stroke_prev_end: 直前ストロークの終点(continue用)
#   ]
#   戻り値: [ action: seq・time_ms・工程注釈は空(Controllerが補う) ]
#   エラー: [ ValueError: 範囲外の添字、存在しないレイヤー ]
# }
def decode_action(
    choice: ActionChoice,
    observation: Observation,
    canvas_side: int,
    stroke_samples: int,
    pressure_level: int,
    stroke_shape: StrokeShape | None = None,
    stroke_prev_end: GridPoint | None = None,
) -> DrawingAction:
    if canvas_side < 1:
        raise ValueError(f"canvas_sideは1以上にしてください: {canvas_side}")
    kind = ACTION_KINDS[_require_range("kind_index", choice.kind_index, len(ACTION_KINDS))]
    if kind == "select_layer":
        index = _require_range("layer_index", choice.layer_index, MAX_LAYERS)
        # Policyがマスクで除外した(不可視・書込不可の)レイヤーは変換でも拒否する。
        if index >= len(observation.layer_summary) or not layer_mask(observation)[index]:
            raise ValueError(f"layer_indexに対応する選択可能なレイヤーがありません: {index}")
        return DrawingAction(kind, {"layer_id": observation.layer_summary[index].layer_id})
    if kind == "select_color":
        return DrawingAction(
            kind, {"color_id": _require_range("color_id", choice.color_id, len(PALETTE))}
        )
    if kind == "select_brush":
        index = _require_range("brush_index", choice.brush_index, len(BRUSH_IDS))
        if not brush_mask()[index]:
            raise ValueError(f"brush_indexはstroke_v2で使えないブラシです: {index}")
        return DrawingAction(kind, {"brush_id": BRUSH_IDS[index]})
    if kind == "set_brush_params":
        size = _require_range("size_level", choice.size_level, SIZE_LEVELS)
        opacity = _require_range("opacity_level", choice.opacity_level, OPACITY_LEVELS)
        return DrawingAction(kind, {"size_level_id": size, "opacity_level_id": opacity})
    if kind == "stroke_v2":
        shape = stroke_shape or StrokeShape()
        walked = resolve_stroke(choice, shape, stroke_prev_end)
        samples = build_stroke_samples(
            walked.path, shape, canvas_side, stroke_samples, pressure_level
        )
        return DrawingAction(kind, {"samples": samples})
    return DrawingAction("advance_stage", {})
