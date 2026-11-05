"""Policyの選択結果をDrawingActionへ変換する。目標画像を知らない固定の写像。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from image_drawer.drawing import PALETTE, DrawingAction, default_catalog
from image_drawer.drawing.stage import default_stage_table
from image_drawer.training.observation import ACTION_KINDS, Observation
from image_drawer.training.policy_params import HEAD_SIZES

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
#     使う種別でだけ設定, stroke_u: strokeの6個の潜在値(sigmoid前)
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
    stroke_u: tuple[float, ...] | None = None


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
#   責務: [ kind_mask: 選べる種別の真偽(長さ6)。選択可能なレイヤーが無ければselect_layerも不可 ]
# }
def kind_mask(observation: Observation) -> NDArray[np.bool_]:
    legal = set(legal_kinds(observation.stage_id))
    mask = np.array([kind in legal for kind in ACTION_KINDS], dtype=bool)
    if not layer_mask(observation).any():
        mask[ACTION_KINDS.index("select_layer")] = False
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
#   責務: [ build_stroke_samples: 3制御点を等間隔に補間してstroke_v2のsamplesを作る ]
#   処理: [ 座標はsigmoid(u)で[0,1]へ写し、(canvas_side-1)*256のq8整数へ丸める ]
# }
def build_stroke_samples(
    stroke_u: tuple[float, ...], canvas_side: int, sample_count: int, pressure_level: int
) -> list[dict[str, int | None]]:
    if len(stroke_u) != HEAD_SIZES["stroke"] or sample_count < 2:
        raise ValueError("stroke_uは6個、sample_countは2以上にしてください")
    if canvas_side < 1:
        raise ValueError(f"canvas_sideは1以上にしてください: {canvas_side}")
    if not 0 <= pressure_level <= MAX_PRESSURE_LEVEL:
        raise ValueError(f"pressure_levelは0〜{MAX_PRESSURE_LEVEL}にしてください: {pressure_level}")
    latent = np.asarray(stroke_u, dtype=np.float64)
    if not np.isfinite(latent).all():
        raise ValueError("stroke_uにNaNまたは無限大が含まれています")
    unit = 1.0 / (1.0 + np.exp(-np.clip(latent, -50.0, 50.0)))
    points = unit.reshape(3, 2)
    limit = (canvas_side - 1) * 256
    samples: list[dict[str, int | None]] = []
    for index in range(sample_count):
        t = index / (sample_count - 1)
        # 始点->中間->終点を、それぞれ区間の半分ずつで線形補間する。
        a, b, local = (
            (points[0], points[1], 2 * t) if t <= 0.5 else (points[1], points[2], 2 * t - 1)
        )
        x, y = a + (b - a) * local
        samples.append(
            {
                "x_q8": int(np.clip(round(float(x) * limit), 0, limit)),
                "y_q8": int(np.clip(round(float(y) * limit), 0, limit)),
                "time_us": index * STROKE_POINT_STEP_US,
                "pressure_level_id": pressure_level,
                "tilt_x_level_id": None,
                "tilt_y_level_id": None,
            }
        )
    return samples


# {
#   責務: [ decode_action: 選択結果をactor=modelのDrawingActionへ変換する ]
#   処理: [ 種別ごとに必要なheadの値を範囲検査し、固定のpayloadを作る ]
#   引数: [ choice, observation: レイヤーIDの解決に使う, canvas_side, stroke_samples, pressure_level ]
#   戻り値: [ action: seq・time_ms・工程注釈は空(Controllerが補う) ]
#   エラー: [ ValueError: 範囲外の添字、存在しないレイヤー ]
# }
def decode_action(
    choice: ActionChoice,
    observation: Observation,
    canvas_side: int,
    stroke_samples: int,
    pressure_level: int,
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
        return DrawingAction(kind, {"brush_id": BRUSH_IDS[index]})
    if kind == "set_brush_params":
        size = _require_range("size_level", choice.size_level, SIZE_LEVELS)
        opacity = _require_range("opacity_level", choice.opacity_level, OPACITY_LEVELS)
        return DrawingAction(kind, {"size_level_id": size, "opacity_level_id": opacity})
    if kind == "stroke_v2":
        if choice.stroke_u is None:
            raise ValueError("stroke_uがありません")
        samples = build_stroke_samples(choice.stroke_u, canvas_side, stroke_samples, pressure_level)
        return DrawingAction(kind, {"samples": samples})
    return DrawingAction("advance_stage", {})
