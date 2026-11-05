"""Rewardの画像特徴(暫定数値化 stage-reward-impl-v0)。numpyだけの純関数。

入力は形状(H, W, 3)・値域[0, 1]のRGB画像です。出力は[0, 1]の類似度(大きいほど良い)です。
縮小は箱フィルタ、輪郭は輝度のSobel大きさを3x3平均した上でのsoft Dice(2ab/(a^2+b^2)形)、
RGB/輝度類似度は 1 - 平均絶対差、ヒストグラムは16bin/チャンネルの交差です(設計書10節)。
"""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from image_drawer.training.image_ops import box_downscale

REWARD_IMPL_VERSION = "stage-reward-impl-v0"
HISTOGRAM_BINS = 16
# stage_reward_v2.jsonのfeaturesと同じ特徴名。未知の名前は拒否する。
FEATURE_NAMES: frozenset[str] = frozenset(
    {"edge_32", "edge_64", "edge_128", "rgb_32", "rgb_128", "luma_64", "rgb_hist_16"}
)
_FEATURE_PATTERN = re.compile(r"^(edge|rgb|luma|rgb_hist)_(\d+)$")
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_SOBEL_NORM = 4.0  # 値域[0,1]の理想的な段差で大きさが4になるため、4で割って1へ飽和させる。
_DICE_EPS = 1e-6


# {
#   責務: [
#     effective_resolution: 要求解像度をキャンバス一辺でcapした実解像度を返す
#   ]
#   引数: [requested: 特徴名が指定する解像度, canvas_side: キャンバス一辺]
#   戻り値: [resolution: min(requested, canvas_side)]
#   エラー: [ValueError: どちらかが1未満]
# }
def effective_resolution(requested: int, canvas_side: int) -> int:
    if requested < 1 or canvas_side < 1:
        raise ValueError(f"解像度は1以上にしてください: {requested}, {canvas_side}")
    return min(requested, canvas_side)


# {
#   責務: [
#     composite_over_white: RGBAバイト列を白背景に合成したfloat RGB画像にする
#   ]
#   処理: [
#     1: canvas_side*canvas_side*4のバイト列を検証して読む
#     2: 不透明度で白と線形に混ぜ、[0,1]へ正規化する
#   ]
#   引数: [rgba: RGBAの生バイト列, canvas_side: キャンバス一辺]
#   戻り値: [image: (canvas_side, canvas_side, 3)のfloat64]
#   エラー: [ValueError: バイト長が canvas_side^2*4 でない]
# }
def composite_over_white(rgba: bytes, canvas_side: int) -> NDArray[np.float64]:
    if len(rgba) != canvas_side * canvas_side * 4:
        raise ValueError(f"RGBAのバイト長が不正です: {len(rgba)} (canvas_side={canvas_side})")
    pixels = np.frombuffer(rgba, dtype=np.uint8).reshape(canvas_side, canvas_side, 4)
    color = pixels[..., :3].astype(np.float64) / 255.0
    alpha = pixels[..., 3:4].astype(np.float64) / 255.0
    return color * alpha + (1.0 - alpha)


# {
#   責務: [luminance: RGB画像をRec.709係数の輝度(H, W)にする]
# }
def luminance(image: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(image, dtype=np.float64) @ _LUMA_WEIGHTS


# {
#   責務: [_neighborhood: 端を複製して3x3近傍の9枚(上下左右の平行移動)を返す]
# }
def _neighborhood(plane: NDArray[np.float64]) -> dict[tuple[int, int], NDArray[np.float64]]:
    padded = np.pad(plane, 1, mode="edge")
    height, width = plane.shape
    return {
        (dy, dx): padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }


# {
#   責務: [
#     sobel_magnitude: 輝度平面のSobel勾配の大きさを[0,1]へ飽和正規化して返す
#   ]
#   処理: [端を複製、gx/gyを3x3カーネルで求め、sqrt(gx^2+gy^2)/4を1でclipする]
# }
def sobel_magnitude(luma: NDArray[np.float64]) -> NDArray[np.float64]:
    near = _neighborhood(np.asarray(luma, dtype=np.float64))
    gx = (near[-1, 1] + 2 * near[0, 1] + near[1, 1]) - (
        near[-1, -1] + 2 * near[0, -1] + near[1, -1]
    )
    gy = (near[1, -1] + 2 * near[1, 0] + near[1, 1]) - (
        near[-1, -1] + 2 * near[-1, 0] + near[-1, 1]
    )
    return np.clip(np.sqrt(gx * gx + gy * gy) / _SOBEL_NORM, 0.0, 1.0)


# {
#   責務: [box3_mean: 端を複製した3x3平均で平面を平滑化する]
# }
def box3_mean(plane: NDArray[np.float64]) -> NDArray[np.float64]:
    near = _neighborhood(np.asarray(plane, dtype=np.float64))
    return sum(near.values()) / 9.0


# {
#   責務: [
#     soft_dice: 非負の2平面のsoft Dice 2*sum(ab)/(sum(a^2)+sum(b^2)) を返す
#   ]
#   処理: [両方が全て0(輪郭なし同士)の場合は eps により1になる]
# }
def soft_dice(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    numerator = 2.0 * float(np.sum(first * second)) + _DICE_EPS
    denominator = float(np.sum(first * first) + np.sum(second * second)) + _DICE_EPS
    return min(1.0, numerator / denominator)


# {
#   責務: [edge_similarity: 縮小後の輝度Sobel輪郭(3x3平均)のsoft Dice類似度を返す]
#   引数: [first/second: (S, S, 3)のRGB, resolution: 実解像度(S以下)]
#   戻り値: [score: 0〜1]
# }
def edge_similarity(
    first: NDArray[np.float64], second: NDArray[np.float64], resolution: int
) -> float:
    maps = [
        box3_mean(sobel_magnitude(luminance(box_downscale(image, resolution))))
        for image in (first, second)
    ]
    return soft_dice(maps[0], maps[1])


# {
#   責務: [rgb_similarity: 縮小後のRGBの 1 - 平均絶対差 を返す]
# }
def rgb_similarity(
    first: NDArray[np.float64], second: NDArray[np.float64], resolution: int
) -> float:
    scaled = [box_downscale(image, resolution) for image in (first, second)]
    return float(1.0 - np.mean(np.abs(scaled[0] - scaled[1])))


# {
#   責務: [luma_similarity: 縮小後の輝度の 1 - 平均絶対差 を返す]
# }
def luma_similarity(
    first: NDArray[np.float64], second: NDArray[np.float64], resolution: int
) -> float:
    scaled = [luminance(box_downscale(image, resolution)) for image in (first, second)]
    return float(1.0 - np.mean(np.abs(scaled[0] - scaled[1])))


# {
#   責務: [
#     rgb_hist_similarity: 16bin/チャンネルの正規化ヒストグラム交差のチャンネル平均を返す
#   ]
#   処理: [値vのbinは min(int(v*16), 15)。縮小せずキャンバスの全画素を使う]
# }
def rgb_hist_similarity(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    scores = []
    for channel in range(3):
        hists = []
        for image in (first, second):
            bins = np.minimum((np.asarray(image)[..., channel] * HISTOGRAM_BINS).astype(int), 15)
            counts = np.bincount(bins.ravel(), minlength=HISTOGRAM_BINS).astype(np.float64)
            hists.append(counts / counts.sum())
        scores.append(float(np.minimum(hists[0], hists[1]).sum()))
    return float(np.mean(scores))


# {
#   責務: [
#     parse_feature_name: 特徴名を(種別, 要求解像度)に分解する。未知の名前は拒否する
#   ]
#   戻り値: [(kind, requested): 例 "edge_64" -> ("edge", 64)。rgb_hist_16は ("rgb_hist", 16)]
#   エラー: [ValueError: FEATURE_NAMESに無い名前]
# }
def parse_feature_name(name: str) -> tuple[str, int]:
    if name not in FEATURE_NAMES:
        raise ValueError(f"未知のReward特徴名です: {name}")
    match = _FEATURE_PATTERN.match(name)
    assert match is not None
    return match.group(1), int(match.group(2))


# {
#   責務: [
#     compute_features: 指定した特徴名の類似度と実解像度をまとめて計算する
#   ]
#   処理: [
#     1: 各画像の形状がcanvas_side正方形であることを検証する
#     2: 特徴名ごとに解像度をcanvas_sideへcapして計算する(rgb_hist_16は解像度を持たない)
#   ]
#   引数: [first: 評価対象の合成RGB, second: 目標RGB, canvas_side: 一辺, names: 特徴名]
#   戻り値: [(scores, resolutions): 特徴名->0〜1, 特徴名->実解像度]
#   エラー: [ValueError: 形状不一致、未知の特徴名]
# }
def compute_features(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    canvas_side: int,
    names: tuple[str, ...] | list[str],
) -> tuple[Mapping[str, float], Mapping[str, int]]:
    expected = (canvas_side, canvas_side, 3)
    if np.shape(first) != expected or np.shape(second) != expected:
        raise ValueError(
            f"画像の形状が{expected}ではありません: {np.shape(first)}, {np.shape(second)}"
        )
    scores: dict[str, float] = {}
    resolutions: dict[str, int] = {}
    for name in names:
        kind, requested = parse_feature_name(name)
        if kind == "rgb_hist":
            scores[name], resolutions[name] = rgb_hist_similarity(first, second), canvas_side
            continue
        resolution = effective_resolution(requested, canvas_side)
        function = {"edge": edge_similarity, "rgb": rgb_similarity, "luma": luma_similarity}[kind]
        scores[name], resolutions[name] = function(first, second, resolution), resolution
    return scores, resolutions
