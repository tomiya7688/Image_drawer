"""画像品質・対照の指標。目標画像を読むのは評価器だけで、Policyの経路からは参照されない。

画像品質はRewardの成分(final_quality_weightsの特徴)ごとに別の指標として返し、重み付き合計を作らない。
独立指標(heldout_*)は学習のRewardと異なる解像度・特徴で計算する(未決事項E1の決定は設計書11.1節)。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from image_drawer.drawing.palette import palette_rgb
from image_drawer.training.reward_features import compute_features

Image = NDArray[np.float64]
HELDOUT_SIDE = 64
SSIM_BLOCK = 8
HELDOUT_REQUIRED = ("heldout_ssim64",)
# E1で任意とした指標。拡張(openclip extra)と重みが無ければ測らず、報告書のunverifiedへ書く。
HELDOUT_OPTIONAL = ("heldout_openclip_image", "heldout_openclip_text")
BACKGROUND_DISTANCE = 12.0


# {
#   責務: [
#     component_scores: Rewardのfinal_quality成分を、成分ごとの別の値で返す
#   ]
#   引数: [
#     rgb: 完成画像(白背景合成のRGB, 0〜1), target_rgb: 目標画像(同じ大きさ), weights: 成分名の集合(final_weights)
#   ]
#   戻り値: [
#     scores: 成分名 -> 0〜1の得点。重み付き合計は作らない
#   ]
# }
def component_scores(rgb: Image, target_rgb: Image, components: Sequence[str]) -> dict[str, float]:
    scores, _ = compute_features(rgb, target_rgb, rgb.shape[0], list(components))
    return {name: float(scores[name]) for name in components}


# {
#   責務: [ _resize_box: 正方形画像を箱フィルタで一辺sideへ縮小または拡大する ]
# }
def _resize_box(image: Image, side: int) -> Image:
    from PIL import Image as PilImage

    pixels = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    resized = PilImage.fromarray(pixels).resize((side, side), PilImage.Resampling.BOX)
    return np.asarray(resized, dtype=np.float64) / 255.0


# {
#   責務: [
#     ssim64: 64x64の輝度での構造的類似度(SSIM)。学習のRewardと異なる解像度・指標
#   ]
#   処理: [ 8x8の重ならないブロックごとにSSIMを求めて平均する。同一画像は1 ]
# }
def ssim64(rgb: Image, target_rgb: Image) -> float:
    weights = np.array([0.2126, 0.7152, 0.0722])
    first = _resize_box(rgb, HELDOUT_SIDE) @ weights
    second = _resize_box(target_rgb, HELDOUT_SIDE) @ weights
    count = HELDOUT_SIDE // SSIM_BLOCK

    def blocks(array: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            array.reshape(count, SSIM_BLOCK, count, SSIM_BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(count * count, -1)
        )

    a, b = blocks(first), blocks(second)
    mean_a, mean_b = a.mean(axis=1), b.mean(axis=1)
    var_a, var_b = a.var(axis=1), b.var(axis=1)
    covariance = ((a - mean_a[:, None]) * (b - mean_b[:, None])).mean(axis=1)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2)) / (
        (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim.mean())


# {
#   責務: [ heldout_scores: 必須の独立指標(heldout_*)を返す。任意の指標は含めない ]
# }
def heldout_scores(rgb: Image, target_rgb: Image) -> dict[str, float]:
    return {"heldout_ssim64": ssim64(rgb, target_rgb)}


# {
#   責務: [ blank_image: 失敗時の得点の基準になる白紙画像 ]
# }
def blank_image(side: int) -> Image:
    return np.ones((side, side, 3), dtype=np.float64)


# {
#   責務: [
#     prompt_shuffle_delta: promptを別の項目のものに入れ替えたときの得点低下の平均
#   ]
#   処理: [
#     各項目で score(正しいprompt) - score(入れ替えたprompt) を求めて平均する。
#     0以下ならpromptを使っていない疑い
#   ]
#   引数: [
#     image_ids: 項目, shuffle_map: image_id -> 入れ替え先のimage_id(自分自身は不可)
#     prompts: image_id -> prompt, generate: (image_id, prompt) -> 完成画像
#     score: (image_id, 完成画像) -> 得点
#   ]
# }
def prompt_shuffle_delta(
    image_ids: Sequence[str],
    shuffle_map: Mapping[str, str],
    prompts: Mapping[str, str],
    generate: Callable[[str, str], Image],
    score: Callable[[str, Image], float],
) -> float:
    if not image_ids or any(shuffle_map[i] == i for i in image_ids):
        raise ValueError(
            "image_ids must be non-empty and shuffle_map must not map an item to itself"
        )
    deltas = [
        score(i, generate(i, prompts[i])) - score(i, generate(i, prompts[shuffle_map[i]]))
        for i in image_ids
    ]
    return float(np.mean(deltas))


# {
#   責務: [
#     nearest_train_hit_rate: 生成画像のうち、trainの目標画像に閾値未満まで近いものの割合(記憶・コピーの疑い)
#   ]
#   引数: [
#     generated: 生成画像, train_targets: trainの目標画像, distance: 2枚の距離, tau: 閾値(validationで設定して凍結)
#   ]
# }
def nearest_train_hit_rate(
    generated: Sequence[Image],
    train_targets: Sequence[Image],
    distance: Callable[[Image, Image], float],
    tau: float,
) -> float:
    if not generated or not train_targets:
        raise ValueError("generated and train_targets must not be empty")
    hits = sum(min(distance(g, t) for t in train_targets) < tau for g in generated)
    return hits / len(generated)


# {
#   責務: [
#     nearest_train_distance: 生成画像から、trainの目標画像のうち最も近いものまでの距離(1 - ssim64)
#   ]
#   引数: [ rgb: 生成画像, train_rgbs: trainの目標画像(1枚以上) ]
#   戻り値: [ 最小の距離。同一画像は0 ]
#   エラー: [ ValueError: train_rgbsが空 ]
# }
def nearest_train_distance(rgb: Image, train_rgbs: Sequence[Image]) -> float:
    if not train_rgbs:
        raise ValueError("train_rgbs must not be empty")
    return min(1.0 - ssim64(rgb, target) for target in train_rgbs)


# {
#   責務: [
#     memorization_tau: 記憶・コピーの閾値τを、validationの実画像のtrainへの近さから決める
#   ]
#   処理: [
#     validationの各目標画像について、trainの目標画像までの最小距離を求め、その下側quantile点をτにする。
#     「未見の実画像でも普通は届かない近さ」まで生成画像が近い場合を、コピーの疑いとして数える。
#     validationだけで決めてjsonへ保存して凍結し、testの結果を見て変更しない
#   ]
#   引数: [ validation_rgbs, train_rgbs: 1枚以上の目標画像, quantile: 0〜1(既定0.05) ]
#   エラー: [ ValueError: 画像が空、quantileが範囲外 ]
# }
def memorization_tau(
    validation_rgbs: Sequence[Image], train_rgbs: Sequence[Image], quantile: float = 0.05
) -> float:
    if not validation_rgbs or not train_rgbs:
        raise ValueError("validation_rgbs and train_rgbs must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    distances = [nearest_train_distance(rgb, train_rgbs) for rgb in validation_rgbs]
    return float(np.quantile(distances, quantile))


# {
#   責務: [
#     probe_hit: 指定した色の面積が、背景を除いて最大かを固定パレットの最近傍で判定する
#   ]
#   引数: [ rgb: 完成画像(0〜1), expected_color_id: パレットの色ID(0〜255) ]
# }
def probe_hit(rgb: Image, expected_color_id: int) -> bool:
    palette = np.array([palette_rgb(i) for i in range(256)], dtype=np.float64)
    pixels = (rgb.reshape(-1, 3) * 255.0).round()
    foreground = pixels[np.linalg.norm(pixels - 255.0, axis=1) > BACKGROUND_DISTANCE]
    if len(foreground) == 0:
        return False
    nearest = np.linalg.norm(foreground[:, None, :] - palette[None], axis=2).argmin(axis=1)
    counts = np.bincount(nearest, minlength=256)
    return int(counts.argmax()) == expected_color_id


# {
#   責務: [
#     reward_gain_without_heldout_gain: Rewardの成分は改善したが、独立指標がどれも改善しない警告
#   ]
#   引数: [
#     component_lower_bounds: Reward成分ごとのgain信頼区間の下限
#     heldout_lower_bounds: 独立指標ごとのgain信頼区間の下限
#   ]
# }
def reward_gain_without_heldout_gain(
    component_lower_bounds: Mapping[str, float], heldout_lower_bounds: Mapping[str, float]
) -> bool:
    reward_improved = any(v > 0 for v in component_lower_bounds.values())
    heldout_improved = any(v > 0 for v in heldout_lower_bounds.values())
    return reward_improved and not heldout_improved
