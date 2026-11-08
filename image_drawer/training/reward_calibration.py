"""Rewardの特徴重みを、データから自動で校正する(色の間違いに反応するように)。

既定の重みは手で決めており、白地の画像では色の特徴(rgb・rgb_hist)の低下が小さく、
形が正しく色だけ違う画像のRewardがほとんど下がらない(design_trainer.md 12.6)。
ここでは、特徴ごとに「色だけ違う」「白紙」の低下量を目標画像から測り、
色の誤りが白紙と比べて一定以上の低下になるよう、重みを色に敏感な特徴へ寄せる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from image_drawer.training.reward_features import compute_features, parse_feature_name
from image_drawer.training.reward_spec import (
    EMBEDDED_SPEC,
    StageRewardSpec,
    parse_stage_reward_spec,
)

COLOR_RATIO = 0.5  # 色の誤りの低下を、白紙の低下のこの割合以上にする
LAMBDA_STEPS = 10  # 元の重みから色重視の重みへ寄せる割合の刻み(0, 0.1, ..., 1)
_COLOR_KINDS = ("rgb", "rgb_hist")


# {
#   責務: [ FeatureSensitivity: ある特徴の、色の誤りと白紙での平均の低下量(1-類似度) ]
#   フィールド: [ color_drop: 色だけ違う画像での低下, blank_drop: 白紙での低下 ]
# }
class FeatureSensitivity:
    __slots__ = ("color_drop", "blank_drop")

    def __init__(self, color_drop: float, blank_drop: float) -> None:
        self.color_drop = color_drop
        self.blank_drop = blank_drop


# {
#   責務: [ measure_sensitivity: 目標画像群で、特徴ごとの低下量を測る ]
#   処理: [
#     1: 目標のRGBチャンネルを逆順にした画像を「形が正しく色だけ違う」候補とする
#     2: 白紙を「何も描いていない」候補とする
#     3: 各候補と目標の特徴類似度を計算し、1から引いた値を目標全体で平均する
#   ]
#   引数: [ targets: canvas_side正方形のRGB(0〜1), canvas_side, names: 特徴名 ]
#   戻り値: [ 特徴名 -> FeatureSensitivity ]
#   エラー: [ ValueError: targetsが空 ]
# }
def measure_sensitivity(
    targets: Sequence[NDArray[np.float64]], canvas_side: int, names: Sequence[str]
) -> dict[str, FeatureSensitivity]:
    if not targets:
        raise ValueError("校正に使う目標画像がありません")
    color_sum = dict.fromkeys(names, 0.0)
    blank_sum = dict.fromkeys(names, 0.0)
    for target in targets:
        wrong_color = target[:, :, ::-1].copy()  # 色の入れ替え(形は同じ)
        blank = np.ones_like(target)
        color_scores, _ = compute_features(wrong_color, target, canvas_side, list(names))
        blank_scores, _ = compute_features(blank, target, canvas_side, list(names))
        for name in names:
            color_sum[name] += 1.0 - color_scores[name]
            blank_sum[name] += 1.0 - blank_scores[name]
    count = len(targets)
    return {
        name: FeatureSensitivity(color_sum[name] / count, blank_sum[name] / count) for name in names
    }


# {
#   責務: [ calibrate_weights: 1組の重みを、色の誤りが白紙の一定割合以上の低下になるよう寄せる ]
#   処理: [
#     1: 色の特徴(rgb・rgb_hist)が無ければ、元の重みを返す(辺・輝度だけの工程は変えない)
#     2: 色重視の重み = 元の重み x 色の低下量 を、色の特徴だけで合計1に正規化する
#     3: 元の重みから色重視の重みへ0.1刻みで寄せ、色の低下/白紙の低下が COLOR_RATIO 以上になる最小の割合を採る
#     4: 満たす割合が無ければ色重視の重み(割合1)を返す
#   ]
#   戻り値: [ (weights, reached): 重み、COLOR_RATIOに届いたか(色の特徴が無い場合はTrue) ]
# }
def calibrate_weights(
    weights: Mapping[str, float], sensitivity: Mapping[str, FeatureSensitivity]
) -> tuple[dict[str, float], bool]:
    base = {name: float(value) for name, value in weights.items()}
    color_names = [n for n in base if parse_feature_name(n)[0] in _COLOR_KINDS]
    if not color_names:
        return base, True
    raw = {n: (base[n] * sensitivity[n].color_drop if n in color_names else 0.0) for n in base}
    total = sum(raw.values())
    if total <= 0.0:
        return base, False  # 色の低下が全く測れない(色の無い目標だけ)
    focus = {n: raw[n] / total for n in base}

    def ratio(mix: Mapping[str, float]) -> float:
        color = sum(mix[n] * sensitivity[n].color_drop for n in mix)
        blank = sum(mix[n] * sensitivity[n].blank_drop for n in mix)
        return color / blank if blank > 0.0 else 0.0

    for step in range(LAMBDA_STEPS + 1):
        lam = step / LAMBDA_STEPS
        mix = {n: (1.0 - lam) * base[n] + lam * focus[n] for n in base}
        if ratio(mix) >= COLOR_RATIO:
            return _normalised(mix), True
    return _normalised(focus), False


# {
#   責務: [ _normalised: 重みの合計を1にする(0の重みは残す) ]
# }
def _normalised(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


# {
#   責務: [ calibrate_spec: 仕様の全工程とfinal_qualityの重みを校正した新しい仕様を返す ]
#   処理: [
#     1: 仕様に出てくる全特徴の低下量を、目標画像群で1回だけ測る
#     2: 各工程の重みとfinal_quality_weightsに calibrate_weights を適用する
#     3: profile_idに "-calibrated" を付け、JSON文書にしてparse_stage_reward_specで検証・sha256を作る
#   ]
#   引数: [ spec, targets, canvas_side ]
#   戻り値: [ (校正後のStageRewardSpec, report: 工程名 -> {"before","after","reached"}, document: 保存用のJSON辞書) ]
# }
def calibrate_spec(
    spec: StageRewardSpec, targets: Sequence[NDArray[np.float64]], canvas_side: int
) -> tuple[StageRewardSpec, dict[str, Any], dict[str, Any]]:
    names = sorted(
        {n for w in spec.potential_weights.values() for n in w} | set(spec.final_weights)
    )
    sensitivity = measure_sensitivity(targets, canvas_side, names)
    report: dict[str, Any] = {}
    potential: dict[str, dict[str, float]] = {}
    for stage in spec.stage_ids:
        after, reached = calibrate_weights(spec.potential_weights[stage], sensitivity)
        potential[stage] = after
        report[stage] = {"before": dict(spec.potential_weights[stage]), "after": after,
                         "reached": reached}  # fmt: skip
    final, reached = calibrate_weights(spec.final_weights, sensitivity)
    report["final_quality"] = {"before": dict(spec.final_weights), "after": final,
                               "reached": reached}  # fmt: skip
    profile_id = spec.profile_id.removesuffix("-calibrated") + "-calibrated"
    document = {
        **EMBEDDED_SPEC,
        "reward_profile_id": profile_id,
        "stage_ids": list(spec.stage_ids),
        "stage_potential_weights": potential,
        "final_quality_weights": final,
    }
    return parse_stage_reward_spec(document), report, document
