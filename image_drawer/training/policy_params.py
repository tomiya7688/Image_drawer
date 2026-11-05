"""Policyのパラメータ配列の初期化、勾配の蓄積、Adam状態、softmaxの数式。numpyだけ。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

# 固定headの出力数(設計書4.2節)。kind=6種別、color=256色、brush=12、size=19段階、
# opacity=11段階、layer=最大16。ストロークのheadは構成(G・K・長さ段階)に依存するため
# StrokeShape.head_sizes()が返す(design_stroke_action.md)。
HEAD_SIZES: dict[str, int] = {
    "kind": 6,
    "color": 256,
    "brush": 12,
    "size": 19,
    "opacity": 11,
    "layer": 16,
}
ADAM_EPSILON = 1e-8

Params = dict[str, NDArray[np.float64]]


# {
#   責務: [ init_params: 各headの重み(出力数 x 特徴次元)を小さな正規乱数で初期化する ]
#   引数: [
#     feature_dim: 特徴次元D, rng: 初期化専用の乱数, sizes: head名->出力数(固定head+ストロークhead),
#     scale: 標準偏差
#   ]
#   戻り値: [ params: head名->float64の2次元配列 ]
# }
def init_params(
    feature_dim: int,
    rng: np.random.Generator,
    sizes: Mapping[str, int],
    scale: float = 0.01,
) -> Params:
    return {name: rng.normal(0.0, scale, size=(size, feature_dim)) for name, size in sizes.items()}


# {
#   責務: [ masked_softmax: マスクがTrueの要素だけのsoftmax。偽の要素の確率は厳密に0 ]
#   エラー: [ ValueError: 許可された要素が1つも無い ]
# }
def masked_softmax(
    logits: NDArray[np.float64], mask: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    if mask is None:
        mask = np.ones(logits.shape, dtype=bool)
    if not mask.any():
        raise ValueError("許可された選択肢がありません")
    shifted = np.where(mask, logits, -np.inf)
    shifted = shifted - shifted.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


# {
#   責務: [ categorical_entropy: 確率分布のエントロピー(0 log 0 は0) ]
# }
def categorical_entropy(probabilities: NDArray[np.float64]) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum())


# {
#   責務: [
#     PolicyGrads: head名ごとの勾配バッファ。weight x (出力側の勾配) x 特徴 の外積を足し込む
#   ]
# }
class PolicyGrads:
    def __init__(self, params: Mapping[str, NDArray[np.float64]]) -> None:
        self.arrays: Params = {name: np.zeros_like(value) for name, value in params.items()}

    # {
    #   責務: [ add: arrays[name] += outer(output_grad, features) ]
    # }
    def add(
        self, name: str, output_grad: NDArray[np.float64], features: NDArray[np.float64]
    ) -> None:
        self.arrays[name] += np.outer(output_grad, features)

    # {
    #   責務: [ norm: 全headを連結したL2ノルム ]
    # }
    def norm(self) -> float:
        return float(np.sqrt(sum(float((array**2).sum()) for array in self.arrays.values())))

    # {
    #   責務: [ clip: 全体のL2ノルムがmax_normを超える場合だけ縮める。縮める前のノルムを返す ]
    # }
    def clip(self, max_norm: float) -> float:
        norm = self.norm()
        if norm > max_norm:
            for array in self.arrays.values():
                array *= max_norm / norm
        return norm


# {
#   責務: [
#     AdamState: Adamの1次・2次モーメントとステップ数。目的関数を上げる向きの更新量を返す
#   ]
# }
class AdamState:
    def __init__(self, params: Mapping[str, NDArray[np.float64]]) -> None:
        self.m: Params = {name: np.zeros_like(value) for name, value in params.items()}
        self.v: Params = {name: np.zeros_like(value) for name, value in params.items()}
        self.step_count = 0

    # {
    #   責務: [ step: 勾配(目的関数の上昇方向)から更新量を計算し、モーメントを進める ]
    #   戻り値: [ deltas: head名->加算するべき更新量 ]
    # }
    def step(
        self,
        grads: Mapping[str, NDArray[np.float64]],
        learning_rate: float,
        beta1: float,
        beta2: float,
    ) -> Params:
        self.step_count += 1
        correction1 = 1.0 - beta1**self.step_count
        correction2 = 1.0 - beta2**self.step_count
        deltas: Params = {}
        for name, grad in grads.items():
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * grad * grad
            m_hat = self.m[name] / correction1
            v_hat = self.v[name] / correction2
            deltas[name] = learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
        return deltas


# {
#   責務: [ params_sha256: パラメータ配列のSHA-256(名前順、float64のバイト列) ]
# }
def params_sha256(params: Mapping[str, NDArray[np.float64]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(params):
        array = np.ascontiguousarray(params[name], dtype=np.float64)
        digest.update(name.encode("utf-8") + str(array.shape).encode("ascii") + array.tobytes())
    return digest.hexdigest()
