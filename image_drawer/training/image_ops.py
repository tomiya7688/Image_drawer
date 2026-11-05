"""画像の縮小など、Policy側とReward側の両方が使う純関数(目標画像を知らない)。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# {
#   責務: [_box_weights: 長さsourceをtargetへ縮める面積重み行列(target, source)を作る]
#   処理: [各出力セルが覆う入力区間の重なり幅を重みにし、行ごとに和を1へ正規化する]
# }
def _box_weights(source: int, target: int) -> NDArray[np.float64]:
    scale = source / target
    weights = np.zeros((target, source), dtype=np.float64)
    for row in range(target):
        low, high = row * scale, (row + 1) * scale
        for index in range(int(np.floor(low)), min(int(np.ceil(high)), source)):
            weights[row, index] = min(high, index + 1) - max(low, index)
    return weights / weights.sum(axis=1, keepdims=True)


# {
#   責務: [
#     box_downscale: 画像を箱フィルタ(面積平均)でside x sideへ縮める
#   ]
#   引数: [image: (H, W, C)またはH==Wの(H, W), side: 出力一辺(H以下)]
#   戻り値: [scaled: (side, side, ...)のfloat64]
#   エラー: [ValueError: 形状が不正、またはsideが入力より大きい]
# }
def box_downscale(image: NDArray[np.float64], side: int) -> NDArray[np.float64]:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim not in (2, 3) or array.shape[0] != array.shape[1]:
        raise ValueError(f"正方形の画像を渡してください: shape={array.shape}")
    if not 1 <= side <= array.shape[0]:
        raise ValueError(f"sideは1〜{array.shape[0]}にしてください: {side}")
    weights = _box_weights(array.shape[0], side)
    # 行方向、列方向の順に同じ重みを掛ける(分離可能な箱フィルタ)。
    rows = np.tensordot(weights, array, axes=([1], [0]))
    return np.moveaxis(np.tensordot(weights, rows, axes=([1], [1])), 0, 1)
