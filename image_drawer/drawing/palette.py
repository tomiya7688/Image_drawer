"""固定256色パレット idraw-palette-256-v1。画像・ブラシ・モデルによって変わらない。"""

from __future__ import annotations

from typing import Any

from .validation import DrawingError, require_int

PALETTE_ID = "idraw-palette-256-v1"
PALETTE_SIZE = 256


# {
#   責務: [
#     _build_palette: 設計書の計算式から256色の表を一度だけ作る
#   ]
#   処理: [
#     1: 0..215は6x6x6の立方体を 51*(r,g,b) で作る
#     2: 216..255は floor(255*k/41+0.5) の40段階グレーにする（整数演算のみ）
#   ]
#   戻り値: [
#     colors: 長さ256のRGB8タプル
#   ]
# }
def _build_palette() -> tuple[tuple[int, int, int], ...]:
    colors: list[tuple[int, int, int]] = []
    for index in range(216):
        colors.append((51 * (index // 36), 51 * ((index // 6) % 6), 51 * (index % 6)))
    for index in range(216, PALETTE_SIZE):
        step = index - 215
        value = (255 * step * 2 + 41) // 82
        colors.append((value, value, value))
    return tuple(colors)


PALETTE: tuple[tuple[int, int, int], ...] = _build_palette()


# {
#   責務: [
#     palette_rgb: 色番号に対応するRGB8を返す
#   ]
#   引数: [
#     color_id: 0..255のJSON整数
#   ]
#   戻り値: [
#     rgb: (r, g, b)
#   ]
#   エラー: [
#     DrawingError: 真偽値・小数・範囲外
#   ]
# }
def palette_rgb(color_id: Any) -> tuple[int, int, int]:
    require_int(color_id, "color_id", 0, PALETTE_SIZE - 1)
    return PALETTE[color_id]


__all__ = ["PALETTE", "PALETTE_ID", "PALETTE_SIZE", "DrawingError", "palette_rgb"]
