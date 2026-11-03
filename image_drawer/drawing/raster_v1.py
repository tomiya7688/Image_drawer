"""旧v1 raster-v1：hard_round_v1の整数だけの線分カバレッジ（互換試験用）。"""

from __future__ import annotations

from collections.abc import Sequence


# {
#   責務: [
#     segment_rect: 線分の候補矩形（キャンバスへクリップ済み）を返す
#   ]
#   処理: [ r=ceil(width_px/2) だけ線分の外接矩形を広げ、キャンバスで切る ]
#   引数: [ start/end: 端点, width_px: 線幅, width/height: キャンバス寸法 ]
#   戻り値: [ rect: (xmin, xmax, ymin, ymax)。空ならNone ]
# }
def segment_rect(
    start: Sequence[int], end: Sequence[int], width_px: int, width: int, height: int
) -> tuple[int, int, int, int] | None:
    radius = (width_px + 1) // 2
    xmin = max(0, min(start[0], end[0]) - radius)
    xmax = min(width - 1, max(start[0], end[0]) + radius)
    ymin = max(0, min(start[1], end[1]) - radius)
    ymax = min(height - 1, max(start[1], end[1]) + radius)
    if xmin > xmax or ymin > ymax:
        return None
    return xmin, xmax, ymin, ymax


# {
#   責務: [
#     rect_area: 候補矩形の面積（なければ0）
#   ]
# }
def rect_area(rect: tuple[int, int, int, int] | None) -> int:
    if rect is None:
        return 0
    return (rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1)


# {
#   責務: [
#     is_within_segment: 画素中心が線分から width_px/2 以内かを整数だけで判定する
#   ]
#   処理: [ 4*N <= width_px^2 * D （境界を含む、浮動小数なし） ]
#   引数: [ px/py: 画素中心, start/end: 線分端点, width_px: 線幅 ]
#   戻り値: [ inside: 範囲内ならTrue ]
# }
def is_within_segment(
    px: int, py: int, start: Sequence[int], end: Sequence[int], width_px: int
) -> bool:
    ux, uy = px - start[0], py - start[1]
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = vx * vx + vy * vy
    dot = ux * vx + uy * vy
    if length == 0 or dot <= 0:
        numerator, denominator = ux * ux + uy * uy, 1
    elif dot >= length:
        rx, ry = px - end[0], py - end[1]
        numerator, denominator = rx * rx + ry * ry, 1
    else:
        numerator, denominator = (ux * vy - uy * vx) ** 2, length
    return 4 * numerator <= width_px * width_px * denominator


# {
#   責務: [
#     covered_pixels: 折れ線が覆う画素集合（線分の和集合）を求める
#   ]
#   処理: [ 各線分の候補矩形内だけを判定し、同じ画素は1回だけ数える ]
#   引数: [ points: [x,y]の列, width_px: 線幅, width/height: キャンバス寸法 ]
#   戻り値: [ pixels: 行優先順に整列した(x, y)のリスト ]
# }
def covered_pixels(
    points: Sequence[Sequence[int]], width_px: int, width: int, height: int
) -> list[tuple[int, int]]:
    segments = list(zip(points, points[1:])) if len(points) > 1 else [(points[0], points[0])]
    covered: set[tuple[int, int]] = set()
    for start, end in segments:
        rect = segment_rect(start, end, width_px, width, height)
        if rect is None:
            continue
        for y in range(rect[2], rect[3] + 1):
            for x in range(rect[0], rect[1] + 1):
                if is_within_segment(x, y, start, end, width_px):
                    covered.add((x, y))
    return sorted(covered, key=lambda pixel: (pixel[1], pixel[0]))
