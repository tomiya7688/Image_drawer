"""brush-raster-v2の数値カーネル。整数演算だけで、スタンプ配置と画素alphaを求める。

式の根拠と設計書との差分は docs/developers/drawing_engine_v2.md に記録する。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache
from math import isqrt

from .catalog import U16_MAX, BrushCatalog, BrushPreset
from .validation import DrawingError, round_half_up

MIN_SPACING_Q8 = 64  # スタンプ間隔の下限（1/4px）。極小ブラシで配置数が爆発しないようにする。
AA_HALF_Q12 = 2048  # 縁のアンチエイリアス半幅（0.5px を q12 = 1/4096px で表す）。
MAX_STAMPS_PER_ACTION = 20000
M32 = 0xFFFFFFFF


# {
#   責務: [ Stamp: 配置済みスタンプ1個。座標はq8、径はq8、alphaはu16、rectはクリップ済み候補矩形 ]
# }
@dataclass(frozen=True)
class Stamp:
    cx_q8: int
    cy_q8: int
    diameter_q8: int
    alpha_u16: int
    rect: tuple[int, int, int, int] | None


# {
#   責務: [ sin_cos_q16: 角度(1/100度)の sin/cos を 65536 倍の整数で返す ]
#   処理: [ Decimalのテイラー展開（精度40桁）で求め、四捨五入する。libm差に依存しない ]
# }
@lru_cache(maxsize=None)
def sin_cos_q16(angle_cdeg: int) -> tuple[int, int]:
    pi = Decimal("3.14159265358979323846264338327950288419716939937510")
    with localcontext() as context:
        context.prec = 40
        x = Decimal(angle_cdeg) * pi / Decimal(18000)
        results = []
        for first_term, first_power in ((x, 1), (Decimal(1), 0)):
            term, total, power = first_term, first_term, first_power
            for _ in range(40):
                term = -term * x * x / Decimal((power + 1) * (power + 2))
                power += 2
                total += term
            results.append(int((total * 65536).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)))
    return results[0], results[1]


# {
#   責務: [ grain_hash16: 座標とseedから決定的な16bit擬似乱数を作る（紙目のような固定パターン） ]
# }
def grain_hash16(seed: int, x: int, y: int) -> int:
    value = ((seed * 0x9E3779B1) ^ (x * 0x85EBCA6B) ^ (y * 0xC2B2AE35)) & M32
    value ^= value >> 16
    value = (value * 0x7FEB352D) & M32
    value ^= value >> 15
    value = (value * 0x846CA68B) & M32
    value ^= value >> 16
    return value & 0xFFFF


# {
#   責務: [ _lerp: 整数の線形補間（0.5は上に丸める）。負の差でも床関数で一意 ]
# }
def _lerp(a: int, b: int, offset: int, length: int) -> int:
    return a + (2 * (b - a) * offset + length) // (2 * length)


# {
#   責務: [ stamp_rect: スタンプが及ぶ画素の候補矩形をキャンバスへクリップして返す ]
#   処理: [ 影響半径=R+0.5px。中心から半径内にある画素中心の最小・最大を整数で求める ]
# }
def stamp_rect(
    cx_q8: int, cy_q8: int, diameter_q8: int, width: int, height: int
) -> tuple[int, int, int, int] | None:
    extent_q8 = diameter_q8 // 2 + AA_HALF_Q12 // 16 + 1
    xmin = max(0, -((-(cx_q8 - extent_q8)) // 256))
    xmax = min(width - 1, (cx_q8 + extent_q8) // 256)
    ymin = max(0, -((-(cy_q8 - extent_q8)) // 256))
    ymax = min(height - 1, (cy_q8 + extent_q8) // 256)
    if xmin > xmax or ymin > ymax:
        return None
    return xmin, xmax, ymin, ymax


# {
#   責務: [ _pressure_scaled: 筆圧によるプリセット最小値から100%への係数を整数で求める ]
#   処理: [ f = min + round((65535-min)*pressure/65535) ]
# }
def _pressure_scaled(minimum_u16: int, pressure_u16: int) -> int:
    return minimum_u16 + round_half_up((U16_MAX - minimum_u16) * pressure_u16, U16_MAX)


# {
#   責務: [ plan_stamps: サンプル列から、ストローク全体のスタンプ列を作る（描画前に全件計算） ]
#   処理: [
#     1: 折れ線の弧長(q8)を整数平方根で求める
#     2: 先頭から spacing 間隔で配置し、終端が漏れるなら終点にも1個置く
#     3: 各スタンプの筆圧を隣接サンプル間で線形補間し、径とalphaを決める
#   ]
#   引数: [
#     samples: 検証済みサンプル(x_q8,y_q8,pressure_level_id), preset: ブラシ, catalog: 段階表,
#     size_level_id/opacity_level_id: 調整値, width/height: キャンバス
#   ]
#   戻り値: [ stamps: Stamp列 ]
#   エラー: [ DrawingError: スタンプ数が上限超過 ]
# }
def plan_stamps(
    samples: list[dict],
    preset: BrushPreset,
    catalog: BrushCatalog,
    size_level_id: int,
    opacity_level_id: int,
    width: int,
    height: int,
) -> list[Stamp]:
    base_diameter = catalog.size_q8(size_level_id)
    base_alpha = catalog.opacity_u16(opacity_level_id)
    spacing = max(MIN_SPACING_Q8, round_half_up(base_diameter * preset.spacing_q16, U16_MAX))
    points = [(s["x_q8"], s["y_q8"], catalog.pressure_u16(s["pressure_level_id"])) for s in samples]

    def make(cx: int, cy: int, pressure: int) -> Stamp:
        scale = _pressure_scaled(preset.pressure_min_size_u16, pressure)
        diameter = max(1, round_half_up(base_diameter * scale, U16_MAX))
        opacity = _pressure_scaled(preset.pressure_min_opacity_u16, pressure)
        alpha = round_half_up(base_alpha * opacity, U16_MAX)
        alpha = round_half_up(alpha * preset.flow_u16, U16_MAX)
        return Stamp(cx, cy, diameter, alpha, stamp_rect(cx, cy, diameter, width, height))

    stamps = [make(*points[0])]
    travelled, next_at = 0, spacing
    for (ax, ay, ap), (bx, by, bp) in zip(points, points[1:]):
        length = isqrt((bx - ax) ** 2 + (by - ay) ** 2)
        while length > 0 and next_at <= travelled + length:
            offset = next_at - travelled
            stamps.append(
                make(
                    _lerp(ax, bx, offset, length),
                    _lerp(ay, by, offset, length),
                    _lerp(ap, bp, offset, length),
                )
            )
            next_at += spacing
            if len(stamps) > MAX_STAMPS_PER_ACTION:
                raise DrawingError(f"stroke needs more than {MAX_STAMPS_PER_ACTION} stamps")
        travelled += length
    if travelled > 0 and next_at - spacing != travelled:
        stamps.append(make(*points[-1]))
    return stamps


# {
#   責務: [ _coverage: 距離r12とAA帯から縁の被覆率(0..65535)を求める ]
#   処理: [ 内側半径以内は全面、外側半径以遠は0、間は線形（soft_roundはsmoothstep） ]
# }
def _coverage(r12: int, inner: int, outer: int, is_soft: bool) -> int:
    if r12 <= inner:
        return U16_MAX
    if r12 >= outer:
        return 0
    ramp = (outer - r12) * U16_MAX // (outer - inner)
    if is_soft:
        return ramp * ramp * (3 * U16_MAX - 2 * ramp) // (U16_MAX * U16_MAX)
    return ramp


# {
#   責務: [ stamp_alphas: スタンプ1個が各画素に与える8bit alphaを列挙する ]
#   処理: [
#     1: tipごとに画素中心までの距離（flatは回転して短軸を引き伸ばす）をq12で求める
#     2: 縁の被覆率に、スタンプalphaとgrainの欠落を掛けて8bitにする
#   ]
#   引数: [ stamp: 配置済みスタンプ, preset: ブラシ, seed: grain用の固定seed ]
#   戻り値: [ (x, y, alpha8) の列。alpha8が0の画素は出さない ]
# }
def stamp_alphas(stamp: Stamp, preset: BrushPreset, seed: int) -> Iterator[tuple[int, int, int]]:
    if stamp.rect is None:
        return
    radius = stamp.diameter_q8 * 8  # 半径をq12で表す。
    outer = radius + AA_HALF_Q12
    inner = min(radius * preset.hardness_u16 // U16_MAX, max(0, radius - AA_HALF_Q12))
    is_flat, is_soft = preset.tip == "flat", preset.tip == "soft_round"
    cos_q16 = sin_q16 = minor = 0
    if is_flat:
        sin_q16, cos_q16 = sin_cos_q16(preset.angle_cdeg)
        minor = max(1024, radius * preset.aspect_u16 // U16_MAX)
    xmin, xmax, ymin, ymax = stamp.rect
    for y in range(ymin, ymax + 1):
        dy = y * 256 - stamp.cy_q8
        for x in range(xmin, xmax + 1):
            dx = x * 256 - stamp.cx_q8
            if is_flat:
                u12 = (dx * cos_q16 + dy * sin_q16) >> 12
                v12 = ((dy * cos_q16 - dx * sin_q16) >> 12) * radius // minor
                squared = u12 * u12 + v12 * v12
                r12 = isqrt(squared)
            else:
                squared = (dx * dx + dy * dy) << 8
                if squared >= outer * outer:
                    continue
                r12 = isqrt(squared)
            coverage = _coverage(r12, inner, outer, is_soft)
            if coverage == 0:
                continue
            if preset.grain_u16 and grain_hash16(seed, x, y) < preset.grain_u16:
                continue
            alpha16 = round_half_up(stamp.alpha_u16 * coverage, U16_MAX)
            alpha8 = round_half_up(alpha16 * 255, U16_MAX)
            if alpha8 > 0:
                yield x, y, alpha8


# {
#   責務: [ paint_stamps: スタンプ列を順に画素へ渡す（スタンプ順に1個ずつ適用） ]
#   引数: [ stamps: 計画済みスタンプ, preset: ブラシ, seed: grain seed, apply: (x,y,alpha8)を受ける関数 ]
# }
def paint_stamps(
    stamps: list[Stamp],
    preset: BrushPreset,
    seed: int,
    apply: Callable[[int, int, int], None],
) -> None:
    for stamp in stamps:
        for x, y, alpha8 in stamp_alphas(stamp, preset, seed):
            apply(x, y, alpha8)
