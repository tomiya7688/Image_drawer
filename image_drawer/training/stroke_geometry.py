"""離散ストロークの幾何(格子・方向・長さ・逐次マスク)。numpyだけで、目標画像を知らない。

仕様は docs/developers/design_stroke_action.md 2.1節。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DIRECTIONS = 16  # 22.5度刻み
MODE_NEW, MODE_CONTINUE = 0, 1
DEFAULT_LENGTHS: tuple[int, ...] = (1, 2, 3, 5, 8, 13)
GridPoint = tuple[int, int]
# (head名, 選んだ添字, 許可マスク|None)。log確率とエントロピーの対象になる項。
Term = tuple[str, int, "NDArray[np.bool_] | None"]
Pick = Callable[[str, "NDArray[np.bool_] | None"], int]


# {
#   責務: [ StrokeShape: ストローク表現の構成(格子G、最大線分数K、長さ段階)。head構成hashの元 ]
# }
@dataclass(frozen=True)
class StrokeShape:
    grid: int = 32
    max_segments: int = 4
    lengths: tuple[int, ...] = DEFAULT_LENGTHS

    # {
    #   責務: [ __post_init__: 構成値を検証する ]
    #   エラー: [ ValueError: 整数でない・1未満・長さ段階が空または重複・正でない ]
    # }
    def __post_init__(self) -> None:
        for name in ("grid", "max_segments"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"stroke_{name}は1以上の整数にしてください: {value!r}")
        lengths = tuple(self.lengths)
        if not lengths or len(set(lengths)) != len(lengths):
            raise ValueError("stroke_lengthsは重複のない1個以上の整数にしてください")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in lengths):
            raise ValueError(f"stroke_lengthsは1以上の整数にしてください: {lengths!r}")
        object.__setattr__(self, "lengths", lengths)

    # {
    #   責務: [ head_sizes: ストローク用headの出力数(設計書4節)。dir/lenは線分ごとに別head ]
    # }
    def head_sizes(self) -> dict[str, int]:
        sizes = {
            "stroke_start": self.grid * self.grid,
            "stroke_mode": 2,
            "stroke_segments": self.max_segments,
        }
        for index in range(self.max_segments):
            sizes[f"stroke_dir_{index}"] = DIRECTIONS
            sizes[f"stroke_len_{index}"] = len(self.lengths)
        return sizes

    # {
    #   責務: [ head_hash: head名・出力数・G・K・方向数・長さ段階のSHA-256。checkpointの互換検査に使う ]
    # }
    def head_hash(self) -> str:
        payload = {
            "heads": self.head_sizes(),
            "grid": self.grid,
            "max_segments": self.max_segments,
            "directions": DIRECTIONS,
            "lengths": list(self.lengths),
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# {
#   責務: [ _round_half_away: 0.5を0から遠い方へ丸める ]
# }
def _round_half_away(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


# {
#   責務: [ displacement_table: 方向x長さの整数変位(格子単位)。形状(16, 長さ段階数, 2)=(dx, dy) ]
#   処理: [ θ=d x 22.5度、d=0が+x、yは下向き正の時計回り。変位は(round(L cos θ), round(L sin θ)) ]
# }
def displacement_table(lengths: tuple[int, ...]) -> NDArray[np.int64]:
    table = np.zeros((DIRECTIONS, len(lengths), 2), dtype=np.int64)
    for direction in range(DIRECTIONS):
        theta = direction * 22.5 * math.pi / 180.0
        for index, length in enumerate(lengths):
            table[direction, index] = (
                _round_half_away(length * math.cos(theta)),
                _round_half_away(length * math.sin(theta)),
            )
    return table


# {
#   責務: [ displacement: 方向と長さ段階の添字から整数変位(dx, dy)を返す ]
# }
def displacement(shape: StrokeShape, direction: int, length_index: int) -> GridPoint:
    dx, dy = displacement_table(shape.lengths)[direction, length_index]
    return int(dx), int(dy)


# {
#   責務: [ valid_moves: 位置posから動ける(方向, 長さ段階)の真偽。形状(16, 長さ段階数) ]
#   処理: [ 終点が0<=gx,gy<G(境界上を含む)で、変位が(0,0)でないものだけTrue ]
# }
def valid_moves(shape: StrokeShape, pos: GridPoint) -> NDArray[np.bool_]:
    table = displacement_table(shape.lengths)
    ends_x, ends_y = pos[0] + table[..., 0], pos[1] + table[..., 1]
    inside = (ends_x >= 0) & (ends_x < shape.grid) & (ends_y >= 0) & (ends_y < shape.grid)
    return inside & ((table[..., 0] != 0) | (table[..., 1] != 0))


# {
#   責務: [ start_mask: 1本目の線分を引ける始点マスの真偽(長さG*G、添字はgy*G+gx) ]
# }
def start_mask(shape: StrokeShape) -> NDArray[np.bool_]:
    table = displacement_table(shape.lengths)
    grid = shape.grid
    axis = np.arange(grid)
    gx, gy = np.meshgrid(axis, axis)  # [gy, gx]
    ends_x = gx[..., None, None] + table[..., 0]
    ends_y = gy[..., None, None] + table[..., 1]
    inside = (ends_x >= 0) & (ends_x < grid) & (ends_y >= 0) & (ends_y < grid)
    moving = (table[..., 0] != 0) | (table[..., 1] != 0)
    return (inside & moving).any(axis=(2, 3)).reshape(-1)


# {
#   責務: [ StrokeWalk: 1本のストロークの選択列と、確定した格子点の経路 ]
#   フィールド: [
#     terms: log確率とエントロピーの対象(head名, 添字, マスク)の列
#     mode: 0=new/1=continue, start_cell: newのときだけ始点マス添字
#     requested: segments headで選んだ線分数, dirs/lens: 実際に引いた線分の添字, path: 格子点
#   ]
# }
@dataclass(frozen=True)
class StrokeWalk:
    terms: tuple[Term, ...]
    mode: int
    start_cell: int | None
    requested: int
    dirs: tuple[int, ...]
    lens: tuple[int, ...]
    path: tuple[GridPoint, ...]

    @property
    def segments(self) -> int:
        return len(self.dirs)

    @property
    def truncated(self) -> int:
        return self.requested - len(self.dirs)


# {
#   責務: [ walk_stroke: 選択を逐次に確定し、マスと経路を作る(サンプル・log確率・変換で共通) ]
#   処理: [
#     1: start_mode(continueは直前の終点がある時だけ)
#     2: newならstart、segments
#     3: 線分ごとに、確定済みの現在位置から終点が盤面内の方向、続いてその方向で盤面内の長さだけ許可
#     4: 全候補が無効なら、そこまでの線分で確定する
#   ]
#   引数: [ shape, prev_end: 直前ストロークの終点(なければNone), pick: (head名, マスク)->添字 ]
#   戻り値: [ walk ]
#   エラー: [ ValueError: pickがマスク外・範囲外を返した、線分が0本 ]
# }
def walk_stroke(shape: StrokeShape, prev_end: GridPoint | None, pick: Pick) -> StrokeWalk:
    terms: list[Term] = []

    def take(name: str, mask: NDArray[np.bool_] | None, size: int) -> int:
        index = pick(name, mask)
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"{name}の選択は整数にしてください: {index!r}")
        index = int(index)
        if not 0 <= index < size or (mask is not None and not mask[index]):
            raise ValueError(f"{name}の選択が許可されていません: {index}")
        terms.append((name, index, mask))
        return index

    cells = start_mask(shape)
    can_continue = prev_end is not None and bool(valid_moves(shape, prev_end).any())
    mode = take("stroke_mode", np.array([bool(cells.any()), can_continue]), 2)
    start_cell: int | None = None
    if mode == MODE_NEW:
        start_cell = take("stroke_start", cells, shape.grid * shape.grid)
        pos = (start_cell % shape.grid, start_cell // shape.grid)
    else:
        assert prev_end is not None
        pos = prev_end
    requested = take("stroke_segments", None, shape.max_segments) + 1
    table = displacement_table(shape.lengths)
    path = [pos]
    dirs: list[int] = []
    lens: list[int] = []
    previous: int | None = None
    for index in range(requested):
        valid = valid_moves(shape, pos)
        absolute_ok = valid.any(axis=1)
        if previous is None:
            direction_mask = absolute_ok
        else:  # 相対方向: 添字rは絶対方向(previous+r)%16
            direction_mask = absolute_ok[(previous + np.arange(DIRECTIONS)) % DIRECTIONS]
        if not direction_mask.any():
            break
        relative = take(f"stroke_dir_{index}", direction_mask, DIRECTIONS)
        absolute = relative if previous is None else (previous + relative) % DIRECTIONS
        length_index = take(f"stroke_len_{index}", valid[absolute], len(shape.lengths))
        dx, dy = table[absolute, length_index]
        pos = (pos[0] + int(dx), pos[1] + int(dy))
        path.append(pos)
        dirs.append(relative)
        lens.append(length_index)
        previous = absolute
    if not dirs:
        raise ValueError("引ける線分がありません")
    return StrokeWalk(
        tuple(terms), mode, start_cell, requested, tuple(dirs), tuple(lens), tuple(path)
    )


# {
#   責務: [ replay_pick: 記録済みの選択(ActionChoiceの値)を返すpick。log確率の再計算と変換検証に使う ]
# }
def replay_pick(
    mode: int,
    start_cell: int | None,
    requested: int,
    dirs: tuple[int, ...],
    lens: tuple[int, ...],
) -> Pick:
    def pick(name: str, mask: NDArray[np.bool_] | None) -> int:
        if name == "stroke_mode":
            return mode
        if name == "stroke_start":
            if start_cell is None:
                raise ValueError("stroke_startがありません")
            return start_cell
        if name == "stroke_segments":
            return requested - 1
        kind, _, text = name.rpartition("_")
        values = dirs if kind == "stroke_dir" else lens
        if int(text) >= len(values):
            raise ValueError(f"{name}の選択がありません(線分数の不一致)")
        return values[int(text)]

    return pick
