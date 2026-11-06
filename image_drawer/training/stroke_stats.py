"""受理されたstroke_v2の形(直線率・曲がり角)の集計。学習の記録と評価表で共有する。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from image_drawer.training.action_codec import ActionChoice
from image_drawer.training.stroke_geometry import DIRECTIONS


# {
#   責務: [
#     StrokeStats: 受理されたstrokeの数・1本線の数・切り詰めた線分数・曲がり角(度)の列
#   ]
#   処理: [ 曲がり角は2本目以降の相対方向rの min(r, 方向数-r) x 22.5度(rが0なら直進) ]
# }
@dataclass(frozen=True)
class StrokeStats:
    count: int
    single: int
    truncated: int
    turns_deg: tuple[float, ...]

    @property
    def single_rate(self) -> float | None:
        return self.single / self.count if self.count else None

    @property
    def mean_turn_deg(self) -> float | None:
        return sum(self.turns_deg) / len(self.turns_deg) if self.turns_deg else None


# {
#   責務: [ stroke_stats: 離散ストロークを選んだChoiceの列から集計する(連続座標のChoiceは数えない) ]
# }
def stroke_stats(choices: Iterable[ActionChoice]) -> StrokeStats:
    strokes = [choice for choice in choices if choice.stroke_dirs is not None]
    turns = tuple(
        min(rel, DIRECTIONS - rel) * (360.0 / DIRECTIONS)
        for choice in strokes
        for rel in choice.stroke_dirs[1:]
    )
    return StrokeStats(
        count=len(strokes),
        single=sum(len(choice.stroke_dirs) == 1 for choice in strokes),
        truncated=sum(choice.stroke_segments - len(choice.stroke_dirs) for choice in strokes),
        turns_deg=turns,
    )
