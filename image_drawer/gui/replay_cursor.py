"""Replay画面のカーソル。前進は1グループずつ適用し、後退は工程境界のスナップショットから再生する。"""

from __future__ import annotations

import bisect
import copy
from collections import OrderedDict
from dataclasses import dataclass

from image_drawer.drawing import DrawingAction, DrawingEpisode, DrawingState, Engine
from image_drawer.drawing.engine import RENDERER_V2, RenderedImage

FRAME_CACHE_SIZE = 32


# {
#   責務: [ split_groups: 連続する同じgroup.idのeventを適用単位にまとめる（注釈が無ければ1件ずつ） ]
# }
def split_groups(events: list[DrawingAction]) -> list[list[DrawingAction]]:
    groups: list[list[DrawingAction]] = []
    for event in events:
        same = groups and event.group is not None and groups[-1][-1].group is not None
        if same and groups[-1][-1].group.id == event.group.id:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


# {
#   責務: [
#     ReplayCursor: Episodeの任意のグループ末尾へ移動して状態・画像を返す。Episodeは変更しない
#   ]
#   フィールド: [
#     index: 適用済みの最後のグループ番号（-1は何も描いていない状態）,
#     スナップショット: enter_stageを含むグループの直後のDrawingStateだけ保存する（最大8か所）
#   ]
#   処理: [ 後退・seekは、目的以前で最も近いスナップショット（無ければ初期状態）から再生し直す ]
# }
class ReplayCursor:
    def __init__(self, episode: DrawingEpisode, cache_size: int = FRAME_CACHE_SIZE) -> None:
        self.episode = episode
        self._groups = split_groups(episode.events)
        self._ends = [group[-1].seq for group in self._groups]
        self._boundaries = {
            index
            for index, group in enumerate(self._groups)
            if any(event.type == "enter_stage" for event in group)
        }
        self._stroke_index = [
            (index, event)
            for index, group in enumerate(self._groups)
            for event in group
            if event.type == "stroke_v2"
        ]
        self._stroke_positions = [item[0] for item in self._stroke_index]
        self._snapshots: dict[int, DrawingState] = {}
        self._frames: OrderedDict[int, RenderedImage] = OrderedDict()
        self._cache_size = cache_size
        self._engine = self._fresh_engine()
        self.index = -1

    def _fresh_engine(self) -> Engine:
        episode = self.episode
        return Engine(
            episode.width, episode.height, profile=episode.renderer_version, seed=episode.seed
        )

    @property
    def last_index(self) -> int:
        return len(self._groups) - 1

    @property
    def seq(self) -> int:
        return self._ends[self.index] if self.index >= 0 else -1

    @property
    def state(self) -> DrawingState:
        return self._engine.state

    def _apply_group(self, index: int) -> None:
        group = copy.deepcopy(self._groups[index])
        if len(group) > 1 and self.episode.renderer_version == RENDERER_V2:
            self._engine.apply_group(group)
        else:
            for event in group:
                self._engine.apply(event)
        if index in self._boundaries and index not in self._snapshots:
            self._snapshots[index] = copy.deepcopy(self._engine.state)

    # {
    #   責務: [ seek_index: 指定グループ末尾（-1..last_index）へ移動する ]
    # }
    def seek_index(self, target: int) -> None:
        target = max(-1, min(target, self.last_index))
        if target < self.index:
            base = max((b for b in self._snapshots if b <= target), default=-1)
            if base < 0:
                self._engine = self._fresh_engine()
            else:
                self._engine.state = copy.deepcopy(self._snapshots[base])
            self.index = base
        for index in range(self.index + 1, target + 1):
            self._apply_group(index)
        self.index = target

    # {
    #   責務: [ seek: seq以下で最も後ろのグループ末尾へ移動する ]
    # }
    def seek(self, seq: int) -> None:
        self.seek_index(bisect.bisect_right(self._ends, seq) - 1)

    def forward(self) -> None:
        self.seek_index(self.index + 1)

    def back(self) -> None:
        self.seek_index(self.index - 1)

    # {
    #   責務: [ stage_boundary_index: 工程に入るグループの番号。その工程が無ければNone ]
    # }
    def stage_boundary_index(self, stage_id: str) -> int | None:
        for index in sorted(self._boundaries):
            if any(
                event.type == "enter_stage" and event.payload.get("stage_id") == stage_id
                for event in self._groups[index]
            ):
                return index
        return None

    def jump_to_stage(self, stage_id: str) -> bool:
        index = self.stage_boundary_index(stage_id)
        if index is None:
            return False
        self.seek_index(index)
        return True

    # {
    #   責務: [ image: 現在位置の合成画像。最近使った32枚だけ保持する ]
    # }
    def image(self) -> RenderedImage:
        if self.index in self._frames:
            self._frames.move_to_end(self.index)
            return self._frames[self.index]
        image = self._engine.render()
        self._frames[self.index] = image
        if len(self._frames) > self._cache_size:
            self._frames.popitem(last=False)
        return image

    # {
    #   責務: [ last_stroke: 現在位置までで最後のstroke_v2 event。画素は変えず強調表示に使う ]
    # }
    def last_stroke(self) -> DrawingAction | None:
        position = bisect.bisect_right(self._stroke_positions, self.index)
        return self._stroke_index[position - 1][1] if position else None
