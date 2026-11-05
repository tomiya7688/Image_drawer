"""製品コードのReplayer。EpisodeのeventをEngineへ再適用し、画素・レイヤー・工程状態を得る。

StageControllerは使わない。Controllerが受理した記録だけで同じ状態へ戻れること（自己完結）を示す。
Stage規則（有効筆跡など）は再判定しない。
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field

from .engine import RENDERER_V2, Engine, RenderedImage
from .episode import DrawingEpisode, check_workflow_identity
from .models import FINISH_STAGE_ID, DrawingAction, DrawingState
from .validation import DrawingError, ReplayError


# {
#   責務: [ ReplayedLayer: 再生後のレイヤー1枚（ID・名前・role・可視性・RGBA8） ]
#   フィールド: [ rgba: 行優先のRGBA8。幅×高さ×4バイト ]
# }
@dataclass(frozen=True)
class ReplayedLayer:
    layer_id: str
    name: str
    role: str | None
    visible: bool
    rgba: bytes


# {
#   責務: [ ReplayResult: 再生結果。合成画像・作成順のレイヤー・Engine状態 ]
# }
@dataclass(frozen=True)
class ReplayResult:
    image: RenderedImage
    layers: tuple[ReplayedLayer, ...]
    state: DrawingState


# {
#   責務: [ _snapshot: Engineの現在状態をReplayResultにする（imageは合成、layersは複製） ]
#   引数: [ engine: 再生中のEngine, copy_state: Trueなら状態を深く複製する ]
# }
def _snapshot(engine: Engine, copy_state: bool) -> ReplayResult:
    state = engine.state
    layers = tuple(
        ReplayedLayer(layer.id, layer.name, layer.role, layer.visible, bytes(layer.pixels.data))
        for layer in state.layers
    )
    return ReplayResult(engine.render(), layers, copy.deepcopy(state) if copy_state else state)


# {
#   責務: [ ReplayFrame: iter_replayが返す1イベント分の位置。capture()で結果を得る ]
#   フィールド: [ event: 直前に適用したイベント, is_group_end: グループ（v1は1件）の最後か ]
#   処理: [ capture()は合成と複製を行う。グループの途中状態は公開しないため、グループ末尾以外と、次へ進んだ後のframeではReplayErrorにする ]
# }
@dataclass(frozen=True)
class ReplayFrame:
    event: DrawingAction
    is_group_end: bool
    _engine: Engine = field(repr=False, compare=False)

    def capture(self) -> ReplayResult:
        if len(self._engine.state.history) != self.event.seq + 1:
            raise ReplayError(
                "capture is only available at the end of the group, before the next step",
                self.event.seq,
            )
        return _snapshot(self._engine, copy_state=True)


# {
#   責務: [ _partition: eventを適用単位へ分ける。v2はgroup注釈、v1は1件ずつ ]
#   エラー: [ ReplayError: v2でgroup注釈が欠ける・不整合 ]
# }
def _partition(episode: DrawingEpisode, is_v2: bool) -> list[list[DrawingAction]]:
    if not is_v2:
        return [[event] for event in episode.events]
    groups: list[list[DrawingAction]] = []
    position, events = 0, episode.events
    while position < len(events):
        first = events[position].group
        if first is None:
            raise ReplayError("v2 event has no group annotation", events[position].seq)
        chunk = events[position : position + first.size]
        valid = len(chunk) == first.size and all(
            e.group is not None
            and (e.group.id, e.group.index, e.group.size) == (first.id, index, first.size)
            for index, e in enumerate(chunk)
        )
        if not valid:
            raise ReplayError("inconsistent group annotation", events[position].seq)
        groups.append(chunk)
        position += first.size
    return groups


# {
#   責務: [ _check_until: until_seqが範囲内で、v2ではグループ末尾であることを確認する ]
# }
def _check_until(until_seq: int, groups: list[list[DrawingAction]], is_v2: bool) -> None:
    total = sum(len(group) for group in groups)
    if type(until_seq) is not int or not 0 <= until_seq < total:
        raise ReplayError(f"until_seq {until_seq!r} is out of range 0..{total - 1}")
    if is_v2 and until_seq not in {group[-1].seq for group in groups}:
        raise ReplayError("until_seq must be the last event of a group", until_seq)


# {
#   責務: [ _check_annotations: 各eventのstage_idが、直前までの連鎖から導かれる工程と一致するか確認する ]
#   戻り値: [ stage: グループ末尾の工程 ]
# }
def _check_annotations(stage: str | None, group: list[DrawingAction]) -> str | None:
    for event in group:
        if event.type == "enter_stage":
            stage = event.payload["stage_id"]
        elif event.type == "finish":
            stage = FINISH_STAGE_ID
        if event.stage_id != stage:
            raise ReplayError(
                f"stage_id annotation {event.stage_id!r} does not match {stage!r}", event.seq
            )
    return stage


# {
#   責務: [ _apply: グループをEngineへ適用する。1件はapply、複数件はapply_group（全件成功か全件不変） ]
#   エラー: [ ReplayError: Engineが拒否した（seq付き、元の例外を保持） ]
# }
def _apply(engine: Engine, group: list[DrawingAction], is_v2: bool) -> None:
    events = copy.deepcopy(group)
    try:
        if is_v2 and len(events) > 1:
            engine.apply_group(events)
        else:
            for event in events:
                engine.apply(event)
    except DrawingError as exc:
        raise ReplayError(str(exc), group[0].seq) from exc


# {
#   責務: [ _new_engine: headerを確認してEngineを作る ]
#   エラー: [ ReplayError: workflow識別・寸法・seed・rendererの不一致 ]
# }
def _new_engine(episode: DrawingEpisode, is_v2: bool) -> Engine:
    try:
        if is_v2:
            check_workflow_identity(episode.workflow_id, episode.workflow_sha256)
        return Engine(
            episode.width, episode.height, profile=episode.renderer_version, seed=episode.seed
        )
    except DrawingError as exc:
        raise ReplayError(f"episode header is not replayable: {exc}") from exc


# {
#   責務: [ _run: グループ単位で再生し、適用後のEngineと適用したグループを順に返す ]
#   処理: [ until_seqのグループまで。完成Episodeを最後まで再生したらfinishedを確認する ]
# }
def _run(
    episode: DrawingEpisode, until_seq: int | None
) -> Iterator[tuple[Engine, list[DrawingAction]]]:
    is_v2 = episode.renderer_version == RENDERER_V2
    engine = _new_engine(episode, is_v2)
    groups = _partition(episode, is_v2)
    if until_seq is not None:
        _check_until(until_seq, groups, is_v2)
    stage: str | None = None
    for group in groups:
        _apply(engine, group, is_v2)
        if is_v2:
            stage = _check_annotations(stage, group)
            if stage != engine.state.stage_id:
                raise ReplayError("stage_id does not match the Engine state", group[-1].seq)
        yield engine, group
        if until_seq is not None and group[-1].seq >= until_seq:
            return
    if until_seq is None and episode.status == "complete" and not engine.state.finished:
        raise ReplayError("a complete episode did not reach finish")


# {
#   責務: [ replay: EpisodeをEngineへ再適用し、最終（またはuntil_seq時点）の状態を返す ]
#   引数: [ episode: 入力（変更しない）, until_seq: そのseqまで。v2ではグループ末尾のseqだけ ]
#   エラー: [ ReplayError: Engine拒否・注釈不一致・until_seqの不正・headerの不一致 ]
# }
def replay(episode: DrawingEpisode, *, until_seq: int | None = None) -> ReplayResult:
    engine = None
    for engine, _group in _run(episode, until_seq):
        pass
    if engine is None:
        raise ReplayError("episode has no events")
    return _snapshot(engine, copy_state=False)


# {
#   責務: [ iter_replay: グループを適用するたびに、そのグループの各eventのframeを順に返す ]
#   処理: [ グループ全体を適用した後にframeを返すため、capture()できるのはグループ末尾のframeだけ ]
# }
def iter_replay(episode: DrawingEpisode) -> Iterator[ReplayFrame]:
    for engine, group in _run(episode, None):
        for event in group:
            yield ReplayFrame(event, event is group[-1], engine)
