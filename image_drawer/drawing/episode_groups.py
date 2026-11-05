"""v2 Episodeのグループ・工程連鎖・レイヤー表のsemantic validatorと、末尾の未確定グループの切り詰め。

文法の実体はgroup_grammarにある。画素・筆跡の有効性（Stage規則）は判定しない。
"""

from __future__ import annotations

from dataclasses import replace

from .episode import DRAFT_REASONS, DrawingEpisode
from .engine import RENDERER_V2
from .episode_events import check_event_shapes
from .group_grammar import ChainState, check_group, check_initial_group
from .models import MAX_EVENTS, DrawingAction
from .stage.table import default_stage_table
from .validation import EpisodeFormatError


# {
#   責務: [ split_groups: R1（seq・時刻）とR2（グループの連続・index・size・完結）を検査してグループへ分ける ]
#   戻り値: [ groups: グループごとのイベント列 ]
#   エラー: [ EpisodeFormatError: R1/R2違反（末尾の未完結グループを含む） ]
# }
def split_groups(events: list[DrawingAction]) -> list[list[DrawingAction]]:
    if not 1 <= len(events) <= MAX_EVENTS:
        raise EpisodeFormatError(f"R1: events must have 1..{MAX_EVENTS} items")
    previous_time = 0
    for index, event in enumerate(events):
        if event.seq != index:
            raise EpisodeFormatError(f"R1: seq must be {index}, got {event.seq}")
        if event.time_ms < previous_time:
            raise EpisodeFormatError(f"R1 (seq {index}): time_ms must be non-decreasing")
        previous_time = event.time_ms
    groups: list[list[DrawingAction]] = []
    position = 0
    while position < len(events):
        group_info = events[position].group
        if group_info.id != len(groups) or group_info.index != 0:
            raise EpisodeFormatError(
                f"R2 (seq {position}): group id must be {len(groups)} starting at index 0"
            )
        end = position + group_info.size
        if end > len(events):
            raise EpisodeFormatError(
                f"R2 (seq {position}): trailing group {group_info.id} is incomplete"
            )
        chunk = events[position:end]
        for offset, event in enumerate(chunk):
            current = event.group
            if (current.id, current.index, current.size) != (
                group_info.id,
                offset,
                group_info.size,
            ):
                raise EpisodeFormatError(f"R2 (seq {event.seq}): inconsistent group annotation")
        groups.append(chunk)
        position = end
    return groups


# {
#   責務: [ _check_header: R8（status/理由）とR11（予算）を検査する ]
# }
def _check_header(episode: DrawingEpisode) -> None:
    complete = episode.status == "complete"
    if complete != (episode.termination_reason == "finished") or (
        not complete and episode.termination_reason not in DRAFT_REASONS
    ):
        raise EpisodeFormatError("R8: status and termination_reason are inconsistent")
    budget = episode.budget
    if budget["max_attempts"] < budget["max_actions"]:
        raise EpisodeFormatError("R11: max_attempts must be >= max_actions")


# {
#   責務: [ validate_episode_structure: v2 EpisodeのR1〜R12を検査する ]
#   エラー: [ EpisodeFormatError: v2でない、またはいずれかの規則に違反 ]
# }
def validate_episode_structure(episode: DrawingEpisode) -> None:
    if episode.renderer_version != RENDERER_V2:
        raise EpisodeFormatError("structure validation applies to v2 episodes only")
    check_event_shapes(episode.events)
    groups = split_groups(episode.events)
    _check_header(episode)
    table, state = default_stage_table(), ChainState()
    check_initial_group(table, state, groups[0])
    for group in groups[1:]:
        check_group(table, state, episode.style_intent, group)
    if state.model_events > episode.budget["max_actions"]:
        raise EpisodeFormatError("R11: model events exceed budget.max_actions")
    if state.finished != (episode.status == "complete"):
        raise EpisodeFormatError("R7: complete requires finish as the last event; draft forbids it")


# {
#   責務: [ truncate_to_last_complete_group: 末尾の未完結グループを取り除き、finishのないEpisodeをdraftにする ]
#   処理: [ 完成Episodeは変更しない。グループ以外の破損は直さず、後段の検証に任せる ]
#   エラー: [ EpisodeFormatError: 確定グループが1つも残らない ]
# }
def truncate_to_last_complete_group(episode: DrawingEpisode) -> DrawingEpisode:
    events = list(episode.events)
    last = events[-1].group
    if last.index != last.size - 1:
        start = len(events) - 1 - last.index
        if start >= 0 and events[start].group.index == 0 and events[start].group.id == last.id:
            events = events[:start]
    if not events:
        raise EpisodeFormatError("R2: no complete group remains after truncation")
    if (
        events[-1].type == "finish"
        and episode.status == "complete"
        and len(events) == len(episode.events)
    ):
        return episode
    if any(event.type == "finish" for event in events):
        return replace(episode, events=events)
    return replace(episode, events=events, status="draft", termination_reason="recovered_truncated")
