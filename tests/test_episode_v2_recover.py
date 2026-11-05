"""S1c-2: 末尾の未確定グループだけを切り詰める回復（グループ以外の破損は回復しない）。"""

from __future__ import annotations

import json

import pytest
from episode_helpers import full_raw, groups_of, transition_index

from image_drawer.drawing.episode_io import dumps_episode, load_episode, loads_episode, save_episode
from image_drawer.drawing.validation import EpisodeFormatError


def truncated_raw(keep_in_group: int = 2) -> tuple[dict, int]:
    """sketch→final_drawingの遷移グループの途中で切れた文書と、確定グループ数を返す。"""
    raw = full_raw()
    groups = groups_of(raw)
    index = transition_index(raw, 1)
    kept = sum(len(g) for g in groups[:index]) + keep_in_group
    raw["events"] = raw["events"][:kept]
    return raw, index


def test_recover_truncates_trailing_incomplete_group_to_last_complete_group():
    raw, index = truncated_raw()
    episode = loads_episode(json.dumps(raw), recover=True)
    complete = sum(len(g) for g in groups_of(full_raw())[:index])
    assert len(episode.events) == complete
    assert episode.events[-1].group.id == index - 1


def test_recover_result_is_draft_with_recovered_truncated_reason():
    raw, _ = truncated_raw()
    episode = loads_episode(json.dumps(raw), recover=True)
    assert (episode.status, episode.termination_reason) == ("draft", "recovered_truncated")


def test_recover_off_by_default_rejects_incomplete_group():
    raw, _ = truncated_raw()
    with pytest.raises(EpisodeFormatError, match="R2"):
        loads_episode(json.dumps(raw))


def test_recover_does_not_modify_complete_episode():
    text = dumps_episode(loads_episode(json.dumps(full_raw())))
    assert dumps_episode(loads_episode(text, recover=True)) == text


def test_recover_cannot_repair_non_group_corruption():
    raw, _ = truncated_raw()
    with pytest.raises(EpisodeFormatError):
        loads_episode(json.dumps({**raw, "workflow_sha256": "0" * 64}), recover=True)
    raw["events"][5]["stage_id"] = "sketch"
    with pytest.raises(EpisodeFormatError, match="R5"):
        loads_episode(json.dumps(raw), recover=True)


def test_recover_with_only_initial_group_incomplete_is_rejected():
    raw = full_raw()
    raw["events"] = raw["events"][:2]
    with pytest.raises(EpisodeFormatError):
        loads_episode(json.dumps(raw), recover=True)


def test_recovered_draft_can_be_saved_and_loaded(tmp_path):
    raw, _ = truncated_raw()
    recovered = loads_episode(json.dumps(raw), recover=True)
    save_episode(recovered, tmp_path / "draft.json")
    assert load_episode(tmp_path / "draft.json") == recovered
