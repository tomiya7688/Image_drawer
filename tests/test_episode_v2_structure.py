"""S1c-1: v2 Episodeのグループ・工程連鎖・レイヤー表の構造検証（R1〜R12）。"""

from __future__ import annotations

import copy

import pytest
from episode_helpers import (
    draft_controller,
    find_group,
    full_raw,
    groups_of,
    initial_controller,
    raw_of,
    rebuild,
    review_controller,
    transition_index,
)

from image_drawer.drawing import DrawingEpisode, DrawingError
from image_drawer.drawing.episode_groups import validate_episode_structure
from image_drawer.drawing.validation import EpisodeFormatError


def check(raw: dict) -> None:
    validate_episode_structure(DrawingEpisode.from_dict(raw))


def reject(raw: dict, rule: str | None = None) -> None:
    error = EpisodeFormatError if rule else DrawingError
    with pytest.raises(error, match=rule):
        check(raw)


def event(type_: str, actor: str, stage: str, payload: dict | None = None) -> dict:
    return {
        "seq": 0, "time_ms": 0, "actor": actor, "type": type_, "payload": payload or {},
        "stage_id": stage, "group": {"id": 0, "index": 0, "size": 1},
    }  # fmt: skip


def edited(mutate) -> dict:
    """グループ列を加工して整合を振り直した辞書を返す。"""
    raw = full_raw()
    groups = groups_of(raw)
    mutate(groups)
    return rebuild(raw, groups)


def test_structure_accepts_controller_generated_episodes():
    for controller in (draft_controller(), review_controller()):
        check(raw_of(controller.episode()))
    check(full_raw())


def test_r1_rejects_seq_gap_duplicate_and_nonzero_start():
    for mutate in (
        lambda raw: raw["events"][3].update(seq=5),
        lambda raw: raw["events"][3].update(seq=2),
        lambda raw: [e.update(seq=e["seq"] + 1) for e in raw["events"]],
    ):
        raw = full_raw()
        mutate(raw)
        reject(raw)


def test_r1_rejects_decreasing_time_ms():
    raw = full_raw()
    raw["events"][2]["time_ms"] = 5
    reject(raw, "R1")


def test_r2_rejects_group_id_gap_or_reuse():
    for new_id in (50, 2):
        raw = full_raw()
        for item in groups_of(raw)[3]:
            item["group"]["id"] = new_id
        reject(raw, "R2")


def test_r2_rejects_group_with_noncontiguous_seq():
    raw = full_raw()
    raw["events"][5], raw["events"][6] = raw["events"][6], raw["events"][5]
    reject(raw)


def test_r2_rejects_group_index_out_of_order():
    raw = full_raw()
    group = groups_of(raw)[transition_index(raw)]
    group[1]["group"]["index"], group[2]["group"]["index"] = 2, 1
    reject(raw, "R2")


def test_r2_rejects_inconsistent_group_size_within_group():
    raw = full_raw()
    groups_of(raw)[transition_index(raw)][2]["group"]["size"] += 1
    reject(raw, "R2")


def test_r2_rejects_truncated_trailing_group():
    raw = full_raw()
    del raw["events"][-1]
    reject(raw, "R2")


def test_r3_rejects_missing_or_reordered_initial_group():
    reject(edited(lambda g: g[0].reverse()), "R3")
    reject(edited(lambda g: g[0].pop()), "R3")


def test_r4_rejects_transition_missing_exit_stage():
    reject(edited(lambda g: g[_t(g)].pop(1)), "R4")


def test_r4_rejects_transition_missing_clear_active_layer():
    reject(edited(lambda g: g[_t(g)].pop()), "R4")


def test_r4_rejects_transition_with_wrong_visibility_count():
    def drop_visibility(groups):
        index = _t(groups, 1)  # sketch -> final_drawing: roughを1件非表示にする
        groups[index] = [e for e in groups[index] if e["type"] != "set_layer_visibility"]

    reject(edited(drop_visibility), "R4")


def test_r4_rejects_advance_stage_alone_in_its_own_group():
    def split(groups):
        index = _t(groups)
        groups[index : index + 1] = [groups[index][:1], groups[index][1:]]

    reject(edited(split), "R4")


def test_r4_rejects_system_event_inside_action_group():
    def add_system(groups):
        index = next(i for i, g in enumerate(groups) if g[0]["type"] == "select_layer")
        groups[index].append(event("clear_active_layer", "system", groups[index][0]["stage_id"]))

    reject(edited(add_system), "R4")


def test_r4_rejects_set_layer_visibility_outside_transition():
    def stray(groups):
        payload = {"layer_id": "sys_rough_001", "visible": False}
        groups.insert(1, [event("set_layer_visibility", "system", "rough", payload)])

    reject(edited(stray), "R4")


def test_r5_rejects_wrong_stage_id_annotation():
    raw = full_raw()
    index = find_group(raw, "select_layer")
    groups_of(raw)[index][0]["stage_id"] = "sketch"
    reject(raw, "R5")


def _enter_edit(groups, nth: int, stage: str) -> None:
    enter = next(e for e in groups[_t(groups, nth)] if e["type"] == "enter_stage")
    enter["payload"]["stage_id"] = stage
    enter["stage_id"] = stage


def _t(groups, nth: int = 0) -> int:
    return [i for i, g in enumerate(groups) if g[0]["type"] == "advance_stage"][nth]


def test_r6_rejects_skipped_stage():
    reject(edited(lambda g: _enter_edit(g, 0, "final_drawing")), "R6")


def test_r6_rejects_backward_enter_stage():
    reject(edited(lambda g: _enter_edit(g, 2, "rough")), "R6")


def test_r6_rejects_duplicate_enter_stage():
    reject(edited(lambda g: _enter_edit(g, 1, "sketch")), "R6")


def test_r7_rejects_events_after_finish():
    def after(groups):
        groups.append([event("select_layer", "model", "finish", {"layer_id": "sys_rough_001"})])

    reject(edited(after), "R7")


def test_r7_rejects_complete_without_finish_and_draft_with_finish():
    reject(edited(lambda g: g.pop()), "R7")
    raw = full_raw()
    raw.update(status="draft", termination_reason="in_progress")
    reject(raw, "R7")


def test_r7_rejects_finish_outside_finalize_group():
    def lone_finish(groups):
        groups.insert(1, [event("finish", "system", "finish")])

    reject(edited(lone_finish), "R4")


def test_r8_rejects_inconsistent_status_and_reason():
    for status, reason in (("complete", "in_progress"), ("draft", "finished")):
        raw = full_raw()
        raw.update(status=status, termination_reason=reason)
        reject(raw, "R8")


def _create(layer_id: str, role: str = "rough", stage: str = "rough") -> dict:
    payload = {"layer_id": layer_id, "name": "x", "role": role}
    return event("create_layer", "model", stage, payload)


def test_r9_rejects_duplicate_layer_id():
    reject(edited(lambda g: g.insert(1, [_create("a1")]) or g.insert(2, [_create("a1")])), "R9")


def test_r9_rejects_visibility_event_for_unknown_layer():
    def unknown(groups):
        visibility = next(e for e in groups[_t(groups, 1)] if e["type"] == "set_layer_visibility")
        visibility["payload"]["layer_id"] = "nope"

    reject(edited(unknown), "R9")


def test_r9_rejects_model_layer_id_with_sys_prefix():
    reject(edited(lambda g: g.insert(1, [_create("sys_custom")])), "R9")


def test_r9_rejects_system_layer_id_with_wrong_sequence_number():
    def wrong(groups):
        create = next(e for e in groups[_t(groups, 0)] if e["type"] == "create_layer")
        create["payload"]["layer_id"] = "sys_sketch_002"

    reject(edited(wrong), "R9")


def test_r9_rejects_model_layer_with_role_not_creatable_in_stage():
    reject(edited(lambda g: g.insert(1, [_create("a1", role="highlight")])), "R9")


@pytest.mark.parametrize(
    "type_",
    [
        "enter_stage", "exit_stage", "set_layer_visibility", "clear_active_layer", "finish",
        "create_layer", "advance_stage", "select_layer", "select_color", "select_brush",
        "stroke_v2",
    ],
)  # fmt: skip
def test_r10_rejects_wrong_actor_for_each_event_type(type_):
    raw = full_raw()
    target = next(e for e in raw["events"] if e["type"] == type_)
    target["actor"] = "system" if target["actor"] == "model" else "model"
    reject(raw)


def test_r11_rejects_more_model_events_than_max_actions():
    raw = full_raw()
    raw["budget"] = {"max_actions": 1, "max_attempts": 5}
    reject(raw, "R11")
    raw["budget"] = {"max_actions": 10, "max_attempts": 5}
    reject(raw, "R11")


def test_r12_rejects_review_reason_not_allowed_for_stage():
    raw = full_raw()
    groups_of(raw)[transition_index(raw)][0]["payload"] = {"review_reason": "no_correction_needed"}
    reject(raw, "R12")


def test_r12_rejects_monochrome_review_when_intent_false():
    raw = full_raw()
    groups_of(raw)[transition_index(raw, 3)][0]["payload"] = {
        "review_reason": "monochrome_line_art_only"
    }
    reject(raw, "R12")


def test_r12_rejects_flat_review_when_intent_false():
    raw = full_raw()
    groups_of(raw)[transition_index(raw, 4)][0]["payload"] = {
        "review_reason": "flat_style_no_shading"
    }
    reject(raw, "R12")


def test_structure_accepts_draft_with_only_initial_group():
    raw = raw_of(initial_controller().episode())
    assert len(raw["events"]) == 3
    check(raw)


def test_structure_accepts_review_exit_in_three_stages():
    raw = raw_of(review_controller().episode())
    reasons = [
        e["payload"].get("review_reason") for e in raw["events"] if e["type"] == "advance_stage"
    ]
    assert reasons[3:] == [
        "monochrome_line_art_only",
        "flat_style_no_shading",
        "no_correction_needed",
    ]
    check(copy.deepcopy(raw))
