"""StageAuditor(保存したEpisodeだけから工程の遵守を再計算する監査器)を確認する。"""

import copy
import dataclasses
import inspect

from stage_helpers import (
    accept,
    accept_all,
    advance,
    draw_actions,
    draw_and_advance,
    full_run,
    make_controller,
    model,
)

from image_drawer.drawing import DrawingEpisode
from image_drawer.drawing.episode_io import dumps_episode, loads_episode
from image_drawer.generalization import auditor
from image_drawer.generalization.auditor import audit_episode


# {
#   責務: [ complete_episode: 7工程を描いてfinishまで進めたEpisodeを返す ]
# }
def complete_episode() -> DrawingEpisode:
    controller = make_controller()
    full_run(controller)
    return controller.episode()


# {
#   責務: [ codes: 監査結果の違反コードの集合 ]
# }
def codes(episode: DrawingEpisode) -> set[str]:
    return {v.code for v in audit_episode(episode).violations}


# {
#   責務: [ find_event: 工程とtypeで最初のeventを探す ]
# }
def find_event(episode: DrawingEpisode, event_type: str, stage_id: str):
    return next(e for e in episode.events if e.type == event_type and e.stage_id == stage_id)


# {
#   責務: [ change_event: eventは不変のため、同じ位置のeventを変更後の複製へ置き換える ]
# }
def change_event(episode: DrawingEpisode, event, **changes) -> None:
    index = episode.events.index(event)
    episode.events[index] = dataclasses.replace(event, **changes)


def test_auditor_accepts_complete_seven_stage_episode():
    result = audit_episode(complete_episode())
    assert result.passed and result.violations == []
    assert result.review_only_stages == []
    assert all(
        result.effective_strokes_by_stage[s] == 1 for s in ("rough", "sketch", "final_drawing")
    )
    assert result.changed_pixels_by_stage["rough"] > 0


def test_auditor_rejects_missing_stage_enter_or_exit():
    episode = complete_episode()
    episode.events.remove(find_event(episode, "enter_stage", "shading"))
    assert "stage_event_missing" in codes(episode)
    episode = complete_episode()
    episode.events.remove(find_event(episode, "exit_stage", "rough"))
    assert "stage_event_missing" in codes(episode)


def test_auditor_rejects_out_of_order_stage_events():
    episode = complete_episode()
    first = find_event(episode, "enter_stage", "sketch")
    second = find_event(episode, "enter_stage", "final_drawing")
    change_event(episode, first, payload=second.payload)
    change_event(episode, second, payload=first.payload)
    assert "stage_order" in codes(episode)


def test_auditor_rejects_required_stage_without_effective_stroke():
    episode = complete_episode()
    # 空のレイヤーを消しゴムで消すと受理されるが、画素は変わらず、stroke_v2でもない。
    brush = find_event(episode, "select_brush", "rough")
    change_event(episode, brush, payload={"brush_id": "eraser_hard"})
    change_event(episode, find_event(episode, "stroke_v2", "rough"), type="erase_v2")
    assert "required_stage_without_effective_stroke" in codes(episode)


def test_auditor_rejects_write_to_forbidden_layer():
    episode = complete_episode()
    event = find_event(episode, "select_layer", "sketch")
    change_event(episode, event, payload={"layer_id": "sys_rough_001"})
    assert "forbidden_layer_write" in codes(episode)


def test_auditor_rejects_skip_stage_and_v2_set_phase():
    for forbidden in ("skip_stage", "set_phase"):
        episode = complete_episode()
        change_event(episode, find_event(episode, "select_color", "rough"), type=forbidden)
        assert "forbidden_action" in codes(episode)


def test_auditor_rejects_review_reason_not_allowed_for_stage():
    episode = complete_episode()
    event = find_event(episode, "advance_stage", "rough")
    change_event(episode, event, payload={"review_reason": "no_correction_needed"})
    assert "review_reason_not_allowed" in codes(episode)


def test_auditor_does_not_read_controller_logs():
    # 入力はEpisodeだけで、Controllerの型や試行ログを参照しない。
    parameters = list(inspect.signature(audit_episode).parameters)
    assert parameters == ["episode", "table"]
    source = inspect.getsource(auditor)
    assert "StageController" not in source and "attempts" not in source
    # 保存して読み直したEpisode(試行ログを持たない)でも同じ結果になる。
    episode = complete_episode()
    reloaded = loads_episode(dumps_episode(episode))
    assert audit_episode(reloaded).passed and not audit_episode(copy.deepcopy(reloaded)).violations


def test_auditor_and_controller_agree_on_generated_episodes():
    episodes = [complete_episode()]
    # 拒否された提案(skip_stageなど)を含む経路。
    rejected = make_controller()
    assert not rejected.propose(model("skip_stage")).accepted
    assert not rejected.propose(advance()).accepted
    full_run(rejected)
    episodes.append(rejected.episode())
    # レビューのみで通過する経路(基本色と陰影)。
    review = make_controller("白黒の線画のみ 影なし")
    for stage_id in ("rough", "sketch", "final_drawing"):
        draw_and_advance(review, stage_id)
    accept(review, advance("monochrome_line_art_only"))
    accept(review, advance("flat_style_no_shading"))
    accept_all(review, draw_actions("refine"))
    accept(review, advance())
    review_episode = review.episode()
    episodes.append(review_episode)
    for episode in episodes:
        result = audit_episode(episode)
        assert result.passed, result.violations
    # Controllerが受理した結果と、監査器の判定が食い違った件数は0。
    assert audit_episode(review_episode).review_only_stages == ["base_color", "shading"]
