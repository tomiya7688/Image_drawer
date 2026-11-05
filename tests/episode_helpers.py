"""S1cのテストが共有する入力生成。Controllerの出力を元に、辞書を加工して不正Episodeを作る。"""

from __future__ import annotations

import copy
from typing import Any

from stage_helpers import (
    accept,
    accept_all,
    advance,
    advance_to,
    draw_actions,
    draw_and_advance,
    full_run,
    make_controller,
)

from image_drawer.drawing import DrawingEpisode
from image_drawer.drawing.stage.controller import StageController

WIDTH, HEIGHT = 12, 8
REVIEW_PROMPT = "白黒の線画だけ、影なしで描いて"


def full_controller() -> StageController:
    """7工程すべてを描いてfinishまで進めたController。"""
    controller = make_controller(width=WIDTH, height=HEIGHT)
    full_run(controller)
    return controller


def review_controller() -> StageController:
    """基本色・陰影・修正をレビュー退出するController。"""
    controller = make_controller(REVIEW_PROMPT, WIDTH, HEIGHT)
    for stage_id in ("rough", "sketch", "final_drawing"):
        draw_and_advance(controller, stage_id)
    accept(controller, advance("monochrome_line_art_only"))
    accept(controller, advance("flat_style_no_shading"))
    accept(controller, advance("no_correction_needed"))
    return controller


def draft_controller(max_actions: int = 12) -> StageController:
    """max_actions切れで閉じたdraftのController（final_drawing途中）。"""
    controller = make_controller(width=WIDTH, height=HEIGHT, max_actions=max_actions)
    advance_to(controller, "final_drawing")
    for action in draw_actions("final_drawing"):
        if controller.closed:
            break
        accept(controller, action)
    return controller


def initial_controller() -> StageController:
    return make_controller(width=WIDTH, height=HEIGHT)


def full_episode() -> DrawingEpisode:
    return full_controller().episode()


def raw_of(episode: DrawingEpisode) -> dict[str, Any]:
    """Episodeを独立した辞書にする（加工して不正入力を作るため）。"""
    return copy.deepcopy(episode.to_dict())


def full_raw() -> dict[str, Any]:
    return raw_of(full_episode())


def groups_of(raw: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """eventをgroup.idごとの列にして返す。"""
    groups: list[list[dict[str, Any]]] = []
    for event in raw["events"]:
        if event["group"]["id"] == len(groups):
            groups.append([])
        groups[-1].append(event)
    return groups


def rebuild(raw: dict[str, Any], groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """グループ列からseq・groupを振り直した辞書を返す（構造の加工後に整合を保つ）。"""
    events = []
    for group_id, group in enumerate(groups):
        for index, event in enumerate(group):
            event = copy.deepcopy(event)
            event["seq"] = len(events)
            event["group"] = {"id": group_id, "index": index, "size": len(group)}
            events.append(event)
    return {**raw, "events": events}


def find_group(raw: dict[str, Any], type_: str, nth: int = 0, size_min: int = 1) -> int:
    """指定typeを含むグループのうちnth番目の位置を返す。"""
    hits = [
        i
        for i, group in enumerate(groups_of(raw))
        if len(group) >= size_min and any(e["type"] == type_ for e in group)
    ]
    return hits[nth]


def transition_index(raw: dict[str, Any], nth: int = 0) -> int:
    """nth番目の遷移（advance_stageで始まる）グループの位置。"""
    hits = [i for i, g in enumerate(groups_of(raw)) if g[0]["type"] == "advance_stage"]
    return hits[nth]


def replay_pythonhashseed_script(path: str) -> str:
    """別プロセスでEpisodeを再生しRGBAのSHA-256を出力するスクリプト。"""
    return (
        "import hashlib, sys\n"
        "from image_drawer.drawing.episode_io import load_episode\n"
        "from image_drawer.drawing.replay import replay\n"
        f"result = replay(load_episode({path!r}))\n"
        "print(hashlib.sha256(result.image.rgba).hexdigest())\n"
    )


__all__ = [
    "REVIEW_PROMPT",
    "accept",
    "accept_all",
    "draft_controller",
    "find_group",
    "full_controller",
    "full_episode",
    "full_raw",
    "groups_of",
    "initial_controller",
    "raw_of",
    "rebuild",
    "replay_pythonhashseed_script",
    "review_controller",
    "transition_index",
]


def signature(state) -> tuple:
    """画素・レイヤー（id・role・visible）・選択状態・工程・finishedの比較用の値。"""
    layers = [(la.id, la.role, la.visible, bytes(la.pixels.data)) for la in state.layers]
    return (
        layers, state.active_layer_id, state.selected_color_id, state.selected_brush_id,
        state.size_level_id, state.opacity_level_id, state.stage_id, state.finished,
    )  # fmt: skip
