"""S1c-3: ReplayerとvalidatorがStageControllerの実装に依存しないことの確認。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from episode_helpers import (
    full_controller,
    full_raw,
    groups_of,
    initial_controller,
    raw_of,
    rebuild,
    transition_index,
)

from image_drawer.drawing import DrawingEpisode
from image_drawer.drawing.episode_groups import validate_episode_structure
from image_drawer.drawing.episode_io import dumps_episode, loads_episode
from image_drawer.drawing.replay import replay
from image_drawer.drawing.stage.controller import StageController
from image_drawer.drawing.validation import EpisodeFormatError

DRAWING = Path(__file__).resolve().parents[1] / "image_drawer" / "drawing"
INDEPENDENT_MODULES = (
    "replay.py", "episode_groups.py", "group_grammar.py", "episode_events.py", "episode_io.py",
)  # fmt: skip
ALLOWED_STAGE_MODULES = {"stage.table", "stage.errors"}


def stage_imports(name: str) -> set[str]:
    """モジュールがimportするstageパッケージ内の名前（相対importを含む）。"""
    found: set[str] = set()
    for node in ast.walk(ast.parse((DRAWING / name).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module.startswith("stage") or ".stage" in module:
                found.add("stage." + module.split("stage", 1)[1].lstrip("."))
                if module.rstrip(".") in ("stage", "image_drawer.drawing.stage"):
                    found.update(f"stage.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.update(f"stage.{a.name}" for a in node.names if a.name == "stage")
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if ".stage" in a.name)
    return found


def test_replay_modules_do_not_import_stage_controller_rules_groups_or_progress():
    for name in INDEPENDENT_MODULES:
        imported = {item.rstrip(".") for item in stage_imports(name)}
        assert imported <= ALLOWED_STAGE_MODULES, (name, imported)
    assert "stage.table" in stage_imports("group_grammar.py")


def test_replay_works_without_constructing_stage_controller(monkeypatch):
    text = dumps_episode(full_controller().episode())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("StageController must not be constructed")

    monkeypatch.setattr(StageController, "__init__", forbidden)
    result = replay(loads_episode(text))
    assert result.state.finished


def test_validator_group_grammar_is_independent_implementation():
    raw = full_raw()
    validate_episode_structure(DrawingEpisode.from_dict(raw))
    groups = groups_of(raw)
    index = transition_index(raw)
    # 文法を1か所壊す: create_layerとclear_active_layerの順を入れ替える
    groups[index][-1], groups[index][-2] = groups[index][-2], groups[index][-1]
    with pytest.raises(EpisodeFormatError, match="R4"):
        validate_episode_structure(DrawingEpisode.from_dict(rebuild(raw, groups)))


def test_replayer_does_not_judge_stage_rules():
    """筆跡なしでroughを退出する記録でも、構造が正しければ再生される（Stage規則の再監査は範囲外）。"""
    raw = raw_of(initial_controller().episode())
    advance = {
        "seq": 0, "time_ms": 0, "actor": "model", "type": "advance_stage", "payload": {},
        "stage_id": "rough", "group": {"id": 0, "index": 0, "size": 1},
    }  # fmt: skip
    full = groups_of(full_raw())
    transition = [advance, *full[transition_index(full_raw())][1:]]
    for event in transition:
        event["stage_id"] = event["stage_id"] if event["type"] != "advance_stage" else "rough"
    episode = DrawingEpisode.from_dict(rebuild(raw, [*groups_of(raw), transition]))
    validate_episode_structure(episode)
    result = replay(episode)
    assert result.state.stage_id == "sketch"
    assert not any(any(layer.rgba) for layer in result.layers)
