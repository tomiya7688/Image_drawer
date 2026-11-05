"""S4-8: 文章だけのWorkflowのE2E。水準A（配線）だけを確認し、モデルの品質は確認しない。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from inference_helpers import SIDE, scripted_registry

from image_drawer.dsl import parse_workflow
from image_drawer.drawing.episode_io import loads_episode
from image_drawer.drawing.replay import replay
from image_drawer.runtime import WorkflowRuntime
from image_drawer.steps import create_mock_registry, register_drawing_steps

WORKFLOW = """INPUT prompt: Text
episode = DRAW_EPISODE(prompt, policy="%s", max_actions=1000, seed=7, width=%d, height=%d)
image = REPLAY_EPISODE(episode)
OUTPUT image
"""
LEVEL_A_NOTICE = "水準A（配線のみ）: テスト用の台本Policyによる確認で、モデル未検証"


def run_workflow(policies, output_dir: Path, policy_id: str, side: int):
    registry = create_mock_registry()
    register_drawing_steps(registry, policies=policies, output_dir=output_dir)
    workflow = parse_workflow(WORKFLOW % (policy_id, side, side), registry)
    return WorkflowRuntime(registry).execute(
        workflow, external_inputs={"prompt": "猫が座っている"}, run_id="run-e2e"
    )


def artifacts(result, kind: str) -> list:
    return [a for a in result.trajectory.artifacts if a.artifact_type == kind]


def test_text_only_workflow_end_to_end_with_scripted_policy(tmp_path: Path) -> None:
    print(LEVEL_A_NOTICE)
    result = run_workflow(scripted_registry(), tmp_path, "scripted", SIDE)
    (text,) = artifacts(result, "Text")
    (episode_artifact,) = artifacts(result, "Episode")
    assert text.parent_artifact_ids == [] and episode_artifact.parent_artifact_ids == [text.id]
    # 参照画像が入力に無い: Workflowの外部入力は文章1つだけ
    assert result.trajectory.input_metadata == {"external_input_ids": ["prompt"]}
    assert result.trajectory.prompt == "猫が座っている"

    path = Path(episode_artifact.uri)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == episode_artifact.metadata["episode_sha256"]
    )
    episode = loads_episode(path.read_bytes())
    assert episode.status == "complete" and episode.prompt == "猫が座っている"
    stages = [e.payload["stage_id"] for e in episode.events if e.type == "enter_stage"]
    assert stages == ["rough", "sketch", "final_drawing", "base_color", "shading", "refine"]
    assert episode_artifact.metadata["policy_trained"] is False

    replayed = next(a for a in artifacts(result, "Image") if a.producing_step_id == "image")
    final = replay(episode).image.rgba
    assert replayed.metadata["rgba_sha256"] == hashlib.sha256(final).hexdigest()
    assert replayed.metadata["rgba_sha256"] == episode_artifact.metadata["final_image_rgba_sha256"]
    assert json.loads(json.dumps(episode_artifact.metadata, default=str))["status"] == "complete"


def test_text_only_workflow_end_to_end_with_trained_checkpoint(tmp_path: Path) -> None:
    sidecar = os.environ.get("IMAGE_DRAWER_E2E_CHECKPOINT")
    config_path = os.environ.get("IMAGE_DRAWER_E2E_TRAIN_CONFIG")
    if not sidecar or not config_path or not Path(sidecar).is_file():
        pytest.skip("水準B未検証: 学習済みcheckpoint(IMAGE_DRAWER_E2E_CHECKPOINT)が存在しない")
    pytest.importorskip("numpy")
    from image_drawer.inference.policy import PolicyRegistry
    from image_drawer.inference.training_policy import register_checkpoint
    from image_drawer.training import TrainConfig

    config = TrainConfig.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
    registry = PolicyRegistry()
    register_checkpoint(registry, "trained", Path(sidecar), config)
    result = run_workflow(registry, tmp_path, "trained", config.canvas_side)
    (episode_artifact,) = artifacts(result, "Episode")
    assert episode_artifact.metadata["policy_trained"] is True
    replayed = next(a for a in artifacts(result, "Image") if a.producing_step_id == "image")
    assert replayed.metadata["rgba_sha256"] == episode_artifact.metadata["final_image_rgba_sha256"]
