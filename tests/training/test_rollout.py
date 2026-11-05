"""Rollout・integrityのテスト(設計書8.3節)。"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from training_helpers import complete_script, run_scripted, small_config  # noqa: E402

from image_drawer.drawing import DrawingAction  # noqa: E402
from image_drawer.training import rollout as rollout_module  # noqa: E402
from image_drawer.training.integrity import (  # noqa: E402
    IntegrityError,
    forbidden_imports,
    verify_allowed_actions,
    verify_trial_replay,
)
from image_drawer.training.rollout import RolloutRunner, Trial  # noqa: E402

FORBIDDEN_NAMES = {"RewardTarget", "TargetFeatures", "RewardEvaluator", "TrialReward"}


def test_rollout_completes_all_stages_with_complete_script() -> None:
    trial = run_scripted(complete_script())
    assert trial.end_reason == "finish" and trial.actor == "model"
    assert trial.episode.status == "complete" and trial.final_composite is not None
    assert all(step.accepted for step in trial.steps)
    assert sum(1 for step in trial.steps if step.composite_after is not None) == len(trial.steps)


def test_rollout_records_rejected_attempts() -> None:
    script = [DrawingAction("advance_stage", {})] * 2 + complete_script()
    trial = run_scripted(script)
    assert [step.accepted for step in trial.steps[:2]] == [False, False]
    assert trial.steps[0].reject_reason and trial.steps[0].composite_after is None
    assert [step.attempt_index for step in trial.steps[:3]] == [1, 2, 3]
    # 拒否はEpisodeのeventに入らず、seqも進まない。
    seqs = [event.seq for event in trial.episode.events]
    assert seqs == list(range(len(seqs)))
    accepted_events = sum(len(step.events) for step in trial.steps if step.accepted)
    assert len(trial.episode.events) == 3 + accepted_events
    assert trial.end_reason == "finish"


def test_rollout_budget_exhaustion_returns_unfinished_draft() -> None:
    script = [DrawingAction("select_color", {"color_id": 1})]
    trial = run_scripted(script, small_config(max_actions=5))
    assert trial.end_reason == "action_budget"
    assert trial.episode.status == "draft"
    assert trial.episode.termination_reason == "action_budget_exhausted"
    attempts = run_scripted([DrawingAction("advance_stage", {})], small_config(max_actions=5))
    assert attempts.end_reason == "attempt_budget" and len(attempts.steps) == 100


def test_rollout_curriculum_stop_is_not_counted_as_completed() -> None:
    trial = run_scripted(complete_script(), stop_after_stage="rough")
    assert trial.end_reason == "curriculum_stop" and trial.stop_after_stage == "rough"
    assert trial.episode.status == "draft"
    assert {step.stage_id for step in trial.steps} == {"rough"}


def test_rollout_records_policy_error() -> None:
    trial = run_scripted(complete_script(), raise_at=3)
    assert trial.end_reason == "policy_error" and len(trial.steps) == 3


def test_trial_replay_matches_final_composite() -> None:
    for trial in (
        run_scripted(complete_script()),
        run_scripted(complete_script(), stop_after_stage="sketch"),
        run_scripted([DrawingAction("select_color", {"color_id": 1})], small_config(max_actions=5)),
    ):
        verify_trial_replay(trial)
        verify_allowed_actions(trial)


def test_integrity_rejects_trial_with_foreign_pixels() -> None:
    trial = run_scripted(complete_script())
    pixels = bytearray(trial.final_composite)
    pixels[0:4] = bytes([1, 2, 3, 255])
    with pytest.raises(IntegrityError):
        verify_trial_replay(dataclasses.replace(trial, final_composite=bytes(pixels)))
    tampered = dataclasses.replace(trial.episode, events=list(trial.episode.events)[:-2])
    with pytest.raises(IntegrityError):
        verify_trial_replay(dataclasses.replace(trial, episode=tampered))


def test_trial_actor_is_model_only() -> None:
    trial = run_scripted(complete_script())
    with pytest.raises(ValueError):
        dataclasses.replace(trial, actor="human")
    events = list(trial.episode.events)
    events[3] = dataclasses.replace(events[3], actor="human")
    with pytest.raises(ValueError):
        dataclasses.replace(trial, episode=dataclasses.replace(trial.episode, events=events))
    pasted = dataclasses.replace(events[3], actor="model", type="paste")
    events[3] = pasted
    episode = dataclasses.replace(trial.episode, events=events)
    with pytest.raises(IntegrityError):
        verify_allowed_actions(dataclasses.replace(trial, episode=episode))


def test_runner_signature_excludes_evaluator_and_target() -> None:
    for function in (RolloutRunner.__init__, RolloutRunner.run_trial):
        text = " ".join(str(hint) for hint in typing.get_type_hints(function).values())
        assert not any(name in text for name in FORBIDDEN_NAMES)
    source = Path(rollout_module.__file__).read_text(encoding="utf-8")
    assert "RewardTarget" not in source and "RewardEvaluator" not in source


def test_training_package_does_not_import_part_bank_or_compose() -> None:
    assert forbidden_imports(Path(rollout_module.__file__).parent) == []
    assert isinstance(Trial, type)
