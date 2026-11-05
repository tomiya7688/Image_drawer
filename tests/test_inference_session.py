"""S4-1: 推論ループ（generate_episode）。水準A（配線）の確認で、モデルの品質は確認しない。"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import threading
from pathlib import Path

import pytest
from inference_helpers import SIDE, complete_actions, scripted_policy, scripted_registry
from stage_helpers import advance, model

from image_drawer.drawing.episode_io import dumps_episode, loads_episode
from image_drawer.inference import DrawRequest, DrawRequestError, UnknownPolicyError
from image_drawer.inference.session import generate_episode

INFERENCE_DIR = Path(__file__).resolve().parents[1] / "image_drawer" / "inference"


def request(**kwargs) -> DrawRequest:
    base = {"prompt": "猫", "policy": "scripted", "width": SIDE, "height": SIDE}
    return DrawRequest(**{**base, **kwargs})


def random_registry():
    pytest.importorskip("numpy")
    from image_drawer.inference.training_policy import register_random_legal
    from image_drawer.training import TrainConfig

    registry = scripted_registry()
    register_random_legal(registry, TrainConfig(canvas_side=SIDE, summary_side=4, max_actions=30))
    return registry


def test_generate_episode_completes_with_scripted_policy_and_finishes_once() -> None:
    outcome = generate_episode(request(), scripted_registry())
    assert (outcome.status, outcome.reason) == ("complete", "finished")
    assert outcome.episode.status == "complete"
    assert [e.type for e in outcome.episode.events].count("finish") == 1
    assert outcome.episode.events[-1].type == "finish"
    assert outcome.rejected_attempts == 0 and outcome.system_events > 0
    assert len(outcome.image.rgba) == SIDE * SIDE * 4
    assert loads_episode(dumps_episode(outcome.episode)).status == "complete"


def test_generate_episode_returns_draft_when_action_budget_is_exhausted() -> None:
    outcome = generate_episode(request(max_actions=5), scripted_registry())
    assert (outcome.status, outcome.reason) == ("draft", "action_budget_exhausted")
    assert outcome.episode.status == "draft"
    assert all(e.type != "finish" for e in outcome.episode.events)
    assert outcome.accepted_actions == 5


def test_generate_episode_returns_draft_when_attempt_limit_is_reached() -> None:
    bad = [model("skip_stage")]
    outcome = generate_episode(request(max_actions=1), scripted_registry(actions=bad))
    assert (outcome.status, outcome.reason) == ("draft", "attempt_budget_exhausted")
    assert outcome.accepted_actions == 0 and outcome.rejected_attempts == 100


def test_generate_episode_returns_draft_with_event_limit_exceeded_near_max_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("image_drawer.drawing.engine.MAX_EVENTS", 9)
    outcome = generate_episode(request(), scripted_registry())
    assert (outcome.status, outcome.reason) == ("draft", "event_limit_exceeded")
    assert len(outcome.episode.events) <= 9


def test_generate_episode_policy_exception_becomes_draft_with_policy_error() -> None:
    outcome = generate_episode(request(), scripted_registry(raise_at=3))
    assert (outcome.status, outcome.reason) == ("draft", "policy_error")
    assert outcome.accepted_actions == 3


def test_generate_episode_cancel_event_stops_with_reason_cancelled() -> None:
    cancel = threading.Event()
    seen: list[int] = []

    def on_progress(result, controller) -> None:
        seen.append(result.attempt)
        if len(seen) == 2:
            cancel.set()

    registry = scripted_registry()
    outcome = generate_episode(request(), registry, cancel=cancel, on_progress=on_progress)
    assert (outcome.status, outcome.reason) == ("draft", "cancelled")
    assert outcome.accepted_actions == 2 and outcome.episode.status == "draft"


def test_rejected_attempts_are_logged_and_do_not_consume_seq() -> None:
    script = [model("skip_stage"), advance(), *complete_actions()]
    outcome = generate_episode(request(), scripted_registry(actions=script))
    assert outcome.status == "complete" and outcome.rejected_attempts == 2
    rejected = [r for r in outcome.attempts if not r.accepted]
    assert all(r.reason and r.seqs == () for r in rejected)
    seqs = [event.seq for event in outcome.episode.events]
    assert seqs == list(range(len(seqs)))


def test_policy_observation_contains_only_allowed_fields() -> None:
    registry = random_registry()
    from image_drawer.training.observation import OBSERVATION_KEYS, Observation, ObservationBuilder

    assert {f.name for f in dataclasses.fields(Observation)} == OBSERVATION_KEYS
    captured: list[Observation] = []
    original = ObservationBuilder.build

    def spy(self, *args, **kwargs):
        captured.append(original(self, *args, **kwargs))
        return captured[-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ObservationBuilder, "build", spy)
        generate_episode(request(policy="random-legal-v1", max_actions=10), registry)
    assert captured
    assert all({f.name for f in dataclasses.fields(o)} == OBSERVATION_KEYS for o in captured)


def test_policy_first_action_is_independent_of_dataset_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = random_registry()
    req = request(policy="random-legal-v1", max_actions=8, seed=3)
    baseline = generate_episode(req, registry)

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "image.png").write_bytes(b"not-opened")
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if str(dataset) in str(file):
            raise AssertionError(f"推論がDatasetを開きました: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.chdir(tmp_path)
    again = generate_episode(req, registry)
    assert baseline.episode.events[3].to_dict() == again.episode.events[3].to_dict()
    assert dumps_episode(baseline.episode) == dumps_episode(again.episode)


def test_same_seed_prompt_checkpoint_gives_identical_event_sequence() -> None:
    registry = random_registry()

    def run(seed: int):
        return generate_episode(
            request(policy="random-legal-v1", max_actions=40, seed=seed), registry
        )

    first, second, other = run(5), run(5), run(6)
    assert dumps_episode(first.episode) == dumps_episode(second.episode)
    assert first.image.rgba == second.image.rgba
    assert dumps_episode(first.episode) != dumps_episode(other.episode)


def test_inference_module_does_not_import_reward_evaluator() -> None:
    imported: set[str] = set()
    for path in INFERENCE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
    forbidden = ("reward", "RewardEvaluator", "RewardTarget", "dataset", "TargetFeatures")
    assert not [name for name in imported if any(word in name for word in forbidden)]


def test_request_validation_and_policy_errors() -> None:
    registry = scripted_registry()
    with pytest.raises(UnknownPolicyError):
        generate_episode(request(policy="missing"), registry)
    with pytest.raises(DrawRequestError):
        generate_episode(request(max_actions=0), registry)
    with pytest.raises(DrawRequestError):
        generate_episode(request(width=32, height=32), registry)
    with pytest.raises(DrawRequestError):
        generate_episode(request(seed=True), registry)  # type: ignore[arg-type]
    assert scripted_policy(canvas=None).canvas is None
