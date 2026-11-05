"""S4-3: DRAW_EPISODE / REPLAY_EPISODE Step。水準A（配線）の確認で、モデルの品質は確認しない。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from inference_helpers import SIDE, scripted_registry

from image_drawer.dsl import parse_workflow, serialize_workflow
from image_drawer.runtime import RuntimeExecutionError, WorkflowRuntime, validate_workflow
from image_drawer.runtime.validation import WorkflowValidationError
from image_drawer.steps import (
    DrawIncomplete,
    create_mock_registry,
    register_drawing_steps,
)

WORKFLOW = """INPUT prompt: Text
episode = DRAW_EPISODE(prompt, policy="scripted", max_actions=%d, seed=7, width=16, height=16)
image = REPLAY_EPISODE(episode)
OUTPUT image
"""


def make_registry(tmp_path: Path):
    registry = create_mock_registry()
    register_drawing_steps(registry, policies=scripted_registry(), output_dir=tmp_path / "out")
    return registry


def run(tmp_path: Path, source: str | None = None, max_actions: int = 1000):
    registry = make_registry(tmp_path)
    workflow = parse_workflow(source or WORKFLOW % max_actions, registry)
    result = WorkflowRuntime(registry).execute(
        workflow, external_inputs={"prompt": "猫"}, run_id="run-1"
    )
    return registry, result


def artifacts_by_type(result) -> dict[str, list]:
    found: dict[str, list] = {}
    for artifact in result.trajectory.artifacts:
        found.setdefault(artifact.artifact_type, []).append(artifact)
    return found


def test_draw_episode_schema_declares_text_input_and_episode_output(tmp_path: Path) -> None:
    schema = make_registry(tmp_path).resolve("DRAW_EPISODE").schema
    assert schema.input_types == ("Text",) and schema.output_type == "Episode"
    assert set(schema.parameters) == {"policy", "max_actions", "seed", "width", "height"}
    assert schema.parameters["policy"].required
    replay = make_registry(tmp_path).resolve("REPLAY_EPISODE").schema
    assert replay.input_types == ("Episode",) and replay.output_type == "Image"


def test_draw_episode_returns_episode_primary_and_image_additional_artifact(
    tmp_path: Path,
) -> None:
    _registry, result = run(tmp_path)
    found = artifacts_by_type(result)
    images = [i for i in found["Image"] if i.producing_step_id in {"episode", "image"}]
    assert len(found["Episode"]) == 1 and len(images) == 2
    episode = found["Episode"][0]
    assert episode.id == "artifact:run-1:episode"
    assert Path(episode.uri).name == "episode.json" and Path(episode.uri).is_absolute()
    assert episode.metadata["status"] == "complete" and episode.metadata["policy_trained"] is False
    draw_execution = next(e for e in result.trajectory.executions if e.step_type == "DRAW_EPISODE")
    assert len(draw_execution.output_artifact_ids) == 2
    assert "execution_identity" in draw_execution.metadata


def test_draw_episode_artifact_lineage_text_to_episode_to_image(tmp_path: Path) -> None:
    _registry, result = run(tmp_path)
    found = artifacts_by_type(result)
    text, episode = found["Text"][0], found["Episode"][0]
    drawn = next(i for i in found["Image"] if i.producing_step_id == "episode")
    replayed = next(i for i in found["Image"] if i.producing_step_id == "image")
    assert episode.parent_artifact_ids == [text.id]
    assert drawn.parent_artifact_ids == [episode.id]
    assert replayed.parent_artifact_ids == [episode.id]


def test_draw_episode_trajectory_stores_uri_and_hash_not_events(tmp_path: Path) -> None:
    _registry, result = run(tmp_path)
    text = json.dumps(asdict(result.trajectory), ensure_ascii=False, default=str)
    assert "stroke_v2" not in text and '"events"' not in text
    episode = artifacts_by_type(result)["Episode"][0]
    assert Path(episode.uri).is_file() and len(episode.metadata["episode_sha256"]) == 64
    assert len(text) < 20000


def test_draw_episode_rejects_image_typed_input_at_validation(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    source = (
        'INPUT sketch: Image\nepisode = DRAW_EPISODE(sketch, policy="scripted")\nOUTPUT episode\n'
    )
    with pytest.raises(ValueError):
        validate_workflow(parse_workflow(source, registry), registry)


def test_draw_episode_rejects_unknown_parameters_such_as_strokes(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    source = 'INPUT prompt: Text\nepisode = DRAW_EPISODE(prompt, policy="p", strokes=[1])\nOUTPUT episode\n'
    with pytest.raises(ValueError):
        validate_workflow(parse_workflow(source, registry), registry)
    assert issubclass(WorkflowValidationError, ValueError)


@pytest.mark.parametrize(
    "override",
    ["max_actions=0", "max_actions=20000", "seed=-1", "seed=4294967296", "width=0", "height=1025"],
)
def test_draw_episode_rejects_values_outside_parameter_ranges(
    tmp_path: Path, override: str
) -> None:
    key = override.split("=")[0]
    base = dict(max_actions=10, seed=7, width=16, height=16)
    base.pop(key)
    args = ", ".join(f"{k}={v}" for k, v in base.items())
    source = (
        f'INPUT prompt: Text\nepisode = DRAW_EPISODE(prompt, policy="scripted", {args}, {override})\n'
        "OUTPUT episode\n"
    )
    registry = make_registry(tmp_path)
    workflow = parse_workflow(source, registry)
    with pytest.raises(RuntimeExecutionError):
        WorkflowRuntime(registry).execute(workflow, external_inputs={"prompt": "猫"})


def test_draw_episode_draft_raises_runtime_error_with_draft_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeExecutionError) as caught:
        run(tmp_path, max_actions=5)
    error = caught.value
    assert isinstance(error.__cause__, DrawIncomplete)
    assert error.__cause__.reason == "action_budget_exhausted"
    draft = Path(error.__cause__.draft_path)
    assert draft.name == "draft.json" and draft.is_file()
    assert str(draft) in error.trajectory.executions[-1].error
    assert not [a for a in error.trajectory.artifacts if a.artifact_type == "Episode"]


def test_replay_episode_returns_image_identical_to_drawn_image(tmp_path: Path) -> None:
    _registry, result = run(tmp_path)
    found = artifacts_by_type(result)
    drawn = next(i for i in found["Image"] if i.producing_step_id == "episode")
    replayed = next(i for i in found["Image"] if i.producing_step_id == "image")
    assert replayed.metadata["rgba_sha256"] == drawn.metadata["rgba_sha256"]
    assert replayed.metadata["png_sha256"] == drawn.metadata["png_sha256"]
    assert Path(replayed.uri).read_bytes() == Path(drawn.uri).read_bytes()
    assert (replayed.metadata["width"], replayed.metadata["height"]) == (SIDE, SIDE)


def replay_step_with(tmp_path: Path, mutate):
    registry, result = run(tmp_path)
    episode = artifacts_by_type(result)["Episode"][0]
    mutate(episode)
    step = registry.resolve("REPLAY_EPISODE")
    from image_drawer.steps import StepContext

    context = StepContext("run-2", "image", "inference", {})
    return step.run([episode], {}, context)


def test_replay_episode_fails_when_episode_file_hash_mismatches(tmp_path: Path) -> None:
    def tamper(episode) -> None:
        path = Path(episode.uri)
        path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="Replay不一致"):
        replay_step_with(tmp_path, tamper)


def test_replay_episode_fails_when_replayed_pixels_differ_from_recorded_hash(
    tmp_path: Path,
) -> None:
    def wrong(episode) -> None:
        episode.metadata["final_image_rgba_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="Replay不一致"):
        replay_step_with(tmp_path, wrong)


def test_dsl_text_to_image_workflow_parses_validates_and_roundtrips(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    workflow = parse_workflow(WORKFLOW % 1000, registry)
    validate_workflow(workflow, registry)
    canonical = serialize_workflow(workflow, registry)
    again = parse_workflow(canonical, registry)
    assert serialize_workflow(again, registry) == canonical
    assert again.edges == workflow.edges and "DRAW_EPISODE" in canonical


def test_register_drawing_steps_keeps_part_bank_and_mock_steps(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    schemas = registry.schemas()
    for step_type in (
        "INPUT",
        "RETRIEVE_PARTS",
        "COMPOSE",
        "SELECT",
        "DRAW_EPISODE",
        "REPLAY_EPISODE",
    ):
        assert step_type in schemas
    with pytest.raises(ValueError):
        register_drawing_steps(registry, policies=scripted_registry(), output_dir=tmp_path)


def test_compose_output_cannot_feed_draw_episode(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    source = (
        "INPUT prompt: Text\n"
        "parts = RETRIEVE_PARTS(prompt)\n"
        "draft = COMPOSE(parts)\n"
        'episode = DRAW_EPISODE(draft, policy="scripted")\n'
        "OUTPUT episode\n"
    )
    with pytest.raises(ValueError):
        validate_workflow(parse_workflow(source, registry), registry)
