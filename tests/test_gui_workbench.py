from pathlib import Path

from PIL import Image

from image_drawer.core import (
    Artifact,
    Score,
    StepExecution,
    Trajectory,
    WorkflowSpec,
)
from image_drawer.gui.controller import RunRecord, WorkbenchController
from image_drawer.part_bank import (
    MetadataHashEmbedder,
    SQLitePartRepository,
)
from image_drawer.runtime import RuntimeResult, WorkflowRuntime
from image_drawer.steps import (
    create_mock_registry,
    create_part_bank_registry,
)


def test_parameter_edit_updates_canonical_dsl():
    registry = create_mock_registry()
    controller = WorkbenchController(registry)

    controller.set_parameter_text("parts", "top", "7")

    canonical = controller.canonical_dsl()
    assert "top=7" in canonical
    assert controller.parameter_values("parts")["top"] == 7


def test_apply_valid_dsl_reconstructs_state_and_invalid_dsl_is_reported():
    registry = create_mock_registry()
    controller = WorkbenchController(registry)
    original = controller.canonical_dsl()

    source = """INPUT prompt: Text

parts = RETRIEVE_PARTS(prompt, category="generic", top=3)
selected = SELECT_PARTS(parts, top=1)
draft = COMPOSE(selected)
scores = EVALUATE(draft, evaluator="mock")
OUTPUT draft
"""
    controller.apply_dsl(source)
    assert controller.parameter_values("parts")["top"] == 3
    assert controller.parameter_values("selected")["top"] == 1
    assert controller.validate() is None

    try:
        controller.apply_dsl(
            "INPUT prompt: Text\ndraft = COMPOSE(missing)\nOUTPUT draft\n"
        )
    except Exception as error:
        assert "undefined variable 'missing'" in str(error)
        assert controller.validation_error is not None
    else:
        raise AssertionError("invalid DSL should fail")

    assert controller.canonical_dsl() != original


def test_workflow_load_and_save_round_trip(tmp_path):
    registry = create_mock_registry()
    controller = WorkbenchController(registry)
    controller.set_parameter("parts", "top", 4)

    path = tmp_path / "workflow.dsl"
    controller.save_workflow(path)

    restored = WorkbenchController(registry)
    restored.load_workflow(path)

    assert restored.canonical_dsl() == controller.canonical_dsl()
    assert restored.parameter_values("parts")["top"] == 4


def test_add_delete_and_reorder_steps():
    registry = create_mock_registry()
    controller = WorkbenchController(registry)
    original_ids = controller.step_ids()

    new_id = controller.add_step("EVALUATE", after_step_id="draft")
    assert new_id in controller.step_ids()
    assert controller.step_ids().index(new_id) == (
        controller.step_ids().index("draft") + 1
    )

    controller.move_step(new_id, 1)
    assert controller.step_ids().index(new_id) > (
        controller.step_ids().index("draft")
    )

    controller.delete_step(new_id)
    assert new_id not in controller.step_ids()
    assert set(controller.step_ids()) == set(original_ids)


def test_mock_workflow_runs_and_populates_history_artifacts_scores():
    registry = create_mock_registry()
    controller = WorkbenchController(registry)
    controller.apply_dsl(
        """INPUT prompt: Text
parts = RETRIEVE_PARTS(prompt, category="generic", top=4)
selected = SELECT_PARTS(parts, top=2)
draft = COMPOSE(selected)
scores = EVALUATE(draft, evaluator="mock")
best = SELECT(draft, scores=scores, strategy="best", top=1)
OUTPUT best
"""
    )

    record = controller.run("cat")

    assert record.run_id == "gui-run-0001"
    assert controller.run_history[0] is record
    assert any(row.step_id == "draft" for row in record.artifacts)
    assert len(record.scores) == 1
    assert record.scores[0].evaluator == "mock"
    assert len(record.candidates) == 1
    assert record.candidates[0].status == "selected"
    assert record.candidates[0].artifact_id == "artifact:gui-run-0001:draft"

    second = controller.run("dog")
    assert second.run_id == "gui-run-0002"
    assert [item.run_id for item in controller.run_history] == [
        "gui-run-0002",
        "gui-run-0001",
    ]


def _seed_real_bank(tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    crop = bank / "parts" / "fixture.png"
    crop.parent.mkdir(parents=True)
    image = Image.new("RGBA", (2, 1))
    image.putdata([(255, 0, 0, 255), (0, 255, 0, 255)])
    image.save(crop, format="PNG")
    image.close()

    from image_drawer.core import Part, SourceImage

    source = SourceImage(
        id="source-gui",
        uri="source/gui.png",
        width=2,
        height=1,
        checksum="sha256:gui",
        dataset="gui-fixture",
        metadata={"split": "train"},
    )
    part = Part(
        id="part-gui",
        source_image_id=source.id,
        category="generic",
        crop_uri="parts/fixture.png",
        bbox=(0, 0, 2, 1),
        tags=["generic"],
        extraction_method="fixture",
        extraction_version="v1",
        metadata={"split": "train", "search_text": "generic"},
    )
    repository = SQLitePartRepository(bank / "metadata.sqlite3")
    repository.store(
        source,
        [part],
        origin_uri="file:///gui-fixture.png",
        provenance={"fixture": True},
    )
    return bank, repository, part


def test_real_part_bank_workflow_runs_from_controller(tmp_path):
    bank, repository, part = _seed_real_bank(tmp_path)
    embedder = MetadataHashEmbedder(dimensions=64)
    registry = create_part_bank_registry(
        repository,
        embedder,
        bank_dir=bank,
    )
    controller = WorkbenchController(
        registry,
        WorkflowRuntime(registry),
        bank_dir=bank,
    )
    controller.set_parameter("parts", "top", 1)
    controller.set_parameter("selected", "top", 1)

    record = controller.run("generic")

    draft = next(
        row
        for row in record.artifacts
        if row.step_id == "draft" and row.artifact_type == "Image"
    )
    assert draft.uri is not None
    path = controller.artifact_path(draft)
    assert path is not None and path.is_file()
    with Image.open(path) as rendered:
        rendered.load()
        assert rendered.size == (512, 512)

    assert any(
        row.artifact_type == "Composition" for row in record.artifacts
    )
    assert len(record.scores) == 1
    assert record.candidates[0].artifact_id == draft.id
    assert record.candidates[0].status == "scored"
    repository.close()


def test_run_record_marks_selected_and_rejected_candidates():
    score_a = Score(
        id="score-a",
        evaluator="mock",
        evaluator_version="v1",
        overall=0.9,
        components={"quality": 0.9},
        metadata={"artifact_id": "a"},
    )
    score_b = Score(
        id="score-b",
        evaluator="mock",
        evaluator_version="v1",
        overall=0.2,
        components={"quality": 0.2},
        metadata={"artifact_id": "b"},
    )
    score_set_artifact = Artifact(
        id="score-set-artifact",
        artifact_type="ScoreSet",
        producing_step_id="evaluate",
        scores=[score_a, score_b],
        metadata={
            "score_set": {
                "id": "scores",
                "evaluator": "mock",
                "evaluator_version": "v1",
                "candidate_artifact_ids": ["a", "b"],
                "scores": [score_a.to_dict(), score_b.to_dict()],
                "metadata": {},
            }
        },
    )
    selected = Artifact(
        id="selected-output",
        artifact_type="Image",
        producing_step_id="select",
        metadata={
            "selection": {
                "selected_artifact_ids": ["a"],
                "ranking": ["a", "b"],
            }
        },
    )
    trajectory = Trajectory(
        id="trajectory",
        workflow_id="workflow",
        workflow_version="v1",
        prompt="prompt",
        executions=[
            StepExecution(
                id="execution",
                step_id="select",
                step_type="SELECT",
                output_artifact_ids=[selected.id],
            )
        ],
        artifacts=[score_set_artifact, selected],
        evaluations=[score_a, score_b],
        selections=[
            {
                "step_id": "select",
                "output_artifact_id": selected.id,
                "decision": {
                    "selected_artifact_ids": ["a"],
                    "ranking": ["a", "b"],
                },
            }
        ],
    )
    result = RuntimeResult(
        trajectory=trajectory,
        outputs={"output": selected},
    )

    record = RunRecord.from_result("run", "prompt", result)

    assert [(row.artifact_id, row.status) for row in record.candidates] == [
        ("a", "selected"),
        ("b", "rejected"),
    ]
