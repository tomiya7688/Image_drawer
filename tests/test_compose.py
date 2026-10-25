"""M4 COMPOSE: 実crop画素・layout・provenance・DSLを検証する。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pytest
from PIL import Image

from image_drawer.core import (
    Artifact, Part, SourceImage, StepSpec, WorkflowSpec, build_artifact_lineage,
)
from image_drawer.dsl import DslSyntaxError, parse_workflow, serialize_workflow
from image_drawer.part_bank import (
    MetadataHashEmbedder, SQLitePartRepository, ingest_directory,
)
from image_drawer.runtime import WorkflowRuntime
from image_drawer.steps import PartBankComposeStep, StepContext, StepResult, create_part_bank_registry


# {
#   責務: [
#     file_from_uri: file URIをOSのパスへ変換する。Windowsの"/C:/..."形式も正しく扱う
#   ]
#   引数: [
#     uri: Artifactに記録された file:// のURI
#   ]
#   戻り値: [
#     path: 実ファイルのパス
#   ]
# }
def file_from_uri(uri: str) -> Path:
    parsed = urlsplit(uri)
    assert parsed.scheme == "file"
    return Path(url2pathname(parsed.path))


@pytest.fixture
def seeded_bank(tmp_path):
    """予測可能なcropとmaskを直接保存した小さなPart Bank。"""
    bank = tmp_path / "bank"
    (bank / "parts").mkdir(parents=True)
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(bank / "parts/red.png")
    Image.new("RGBA", (2, 2), (0, 0, 255, 255)).save(bank / "parts/blue.png")
    mask = Image.new("L", (2, 2), 255)
    mask.putpixel((0, 0), 0)
    mask.putpixel((0, 1), 0)
    mask.save(bank / "parts/mask.png")
    source = SourceImage(
        id="source-test", uri="source/test.png", width=2, height=2,
        checksum="sha256:seed", dataset="tests",
    )
    parts = [
        Part(id="red", source_image_id=source.id, category="front",
             crop_uri="parts/red.png", bbox=(0, 0, 2, 2)),
        Part(id="blue", source_image_id=source.id, category="front",
             crop_uri="parts/blue.png", mask_uri="parts/mask.png",
             bbox=(0, 0, 2, 2)),
    ]
    with SQLitePartRepository(bank / "metadata.sqlite3") as repository:
        repository.store(source, parts, origin_uri="file:///test.png")
        yield repository, bank


def run_compose(repository, *, slots=None, background=None, ids=None, size=(4, 4)):
    context = StepContext(
        run_id="compose-test", step_id="compose", backend="pillow", external_inputs={},
    )
    selected = Artifact(
        id="selected", artifact_type="PartSet",
        metadata={"value": ids if ids is not None else ["red", "blue"]},
    )
    params = {
        "canvas_width": size[0],
        "canvas_height": size[1],
        "background": background if background is not None else [255, 255, 255, 255],
        "slots": slots if slots is not None else [],
    }
    return PartBankComposeStep(repository).run([selected], params, context)


def test_manual_layout_mask_z_order_and_content_addressing(seeded_bank):
    repository, bank = seeded_bank
    slots = [
        {"part_id": "red", "x": 0, "y": 0, "width": 4, "height": 4,
         "z_index": 1},
        {"part_id": "blue", "x": 1, "y": 1, "z_index": 2},
    ]
    result = run_compose(repository, slots=slots)
    repeated = run_compose(repository, slots=slots)

    assert isinstance(result, StepResult)
    assert len(result.additional_artifacts) == 1
    image = result.output
    composition_artifact = result.additional_artifacts[0]
    assert image.artifact_type == "Image"
    assert composition_artifact.artifact_type == "Composition"
    assert image.parent_artifact_ids == [composition_artifact.id]
    assert composition_artifact.parent_artifact_ids == ["selected"]
    assert image.metadata["composition_artifact_id"] == composition_artifact.id
    assert image.metadata["part_ids"] == ["red", "blue"]
    assert image.uri == repeated.output.uri
    assert file_from_uri(image.uri).is_relative_to(bank / "runs" / "renders")
    assert image.metadata["checksum"] == (
        "sha256:" + hashlib.sha256(file_from_uri(image.uri).read_bytes()).hexdigest()
    )
    structured = composition_artifact.metadata["composition"]
    assert structured["placements"][0]["part_id"] == "red"
    assert structured["placements"][0]["scale_x"] == 2.0
    assert structured["placements"][1]["z_index"] == 2
    assert structured["metadata"]["part_sources"]["blue"]["mask_uri"] == "parts/mask.png"
    assert composition_artifact.metadata["layout"]["metadata"]["mode"] == "manual"
    with Image.open(file_from_uri(image.uri)) as canvas:
        assert canvas.size == (4, 4)
        assert canvas.getpixel((1, 1)) == (255, 0, 0, 255)  # maskで青を除外
        assert canvas.getpixel((2, 1)) == (0, 0, 255, 255)

    reversed_z = [dict(slots[0], z_index=5), dict(slots[1], z_index=0)]
    with Image.open(file_from_uri(run_compose(repository, slots=reversed_z).output.uri)) as canvas:
        assert canvas.getpixel((2, 1)) == (255, 0, 0, 255)


def test_rotation_opacity_clipping_and_unselected_part(seeded_bank):
    repository, bank = seeded_bank
    # 赤の左上だけを残した非対称cropで、90度回転を画素で検証する。
    marker = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    marker.putpixel((0, 0), (255, 0, 0, 255))
    marker.save(bank / "parts/red.png")
    result = run_compose(
        repository,
        ids=["red", "blue"],
        slots=[{"part_id": "red", "x": 1, "y": 1, "rotation": 90}],
        background=[0, 0, 0, 0],
    )
    structured = result.additional_artifacts[0].metadata["composition"]
    assert structured["metadata"]["unused_part_ids"] == ["blue"]
    with Image.open(file_from_uri(result.output.uri)) as canvas:
        assert canvas.getpixel((1, 1))[3] == 0
        assert canvas.getpixel((1, 2)) == (255, 0, 0, 255)

    # 左側をはみ出したlayerを安全にclipする。
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(bank / "parts/red.png")
    half = run_compose(
        repository, ids=["red"],
        slots=[{"x": -1, "y": 0, "opacity": 0.5}],
    )
    with Image.open(file_from_uri(half.output.uri)) as canvas:
        assert canvas.getpixel((0, 0)) == (255, 127, 127, 255)
        assert canvas.getpixel((1, 0)) == (255, 255, 255, 255)


def test_invalid_layout_is_rejected_before_writing(seeded_bank):
    repository, bank = seeded_bank
    invalid = [
        {"slots": [{"part_id": "unknown", "x": 0, "y": 0}]},
        {"slots": [{"category": "missing", "x": 0, "y": 0}]},
        {"slots": [{"part_id": "red", "x": 0, "y": 0, "opacity": 1.1}]},
        {"slots": [{"part_id": "red", "x": 0, "y": 0, "scale_x": True}]},
        {"slots": [{"part_id": "red", "x": 0, "y": 0, "width": 2}]},
        {"slots": [{"part_id": "red", "x": 0, "y": 0, "height": -1, "width": 2}]},
        {"slots": [{"part_id": "red", "x": 0, "y": 0, "rotation": float("nan")}]},
        {"size": (0, 4)},
        {"background": [256, 0, 0, 255]},
        {"ids": []},
    ]
    for options in invalid:
        with pytest.raises(ValueError):
            run_compose(repository, **options)
    assert not (bank / "runs").exists()


def test_part_crop_path_cannot_escape_part_bank(seeded_bank):
    repository, bank = seeded_bank
    source = repository.get_source("source-test")
    repository.store(
        source,
        [Part(id="escape", source_image_id=source.id,
              category="front", crop_uri="../outside.png", bbox=(0, 0, 2, 2))],
        origin_uri="file:///additional.png",
    )
    with pytest.raises(ValueError, match="within the Part Bank"):
        run_compose(repository, ids=["escape"])
    assert not (bank / "runs").exists()


def test_real_ingest_retrieve_select_compose_output_and_lineage(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    bank = tmp_path / "bank"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(source / "red.png")
    Image.new("RGB", (2, 2), (0, 0, 255)).save(source / "blue.png")
    report = ingest_directory(source, bank, dataset="composition-fixture")
    assert report.parts_added == 2 and not report.errors

    with SQLitePartRepository(bank / "metadata.sqlite3") as repository:
        registry = create_part_bank_registry(
            repository, MetadataHashEmbedder(dimensions=64),
        )
        workflow = WorkflowSpec(
            id="m4-end-to-end", version="v1",
            nodes=[
                StepSpec(id="prompt", type="INPUT"),
                StepSpec(id="parts", type="RETRIEVE_PARTS",
                         parameters={"category": "generic", "top": 2}),
                StepSpec(id="selected", type="SELECT_PARTS", parameters={"top": 2}),
                StepSpec(id="compose", type="COMPOSE",
                         parameters={"canvas_width": 8, "canvas_height": 4}),
                StepSpec(id="output", type="OUTPUT"),
            ],
            edges=[
                ("prompt", "parts"), ("parts", "selected"),
                ("selected", "compose"), ("compose", "output"),
            ],
            inputs=["prompt"], outputs=["output"],
        )
        runtime = WorkflowRuntime(registry)
        result = runtime.execute(
            workflow, external_inputs={"prompt": "color"},
            run_id="m4-e2e",
        )
        again = runtime.execute(
            workflow, external_inputs={"prompt": "color"},
            run_id="m4-e2e",
        )
        output = result.outputs["output"]
        assert output.artifact_type == "Image"
        assert output.uri == again.outputs["output"].uri
        assert len(result.trajectory.artifacts) == len(workflow.nodes) + 1
        compose_execution = next(
            item for item in result.trajectory.executions if item.step_id == "compose"
        )
        assert len(compose_execution.output_artifact_ids) == 2
        artifacts = {item.id: item for item in result.trajectory.artifacts}
        composition_artifact = next(
            item for item in artifacts.values() if item.artifact_type == "Composition"
        )
        assert artifacts[compose_execution.output_artifact_ids[0]] == composition_artifact
        assert artifacts[compose_execution.output_artifact_ids[1]].artifact_type == "Image"
        assert composition_artifact.parent_artifact_ids == ["artifact:m4-e2e:selected"]
        lineage = build_artifact_lineage(result.trajectory.artifacts, output.id)
        assert composition_artifact.id in lineage
        assert "artifact:m4-e2e:selected" in lineage
        selected = artifacts["artifact:m4-e2e:selected"].metadata["value"]
        assert composition_artifact.metadata["part_ids"] == selected
        assert output.metadata["part_ids"] == selected
        with Image.open(file_from_uri(output.uri)) as rendered:
            assert rendered.size == (8, 4)
            for i, part_id in enumerate(selected):
                with Image.open(bank / repository.get(part_id).crop_uri) as crop:
                    assert rendered.getpixel((i * 4 + 1, 1)) == crop.getpixel((0, 0))


def test_nested_dsl_layout_parameters_round_trip(seeded_bank):
    repository, _ = seeded_bank
    registry = create_part_bank_registry(
        repository, MetadataHashEmbedder(dimensions=32),
    )
    source = """INPUT prompt: Text
parts = RETRIEVE_PARTS(prompt, top=2)
draft = COMPOSE(parts, canvas_width=4, canvas_height=4,
    slots=[{"x": 1, "y": 2, "part_id": "red", "rotation": 90, "opacity": 0.5}])
OUTPUT draft
"""
    workflow = parse_workflow(source, registry)
    canonical = serialize_workflow(workflow, registry)
    assert 'slots=[{"opacity": 0.5, "part_id": "red", "rotation": 90, "x": 1, "y": 2}]' in canonical
    assert serialize_workflow(parse_workflow(canonical, registry), registry) == canonical

    with pytest.raises(DslSyntaxError):
        parse_workflow(source.replace('"x": 1', '"x": 1 + 2'), registry)
    with pytest.raises(DslSyntaxError):
        parse_workflow(source.replace('"x": 1', '1: 1'), registry)
