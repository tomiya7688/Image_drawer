from pathlib import Path

from PIL import Image

from image_drawer.core import Composition, Part, PartPlacement, SourceImage
from image_drawer.dsl import parse_workflow
from image_drawer.part_bank import (
    MetadataHashEmbedder,
    SQLitePartRepository,
    build_composition,
    fixed_grid_layout,
    render_composition,
    save_png_atomic,
)
from image_drawer.runtime import WorkflowRuntime
from image_drawer.steps import create_part_bank_registry


def _write_rgba(path: Path, size, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size)
    image.putdata(pixels)
    image.save(path, format="PNG")
    image.close()


def _write_mask(path: Path, size, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size)
    image.putdata(values)
    image.save(path, format="PNG")
    image.close()


def seed_render_bank(tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    repository = SQLitePartRepository(bank / "metadata.sqlite3")
    source = SourceImage(
        id="source-render",
        uri="source/render.png",
        width=8,
        height=8,
        checksum="sha256:render-fixture",
        dataset="fixture",
        metadata={"split": "train"},
    )

    red = Part(
        id="part-red",
        source_image_id=source.id,
        category="face",
        crop_uri="parts/red.png",
        bbox=(0, 0, 2, 2),
        tags=["red", "cat"],
        extraction_method="fixture",
        extraction_version="v1",
        metadata={"split": "train", "search_text": "red cat face"},
    )
    blue = Part(
        id="part-blue",
        source_image_id=source.id,
        category="face",
        crop_uri="parts/blue.png",
        bbox=(2, 0, 2, 2),
        tags=["blue", "dog"],
        extraction_method="fixture",
        extraction_version="v1",
        metadata={"split": "train", "search_text": "blue dog face"},
    )
    masked = Part(
        id="part-masked",
        source_image_id=source.id,
        category="detail",
        crop_uri="parts/masked.png",
        mask_uri="parts/masked-mask.png",
        bbox=(0, 2, 2, 2),
        extraction_method="fixture",
        extraction_version="v1",
        metadata={"split": "train"},
    )
    stripe = Part(
        id="part-stripe",
        source_image_id=source.id,
        category="detail",
        crop_uri="parts/stripe.png",
        bbox=(2, 2, 2, 1),
        extraction_method="fixture",
        extraction_version="v1",
        metadata={"split": "train"},
    )

    _write_rgba(
        bank / red.crop_uri,
        (2, 2),
        [(255, 0, 0, 255)] * 4,
    )
    _write_rgba(
        bank / blue.crop_uri,
        (2, 2),
        [(0, 0, 255, 255)] * 4,
    )
    _write_rgba(
        bank / masked.crop_uri,
        (2, 2),
        [(255, 0, 0, 255)] * 4,
    )
    _write_mask(
        bank / masked.mask_uri,
        (2, 2),
        [255, 0, 255, 0],
    )
    _write_rgba(
        bank / stripe.crop_uri,
        (2, 1),
        [(255, 255, 255, 255), (255, 255, 255, 255)],
    )

    repository.store(
        source,
        [red, blue, masked, stripe],
        origin_uri="file:///render-fixture.png",
        provenance={"fixture": True},
    )
    return bank, repository, {
        "red": red,
        "blue": blue,
        "masked": masked,
        "stripe": stripe,
    }


def test_fixed_grid_layout_produces_deterministic_pixels(tmp_path):
    bank, repository, parts = seed_render_bank(tmp_path)
    selected = [parts["red"], parts["blue"]]
    layout = fixed_grid_layout(
        selected,
        canvas_width=4,
        canvas_height=2,
    )
    composition = build_composition(
        selected,
        layout,
        background="transparent",
    )

    first = render_composition(
        composition,
        {part.id: part for part in selected},
        bank,
    )
    second = render_composition(
        composition,
        {part.id: part for part in selected},
        bank,
    )
    try:
        assert first.tobytes() == second.tobytes()
        assert list(first.getdata()) == [
            (255, 0, 0, 255),
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (0, 0, 255, 255),
        ]

        first_path = tmp_path / "first.png"
        second_path = tmp_path / "second.png"
        first_digest = save_png_atomic(first, first_path)
        second_digest = save_png_atomic(second, second_path)
        assert first_digest == second_digest
        assert first_path.read_bytes() == second_path.read_bytes()
    finally:
        first.close()
        second.close()
        repository.close()


def test_z_order_places_later_higher_layer_on_top(tmp_path):
    bank, repository, parts = seed_render_bank(tmp_path)
    composition = Composition(
        id="z-order",
        canvas_size=(2, 2),
        placements=[
            PartPlacement(
                id="red-low",
                part_id=parts["red"].id,
                x=0,
                y=0,
                z_index=0,
            ),
            PartPlacement(
                id="blue-high",
                part_id=parts["blue"].id,
                x=0,
                y=0,
                z_index=10,
            ),
        ],
        background="transparent",
    )

    image = render_composition(
        composition,
        {
            parts["red"].id: parts["red"],
            parts["blue"].id: parts["blue"],
        },
        bank,
    )
    try:
        assert set(image.getdata()) == {(0, 0, 255, 255)}
    finally:
        image.close()
        repository.close()


def test_mask_scale_and_opacity_are_applied(tmp_path):
    bank, repository, parts = seed_render_bank(tmp_path)
    composition = Composition(
        id="masked-transform",
        canvas_size=(4, 4),
        placements=[
            PartPlacement(
                id="masked",
                part_id=parts["masked"].id,
                x=0,
                y=0,
                scale_x=2.0,
                scale_y=2.0,
                opacity=0.5,
            )
        ],
        background="transparent",
    )

    image = render_composition(
        composition,
        {parts["masked"].id: parts["masked"]},
        bank,
    )
    try:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0))[3] in (127, 128)
        assert image.getpixel((3, 0))[3] == 0
        assert image.getpixel((0, 3))[3] in (127, 128)
        assert image.getpixel((3, 3))[3] == 0
    finally:
        image.close()
        repository.close()


def test_rotation_and_translation_have_expected_alpha_bounds(tmp_path):
    bank, repository, parts = seed_render_bank(tmp_path)
    composition = Composition(
        id="rotate",
        canvas_size=(6, 6),
        placements=[
            PartPlacement(
                id="stripe",
                part_id=parts["stripe"].id,
                x=2,
                y=1,
                scale_x=2.0,
                scale_y=1.0,
                rotation=90.0,
            )
        ],
        background="transparent",
    )

    image = render_composition(
        composition,
        {parts["stripe"].id: parts["stripe"]},
        bank,
    )
    try:
        assert image.getchannel("A").getbbox() == (2, 1, 3, 5)
    finally:
        image.close()
        repository.close()


def test_real_compose_step_emits_image_and_structured_composition(tmp_path):
    bank, repository, parts = seed_render_bank(tmp_path)
    embedder = MetadataHashEmbedder(dimensions=256)
    registry = create_part_bank_registry(
        repository,
        embedder,
        bank_dir=bank,
    )
    workflow = parse_workflow(
        """INPUT prompt: Text

parts = RETRIEVE_PARTS(
  prompt,
  category="face",
  top=2,
)

draft = COMPOSE(
  parts,
  canvas_width=4,
  canvas_height=2,
  background="transparent",
  spacing=0,
  layout_mode="fixed_grid",
)

OUTPUT draft
""",
        registry,
        workflow_id="compose-runtime",
    )

    result = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "red cat"},
        run_id="compose-test",
    )

    compose_execution = next(
        execution
        for execution in result.trajectory.executions
        if execution.step_id == "draft"
    )
    assert len(compose_execution.output_artifact_ids) == 2

    compose_artifacts = [
        artifact
        for artifact in result.trajectory.artifacts
        if artifact.producing_step_id == "draft"
    ]
    image_artifact = next(
        artifact for artifact in compose_artifacts if artifact.artifact_type == "Image"
    )
    structure_artifact = next(
        artifact
        for artifact in compose_artifacts
        if artifact.artifact_type == "Composition"
    )

    assert image_artifact.parent_artifact_ids == [structure_artifact.id]
    assert structure_artifact.parent_artifact_ids
    assert set(structure_artifact.metadata["selected_part_ids"]) == {
        parts["red"].id,
        parts["blue"].id,
    }
    assert image_artifact.metadata["lineage"]["selected_part_ids"] == (
        structure_artifact.metadata["selected_part_ids"]
    )
    assert structure_artifact.metadata["composition"]["placements"]
    assert structure_artifact.metadata["layout"]["metadata"]["mode"] == "fixed_grid"

    rendered_path = bank / image_artifact.uri
    assert rendered_path.is_file()
    with Image.open(rendered_path) as rendered:
        rendered.load()
        assert rendered.size == (4, 2)

    identity = result.trajectory.metadata["execution_identity"]["draft"]
    assert identity == {
        "renderer": {
            "backend": "pillow",
            "version": "v1",
        }
    }
    repository.close()


def test_real_compose_is_reproducible_across_run_ids(tmp_path):
    bank, repository, _ = seed_render_bank(tmp_path)
    embedder = MetadataHashEmbedder(dimensions=128)
    registry = create_part_bank_registry(
        repository,
        embedder,
        bank_dir=bank,
    )
    workflow = parse_workflow(
        """INPUT prompt: Text
parts = RETRIEVE_PARTS(prompt, category="face", top=2)
draft = COMPOSE(parts, canvas_width=4, canvas_height=2)
OUTPUT draft
""",
        registry,
    )

    first = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "cat"},
        run_id="run-one",
    )
    second = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "cat"},
        run_id="run-two",
    )
    first_image = next(
        artifact
        for artifact in first.trajectory.artifacts
        if artifact.producing_step_id == "draft"
        and artifact.artifact_type == "Image"
    )
    second_image = next(
        artifact
        for artifact in second.trajectory.artifacts
        if artifact.producing_step_id == "draft"
        and artifact.artifact_type == "Image"
    )

    assert first_image.uri != second_image.uri
    assert first_image.metadata["checksum"] == second_image.metadata["checksum"]
    assert (bank / first_image.uri).read_bytes() == (
        bank / second_image.uri
    ).read_bytes()
    repository.close()
