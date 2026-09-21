from image_drawer.core import (
    Artifact,
    Composition,
    Layout,
    Part,
    PartPlacement,
    PartSet,
    Score,
    SourceImage,
    StepExecution,
    StepSpec,
    Trajectory,
    WorkflowSpec,
    build_artifact_lineage,
)


def make_models():
    score = Score(
        id="score-1",
        evaluator="mock",
        evaluator_version="v1",
        overall=0.75,
        components={"composition": 0.8, "quality": 0.7},
        metadata={"weight_set": "default"},
    )
    source = SourceImage(
        id="source-1",
        uri="data/source/1.png",
        width=640,
        height=480,
        checksum="abc123",
        dataset="fixture",
        metadata={"split": "train"},
    )
    part = Part(
        id="part-1",
        source_image_id=source.id,
        category="generic",
        crop_uri="data/parts/1.png",
        bbox=(10, 20, 100, 120),
        mask_uri="data/masks/1.png",
        embedding_refs={"mock-v1": "data/embeddings/1.npy"},
        tags=["fixture"],
        attributes={"pose": "front"},
        quality={"usable": True},
        extraction_method="fixture",
        extraction_version="v1",
    )
    part_set = PartSet(
        id="part-set-1",
        query_id="query-1",
        category="generic",
        part_ids=[part.id],
        retrieval_scores=[0.93],
    )
    layout = Layout(
        id="layout-1",
        canvas_width=512,
        canvas_height=512,
        slots=[{"category": "generic", "x": 0, "y": 0, "width": 512, "height": 512}],
    )
    placement = PartPlacement(
        id="placement-1",
        part_id=part.id,
        x=12.5,
        y=24.0,
        scale_x=0.9,
        scale_y=1.1,
        rotation=5.0,
        z_index=2,
        opacity=0.8,
    )
    composition = Composition(
        id="composition-1",
        canvas_size=(512, 512),
        placements=[placement],
        background={"kind": "solid", "value": "#ffffff"},
    )
    step = StepSpec(
        id="step-1",
        type="COMPOSE",
        backend="mock",
        parameters={"count": 1},
    )
    workflow = WorkflowSpec(
        id="workflow-1",
        version="v1",
        nodes=[step],
        edges=[("input", step.id), (step.id, "output")],
        inputs=["input"],
        outputs=["output"],
    )
    artifact = Artifact(
        id="artifact-1",
        artifact_type="Image",
        uri="runs/1/final.png",
        parent_artifact_ids=["artifact-0"],
        producing_step_id=step.id,
        backend="mock",
        model="mock-compose",
        version="v1",
        seed=7,
        scores=[score],
    )
    execution = StepExecution(
        id="execution-1",
        step_id=step.id,
        step_type=step.type,
        backend=step.backend,
        parameters=step.parameters,
        input_artifact_ids=["artifact-0"],
        output_artifact_ids=[artifact.id],
        seed=7,
        start_time="2026-09-21T00:00:00Z",
        duration_seconds=0.125,
    )
    trajectory = Trajectory(
        id="trajectory-1",
        workflow_id=workflow.id,
        workflow_version=workflow.version,
        prompt="fixture prompt",
        executions=[execution],
        artifacts=[artifact],
        evaluations=[score],
        selections=[{"artifact_id": artifact.id, "selected": True}],
        final_output_ids=[artifact.id],
        timing={"duration_seconds": 0.125},
    )
    return [
        score,
        artifact,
        source,
        part,
        part_set,
        layout,
        placement,
        composition,
        step,
        workflow,
        execution,
        trajectory,
    ]


def test_all_core_models_round_trip_through_json():
    for model in make_models():
        restored = type(model).from_json(model.to_json())
        assert restored == model


def test_tuple_fields_are_json_arrays_and_restored_as_tuples():
    part = make_models()[3]
    encoded = part.to_dict()

    assert encoded["bbox"] == [10, 20, 100, 120]
    restored = Part.from_dict(encoded)
    assert restored.bbox == (10, 20, 100, 120)


def test_missing_optional_fields_use_defaults():
    artifact = Artifact.from_dict(
        {
            "id": "artifact-minimal",
            "artifact_type": "Image",
        }
    )

    assert artifact.uri is None
    assert artifact.parent_artifact_ids == []
    assert artifact.scores == []
    assert artifact.metadata == {}


def test_score_components_remain_structured():
    score = Score(
        id="score-structured",
        evaluator="mock",
        overall=0.5,
        components={"composition": 0.7, "color": 0.3},
    )

    restored = Score.from_json(score.to_json())
    assert restored.components == {"composition": 0.7, "color": 0.3}
    assert restored.overall == 0.5


def test_part_set_requires_parallel_ids_and_scores():
    try:
        PartSet(
            id="part-set-invalid",
            query_id="query",
            part_ids=["part-1", "part-2"],
            retrieval_scores=[0.9],
        )
    except ValueError as error:
        assert "equal length" in str(error)
    else:
        raise AssertionError("PartSet should reject mismatched ids and scores")


def test_artifact_lineage_is_reconstructed():
    root = Artifact(id="a", artifact_type="Text")
    middle = Artifact(
        id="b",
        artifact_type="PartSet",
        parent_artifact_ids=[root.id],
    )
    final = Artifact(
        id="c",
        artifact_type="Image",
        parent_artifact_ids=[middle.id, root.id],
    )

    lineage = build_artifact_lineage([root, middle, final], final.id)

    assert lineage == {
        "c": ("b", "a"),
        "b": ("a",),
        "a": (),
    }


def test_artifact_lineage_rejects_missing_parent():
    final = Artifact(
        id="c",
        artifact_type="Image",
        parent_artifact_ids=["missing"],
    )

    try:
        build_artifact_lineage([final], final.id)
    except KeyError as error:
        assert "missing parent artifact" in str(error)
    else:
        raise AssertionError("missing lineage parent should be reported")
