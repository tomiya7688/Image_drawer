from pathlib import Path

from image_drawer.core import Part, SourceImage, StepSpec, WorkflowSpec
from image_drawer.dsl import parse_workflow
from image_drawer.part_bank import (
    BruteForceCosineIndex,
    MetadataHashEmbedder,
    SQLitePartRepository,
)
from image_drawer.runtime import WorkflowRuntime
from image_drawer.steps import create_part_bank_registry


def seed_repository(path: Path) -> tuple[SQLitePartRepository, dict[str, Part]]:
    repository = SQLitePartRepository(path)
    source = SourceImage(
        id="source-fixture",
        uri="source/fixture.png",
        width=64,
        height=64,
        checksum="sha256:fixture",
        dataset="fixture",
        metadata={"split": "train"},
    )
    parts = {
        "cat": Part(
            id="part-cat",
            source_image_id=source.id,
            category="face",
            crop_uri="parts/cat.png",
            bbox=(0, 0, 16, 16),
            tags=["cat", "red"],
            attributes={"animal": "cat"},
            extraction_method="fixture",
            extraction_version="v1",
            metadata={"split": "train", "search_text": "red cat portrait"},
        ),
        "dog": Part(
            id="part-dog",
            source_image_id=source.id,
            category="face",
            crop_uri="parts/dog.png",
            bbox=(16, 0, 16, 16),
            tags=["dog", "blue"],
            attributes={"animal": "dog"},
            extraction_method="fixture",
            extraction_version="v1",
            metadata={"split": "train", "search_text": "blue dog portrait"},
        ),
        "tree": Part(
            id="part-tree",
            source_image_id=source.id,
            category="background",
            crop_uri="parts/tree.png",
            bbox=(0, 16, 32, 32),
            tags=["tree", "green"],
            attributes={"scene": "forest"},
            extraction_method="fixture",
            extraction_version="v1",
            metadata={"split": "train", "search_text": "green tree forest"},
        ),
    }
    repository.store(
        source,
        parts.values(),
        origin_uri="file:///fixture.png",
        provenance={"fixture": True},
    )
    return repository, parts


def test_metadata_hash_embeddings_are_persisted_and_reused(tmp_path):
    repository, parts = seed_repository(tmp_path / "bank.sqlite3")
    embedder = MetadataHashEmbedder(dimensions=256)

    assert repository.ensure_embeddings(embedder) == len(parts)
    assert repository.ensure_embeddings(embedder) == 0
    assert len(repository.list_embeddings(embedder.identity)) == len(parts)
    repository.close()

    with SQLitePartRepository(tmp_path / "bank.sqlite3") as reopened:
        records = reopened.list_embeddings(embedder.identity)
        assert [record.part_id for record in records] == sorted(
            part.id for part in parts.values()
        )
        assert all(len(record.vector) == 256 for record in records)


def test_multiple_embedding_identities_can_coexist(tmp_path):
    repository, parts = seed_repository(tmp_path / "bank.sqlite3")
    first = MetadataHashEmbedder(dimensions=128)
    second = MetadataHashEmbedder(dimensions=64)

    assert repository.ensure_embeddings(first) == len(parts)
    assert repository.ensure_embeddings(second) == len(parts)
    assert len(repository.list_embeddings(first.identity)) == len(parts)
    assert len(repository.list_embeddings(second.identity)) == len(parts)
    assert first.identity.key != second.identity.key
    repository.close()


def test_local_cosine_index_filters_and_orders_hits():
    embedder = MetadataHashEmbedder(dimensions=32)
    index = BruteForceCosineIndex(embedder.identity)
    index.add(
        "cat",
        embedder.embed_text("cat"),
        metadata={"category": "face", "split": "train"},
    )
    index.add(
        "dog",
        embedder.embed_text("dog"),
        metadata={"category": "face", "split": "train"},
    )
    index.add(
        "tree",
        embedder.embed_text("tree"),
        metadata={"category": "background", "split": "train"},
    )

    hits = index.search(
        embedder.embed_text("cat"),
        2,
        filters={"category": "face"},
    )

    assert [hit.part_id for hit in hits] == ["cat", "dog"]
    assert hits[0].score > hits[1].score
    assert index.identity.backend == "bruteforce-cosine"


def test_repository_query_returns_ranked_partset_and_raw_scores(tmp_path):
    repository, parts = seed_repository(tmp_path / "bank.sqlite3")
    embedder = MetadataHashEmbedder(dimensions=256)

    result = repository.query(
        "red cat",
        embedder=embedder,
        category="face",
        top_k=2,
        filters={"split": "train"},
    )

    assert result.part_ids[0] == parts["cat"].id
    assert set(result.part_ids) == {parts["cat"].id, parts["dog"].id}
    assert result.retrieval_scores[0] >= result.retrieval_scores[1]
    assert result.metadata["embedding"]["key"] == embedder.identity.key
    assert result.metadata["index"] == {
        "backend": "bruteforce-cosine",
        "version": "v1",
    }
    assert result.metadata["filters"] == {
        "category": "face",
        "split": "train",
    }
    repository.close()


def test_query_is_deterministic_for_same_bank_and_configuration(tmp_path):
    repository, _ = seed_repository(tmp_path / "bank.sqlite3")
    embedder = MetadataHashEmbedder(dimensions=128)

    first = repository.query(
        "cat",
        embedder=embedder,
        category="face",
        top_k=2,
    )
    second = repository.query(
        "cat",
        embedder=embedder,
        category="face",
        top_k=2,
    )

    assert first == second
    repository.close()


def test_real_retrieve_parts_executes_from_dsl_and_records_versions(tmp_path):
    repository, parts = seed_repository(tmp_path / "bank.sqlite3")
    embedder = MetadataHashEmbedder(dimensions=256)
    registry = create_part_bank_registry(repository, embedder)
    workflow = parse_workflow(
        """INPUT prompt: Text

parts = RETRIEVE_PARTS(
  prompt,
  category="face",
  top=2,
  embedding_model="default",
  filters=["split=train"],
)

OUTPUT parts
""",
        registry,
        workflow_id="real-retrieval",
    )

    result = WorkflowRuntime(registry).execute(
        workflow,
        external_inputs={"prompt": "cat"},
        run_id="real-retrieval-test",
    )

    output = next(iter(result.outputs.values()))
    assert output.artifact_type == "PartSet"
    assert output.metadata["value"][0] == parts["cat"].id
    assert len(output.metadata["retrieval_scores"]) == 2
    part_set = output.metadata["part_set"]
    assert part_set["part_ids"][0] == parts["cat"].id

    identity = result.trajectory.metadata["execution_identity"]["parts"]
    assert identity["embedding"]["key"] == embedder.identity.key
    assert identity["index"] == {
        "backend": "bruteforce-cosine",
        "version": "v1",
    }
    execution = next(
        item for item in result.trajectory.executions if item.step_id == "parts"
    )
    assert execution.backend == "part-bank"
    assert execution.metadata["execution_identity"] == identity
    repository.close()


def test_real_retrieve_parts_rejects_unknown_embedding_model(tmp_path):
    repository, _ = seed_repository(tmp_path / "bank.sqlite3")
    embedder = MetadataHashEmbedder(dimensions=64)
    registry = create_part_bank_registry(repository, embedder)
    workflow = WorkflowSpec(
        id="bad-embedding",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(
                id="parts",
                type="RETRIEVE_PARTS",
                parameters={
                    "category": "face",
                    "top": 2,
                    "embedding_model": "missing:model",
                    "filters": [],
                },
            ),
        ],
        edges=[("prompt", "parts")],
        inputs=["prompt"],
    )

    try:
        WorkflowRuntime(registry).execute(
            workflow,
            external_inputs={"prompt": "cat"},
            run_id="bad-model",
        )
    except Exception as error:
        assert "requested embedding_model is unavailable" in str(error)
    else:
        raise AssertionError("unknown embedding model should fail")
    repository.close()
