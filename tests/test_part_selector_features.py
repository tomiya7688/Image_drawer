import json

from image_drawer.training import (
    PartSelectorFeatureSchema,
    PartSelectorTrainConfig,
    encode_part_selector_candidate,
    load_part_selector_train_config,
)


def example(prompt="red cat"):
    return {
        "prompt": prompt,
        "context": {
            "category": "face",
            "retrieval_filters": {"split": "train"},
        },
    }


def candidate(tags=None, retrieval_score=0.75):
    return {
        "part_id": "part-a",
        "retrieval_score": retrieval_score,
        "part_metadata": {
            "category": "face",
            "source_image_id": "source-ignored",
            "bbox": [0, 0, 20, 10],
            "tags": tags or ["red", "cat"],
            "attributes": {"pose": "front"},
            "quality": {"usable": True},
            "extraction_method": "fixture",
            "extraction_version": "v1",
            "metadata": {"search_text": "red cat portrait"},
        },
        "embedding_refs": {"mock": "ignored/path"},
        "embedding_identity": {
            "family": "metadata-text",
            "model": "feature-hash",
            "version": "v1",
            "dimensions": 64,
        },
    }


def test_feature_encoding_is_deterministic_and_fixed_size():
    schema = PartSelectorFeatureSchema(text_dimensions=16)

    first = encode_part_selector_candidate(
        example(),
        candidate(),
        schema,
    )
    second = encode_part_selector_candidate(
        example(),
        candidate(),
        schema,
    )

    assert first == second
    assert len(first) == schema.total_dimensions
    assert schema.total_dimensions == 16 * 3 + 3
    assert schema.identity_hash.startswith("sha256:")


def test_feature_encoding_uses_prompt_part_interaction_and_retrieval_score():
    schema = PartSelectorFeatureSchema(text_dimensions=16)
    matching = encode_part_selector_candidate(
        example("red cat"),
        candidate(["red", "cat"], retrieval_score=0.9),
        schema,
    )
    mismatch = encode_part_selector_candidate(
        example("red cat"),
        candidate(["blue", "dog"], retrieval_score=0.1),
        schema,
    )

    assert matching != mismatch
    assert matching[-3] == 0.9
    assert mismatch[-3] == 0.1


def test_feature_encoding_does_not_use_part_or_source_ids():
    schema = PartSelectorFeatureSchema(text_dimensions=8)
    left = candidate()
    right = json.loads(json.dumps(left))
    right["part_id"] = "different-part-id"
    right["part_metadata"]["source_image_id"] = "different-source-id"

    assert encode_part_selector_candidate(
        example(), left, schema
    ) == encode_part_selector_candidate(
        example(), right, schema
    )


def test_feature_schema_hash_changes_when_schema_changes():
    first = PartSelectorFeatureSchema(text_dimensions=8)
    second = PartSelectorFeatureSchema(text_dimensions=16)

    assert first.identity_hash != second.identity_hash


def test_training_config_loads_relative_paths_from_toml(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[training]
id = "fixture"
dataset_path = "dataset.json"
output_dir = "run"
seed = 7
epochs = 3
batch_size = 4
learning_rate = 0.02
hidden_dimensions = [16]
objective = "pairwise"
text_dimensions = 8
top_k = 2
selection_metric = "mrr"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_part_selector_train_config(config_path)

    assert isinstance(config, PartSelectorTrainConfig)
    assert config.dataset_path == str(dataset.resolve())
    assert config.output_dir == str((tmp_path / "run").resolve())
    assert config.hidden_dimensions == [16]
    assert config.seed == 7
