"""Datasetの任意拡張(trace・変種・PromptBuilder v2・失敗を保存しない)をpromptingの抽出器で確認する。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import fields

from prompting_support import FakeTagger, FakeVision, make_preprocessors, write_images

from image_drawer.dataset import (
    DatasetManifest,
    PolicyInput,
    TrainingImage,
    create_default_registry,
    iter_samples,
    register_images,
)
from image_drawer.dataset.manifest import choose_variant
from image_drawer.dataset.prompting_adapters import register_prompting_extractors

SPLITS = ("train", "validation", "test")


def prompting_registry(vision=None, tagger=None):
    registry = create_default_registry()
    preprocessors = make_preprocessors(
        captions=[vision or FakeVision()], taggers=[tagger or FakeTagger()]
    )
    ids = register_prompting_extractors(registry, preprocessors, settings_sha256="s")
    return registry, ids[0], ids[1]


def register(tmp_path, registry, tagger_id, captioner_id, count=3):
    source = write_images(tmp_path / "src", count)
    return register_images(
        source, tmp_path / "ds", registry, tagger_id=tagger_id, captioner_id=captioner_id
    )


def all_samples(dataset, **options):
    return [s for split in SPLITS for s in iter_samples(dataset, split, **options)]


def test_register_images_with_prompting_extractors_writes_trace_and_variants(tmp_path):
    registry, captioner, tagger = prompting_registry()
    report = register(tmp_path, registry, tagger, captioner)
    assert report.added == 3 and report.errors == []
    manifest = DatasetManifest.load(tmp_path / "ds")
    image = manifest.images[0]
    assert set(image.caption_variants) == {"short", "standard", "detailed"}
    assert image.caption == image.caption_variants["standard"]
    trace = image.extraction_trace
    assert trace["caption"]["selected_model"] == "vlm" and trace["caption"]["attempts"]
    assert trace["caption"]["egress"] == {"sent": False, "hosts": []}
    assert trace["tags"]["selected_model"] == "tagger"
    assert {tag["category"] for tag in image.tags} >= {"general", "copyright"}


def test_prompt_builder_v2_does_not_append_tags_when_variants_exist(tmp_path):
    registry, captioner, tagger = prompting_registry()
    register(tmp_path, registry, tagger, captioner)
    manifest = DatasetManifest.load(tmp_path / "ds")
    assert manifest.prompt_builder_version == "prompt-builder-v2"
    for image in manifest.images:
        assert image.prompt_builder_version == "prompt-builder-v2"
        assert image.training_prompt == image.caption_variants["standard"]
        assert "apple" not in image.training_prompt and image.tags


def test_baseline_extractors_still_use_prompt_builder_v1(tmp_path):
    registry = create_default_registry()
    register(tmp_path, registry, "color-layout-tags", "color-layout-caption")
    manifest = DatasetManifest.load(tmp_path / "ds")
    assert manifest.prompt_builder_version == "prompt-builder-v1"
    for image in manifest.images:
        assert image.prompt_builder_version == "prompt-builder-v1"
        assert image.caption_variants == {} and image.extraction_trace == {}
        assert "（" in image.training_prompt


def test_iter_samples_default_contract_is_unchanged(tmp_path):
    registry, captioner, tagger = prompting_registry()
    register(tmp_path, registry, tagger, captioner)
    samples = all_samples(tmp_path / "ds")
    assert len(samples) == 3
    policy, target = samples[0]
    assert isinstance(policy, PolicyInput) and policy.prompt == "白い背景に赤いリンゴが一つ"
    assert [f.name for f in fields(policy)] == ["image_id", "prompt"]
    assert target.image_path.endswith(".png")


def test_iter_samples_variant_seed_selects_prompt_variants_deterministically(tmp_path):
    registry, captioner, tagger = prompting_registry()
    register(tmp_path, registry, tagger, captioner, count=6)
    first = [p.prompt for p, _ in all_samples(tmp_path / "ds", variant_seed=7)]
    again = [p.prompt for p, _ in all_samples(tmp_path / "ds", variant_seed=7)]
    assert first == again
    short = [p.prompt for p, _ in all_samples(tmp_path / "ds", prompt_variant="short")]
    assert set(short) == {"赤いリンゴ"}
    assert set(first) <= {
        "赤いリンゴ",
        "白い背景に赤いリンゴが一つ",
        "中央に一つの赤いリンゴ、白背景のシンプルなイラスト",
    }
    assert any(
        [p.prompt for p, _ in all_samples(tmp_path / "ds", variant_seed=seed)] != first
        for seed in range(8, 40)
    )


def test_iter_samples_variant_distribution_is_close_to_40_40_20():
    counts = Counter(choose_variant(3, f"img_{index}") for index in range(5000))
    assert abs(counts["short"] / 5000 - 0.4) < 0.03
    assert abs(counts["standard"] / 5000 - 0.4) < 0.03
    assert abs(counts["detailed"] / 5000 - 0.2) < 0.03
    assert choose_variant(3, "img_1") == choose_variant(3, "img_1")


def test_failed_extraction_is_not_cached_and_is_retried_on_next_register(tmp_path):
    broken = FakeVision(responses=[OSError("down")])
    registry, captioner, tagger = prompting_registry(vision=broken)
    register(tmp_path, registry, tagger, captioner, count=2)
    assert list((tmp_path / "ds" / "cache").glob("*.json")) == []
    failed = DatasetManifest.load(tmp_path / "ds").images
    assert all(i.caption == "" and i.caption_variants == {} for i in failed)
    assert all("extractor_failed" in {q.code for q in i.quality_issues} for i in failed)
    healthy = FakeVision()
    registry, captioner, tagger = prompting_registry(vision=healthy)
    report = register(tmp_path, registry, tagger, captioner, count=2)
    assert healthy.calls == 2 and report.added == 0
    assert len(list((tmp_path / "ds" / "cache").glob("*.json"))) == 2
    assert len(all_samples(tmp_path / "ds")) == 2


def test_successful_extraction_is_cached_and_providers_are_not_called_again(tmp_path):
    vision, tagger_model = FakeVision(), FakeTagger()
    registry, captioner, tagger = prompting_registry(vision, tagger_model)
    register(tmp_path, registry, tagger, captioner)
    assert vision.calls == 3 and tagger_model.calls == 3
    vision2, tagger2 = FakeVision(), FakeTagger()
    registry, captioner, tagger = prompting_registry(vision2, tagger2)
    register(tmp_path, registry, tagger, captioner)
    assert vision2.calls == 0 and tagger2.calls == 0
    cached = DatasetManifest.load(tmp_path / "ds").images[0]
    assert cached.caption_variants and cached.extraction_trace


def test_old_manifest_without_new_fields_still_loads(tmp_path):
    registry = create_default_registry()
    register(tmp_path, registry, "color-layout-tags", "color-layout-caption")
    path = tmp_path / "ds" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = data["dataset_hash"]
    for image in data["images"]:
        del image["caption_variants"], image["extraction_trace"]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    manifest = DatasetManifest.load(tmp_path / "ds")
    assert manifest.dataset_hash == stored_hash
    assert all(isinstance(i, TrainingImage) and i.caption_variants == {} for i in manifest.images)


def test_target_image_stays_out_of_policy_input_with_prompting_extractors(tmp_path):
    registry, captioner, tagger = prompting_registry()
    register(tmp_path, registry, tagger, captioner)
    for policy, target in all_samples(tmp_path / "ds", variant_seed=1):
        assert set(vars(policy)) == {"image_id", "prompt"} if hasattr(policy, "__dict__") else True
        assert target.image_path not in policy.prompt and target.image_sha256 not in policy.prompt
        assert not hasattr(policy, "image_path")


def test_tagger_can_be_disabled_with_none_while_caption_chain_is_used(tmp_path):
    registry = create_default_registry()
    preprocessors = make_preprocessors(captions=[FakeVision()])
    ids = register_prompting_extractors(registry, preprocessors, settings_sha256="s")
    assert ids == [f"prompting:caption:{preprocessors.profile_id}"]
    register(tmp_path, registry, "none", ids[0])
    manifest = DatasetManifest.load(tmp_path / "ds")
    assert manifest.tagger.id == "none"
    assert all(i.tags == [] and i.training_prompt for i in manifest.images)
    assert all("empty_tags" not in {q.code for q in i.quality_issues} for i in manifest.images)
