"""PromptingのPipelineをDatasetの抽出器として包むアダプタの契約、識別、安全な渡し方を確認する。"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from prompting_support import (
    FakeTagger,
    FakeVision,
    make_preprocessors,
    scene_response,
    write_images,
)

import image_drawer.prompting.core as prompting_core
from image_drawer.dataset import (
    DatasetManifest,
    ExtractorRegistry,
    create_default_registry,
    iter_samples,
    register_images,
)
from image_drawer.dataset.ingest import build_prompt, compute_cache_key
from image_drawer.dataset.prompting_adapters import (
    PromptingCaptionExtractor,
    PromptingTagExtractor,
    encode_png,
    register_prompting_extractors,
)
from image_drawer.prompting import OpenAICompatibleVisionProvider, PromptExtractionError
from image_drawer.prompting.tagging import TagNormalizer

URL = "https://api.example.test/v1/chat/completions"


def caption_adapter(*providers, verifier=None, settings="s1"):
    registry = make_preprocessors(captions=list(providers), verifier=verifier)
    return PromptingCaptionExtractor(
        registry.build_caption_pipeline(),
        extractor_id="prompting:caption:p",
        settings_sha256=settings,
    )


def tag_adapter(*providers, **options):
    registry = make_preprocessors(taggers=list(providers))
    return PromptingTagExtractor(
        registry.build_tag_pipeline(),
        extractor_id="prompting:tags:p",
        settings_sha256="s1",
        **options,
    )


def photo(size=(16, 16)):
    return Image.new("RGB", size, (200, 30, 30))


def real(model="m", endpoint=URL, key="k"):
    return OpenAICompatibleVisionProvider("remote", model, endpoint, api_key=key)


def key_of(info_caption, info_tag=None):
    tagger = info_tag or tag_adapter(FakeTagger()).info
    return compute_cache_key("sha", tagger, info_caption)


def test_caption_adapter_satisfies_caption_extractor_protocol():
    adapter = caption_adapter(FakeVision())
    registry = ExtractorRegistry()
    registry.register_captioner(adapter)
    assert registry.captioner("prompting:caption:p") is adapter
    assert adapter.extract(photo()) == "白い背景に赤いリンゴが一つ"
    assert adapter.extract_variants(photo())["short"] == "赤いリンゴ"
    caption, trace = adapter.extract_with_trace(photo())
    assert caption == adapter.extract(photo())
    assert trace["selected_model"] == "vlm" and trace["fallback_used"] is False


def test_tag_adapter_satisfies_tag_extractor_protocol():
    adapter = tag_adapter(FakeTagger())
    registry = ExtractorRegistry()
    registry.register_tagger(adapter)
    tags = registry.tagger("prompting:tags:p").extract(photo())
    assert {"apple", "red"} <= {tag["label"] for tag in tags}
    assert all(type(tag["confidence"]) is float and tag["category"] for tag in tags)


def test_adapter_sends_png_bytes_without_metadata_to_the_provider():
    source = Image.new("RGB", (8, 8), (1, 2, 3))
    source.info["icc_profile"] = b"profile-bytes"
    source.info["dpi"] = (300, 300)
    provider = FakeVision()
    caption_adapter(provider).extract(source)
    data, mime = provider.images[0]
    assert mime == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n")
    for chunk in (b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"iCCP", b"pHYs"):
        assert chunk not in data
    with Image.open(io.BytesIO(data)) as sent:
        assert sent.size == (8, 8) and sent.getpixel((0, 0)) == (1, 2, 3)
        assert not sent.info


def test_cache_key_changes_when_fallback_order_changes():
    a, b = real("m1"), real("m2")
    assert key_of(caption_adapter(a, b).info) != key_of(caption_adapter(b, a).info)
    assert key_of(caption_adapter(a, b).info) == key_of(caption_adapter(a, b).info)


def test_cache_key_changes_when_a_provider_model_or_endpoint_changes():
    base = key_of(caption_adapter(real("m1")).info)
    assert key_of(caption_adapter(real("m2")).info) != base
    other_host = "https://other.example.test/v1/chat/completions"
    assert key_of(caption_adapter(real("m1", other_host)).info) != base
    assert key_of(caption_adapter(real("m1"), real("m2")).info) != base
    tagger_a = tag_adapter(FakeTagger(config_fingerprint="a")).info
    tagger_b = tag_adapter(FakeTagger(config_fingerprint="b")).info
    info = caption_adapter(real()).info
    assert key_of(info, tagger_a) != key_of(info, tagger_b)


def test_cache_key_changes_when_instruction_or_verifier_changes(monkeypatch):
    base = key_of(caption_adapter(real("m1")).info)
    assert key_of(caption_adapter(real("m1"), verifier=real("v")).info) != base
    assert key_of(caption_adapter(real("m1"), verifier=real("v1")).info) != key_of(
        caption_adapter(real("m1"), verifier=real("v2")).info
    )
    monkeypatch.setattr(prompting_core, "CAPTION_INSTRUCTION", "別の指示文")
    assert key_of(caption_adapter(real("m1")).info) != base
    monkeypatch.undo()
    monkeypatch.setattr(prompting_core, "VERIFY_INSTRUCTION", "別の検証文")
    assert key_of(caption_adapter(real("m1")).info) != base


def test_cache_key_changes_when_settings_or_tag_options_change():
    assert key_of(caption_adapter(real(), settings="a").info) != key_of(
        caption_adapter(real(), settings="b").info
    )
    plain = tag_adapter(FakeTagger()).info
    fewer = tag_adapter(FakeTagger(), include_categories=frozenset({"general"})).info
    assert plain.config_hash != fewer.config_hash


def test_cache_key_is_unchanged_by_api_key_value():
    assert key_of(caption_adapter(real(key="secret-one")).info) == key_of(
        caption_adapter(real(key="secret-two")).info
    )
    assert "secret-one" not in repr(real(key="secret-one"))


def test_chain_is_registered_as_one_extractor_id_per_profile():
    registry = ExtractorRegistry()
    preprocessors = make_preprocessors(
        captions=[FakeVision("a", "m1"), FakeVision("b", "m2")],
        taggers=[FakeTagger("t1", "x"), FakeTagger("t2", "y")],
    )
    ids = register_prompting_extractors(registry, preprocessors, settings_sha256="s")
    profile = preprocessors.profile_id
    assert ids == [f"prompting:caption:{profile}", f"prompting:tags:{profile}"]
    described = registry.describe()
    assert [i.id for i in described["caption"]] == [ids[0]]
    assert [i.id for i in described["tag"]] == [ids[1]]
    with pytest.raises(ValueError):
        register_prompting_extractors(registry, preprocessors, settings_sha256="s")


def test_registry_describe_lists_prompting_extractors_with_hashes():
    registry = create_default_registry()
    preprocessors = make_preprocessors(captions=[FakeVision()], taggers=[FakeTagger()])
    register_prompting_extractors(registry, preprocessors, settings_sha256="s")
    for kind, capability in (("caption", "caption"), ("tag", "tag")):
        info = next(i for i in registry.describe()[kind] if i.id.startswith("prompting:"))
        assert info.capabilities == [capability]
        assert len(info.model_hash) == 64 and len(info.config_hash) == 64
        assert info.version.startswith("prompting-adapter-v1:") and info.runtime == "local"
    remote = caption_adapter(real()).info
    assert remote.runtime == "remote"


def test_tag_adapter_keeps_rating_and_copyright_out_of_prompt_categories():
    payload = {"general": {"apple": 0.9}, "copyright": {"some_series": 0.95}, "rating": {"safe": 1}}
    adapter = tag_adapter(FakeTagger(responses=[payload]))
    tags = adapter.extract(photo())
    by_label = {tag["label"]: tag for tag in tags}
    assert by_label["some series"]["category"] == "copyright"
    assert by_label["some series"]["in_prompt"] is False
    assert by_label["apple"]["in_prompt"] is True
    prompt = build_prompt("りんご", tags)
    assert "apple" in prompt and "some series" not in prompt
    _, trace = adapter.extract_with_trace(photo())
    assert trace["rating"] == "safe"
    override = TagNormalizer(category_overrides={"apple": "rating"})
    registry = make_preprocessors(taggers=[FakeTagger()])
    registry.normalizers["o"] = override
    registry.normalizer_id = "o"
    custom = PromptingTagExtractor(
        registry.build_tag_pipeline(), extractor_id="prompting:tags:q", settings_sha256="s"
    )
    assert all(not t["in_prompt"] for t in custom.extract(photo()) if t["category"] == "rating")


def test_prompting_error_becomes_extractor_failed_quality_issue_and_image_is_kept(tmp_path):
    source = write_images(tmp_path / "src", 1)
    refusing = FakeVision(responses=[PromptExtractionError("refused", "モデルが拒否した")])
    registry = create_default_registry()
    preprocessors = make_preprocessors(captions=[refusing], taggers=[FakeTagger()])
    ids = register_prompting_extractors(registry, preprocessors, settings_sha256="s")
    report = register_images(
        source, tmp_path / "ds", registry, tagger_id=ids[1], captioner_id=ids[0]
    )
    image = DatasetManifest.load(tmp_path / "ds").images[0]
    failed = [i for i in image.quality_issues if i.code == "extractor_failed"]
    assert len(failed) == 1 and "all_models_failed" in failed[0].message
    assert "refused" in failed[0].message
    assert report.added == 1 and image.tags


def test_all_providers_failing_leaves_empty_prompt_and_image_is_skipped_by_iter_samples(tmp_path):
    source = write_images(tmp_path / "src", 2)
    broken = FakeVision(responses=[OSError("down")])
    registry = create_default_registry()
    preprocessors = make_preprocessors(
        captions=[broken, FakeVision("b", "m2", responses=[OSError()])]
    )
    ids = register_prompting_extractors(registry, preprocessors, settings_sha256="s")
    register_images(source, tmp_path / "ds", registry, tagger_id="none", captioner_id=ids[0])
    manifest = DatasetManifest.load(tmp_path / "ds")
    assert len(manifest.images) == 2
    assert all(i.training_prompt == "" for i in manifest.images)
    assert all("empty_prompt" in {q.code for q in i.quality_issues} for i in manifest.images)
    assert [
        s for split in ("train", "validation", "test") for s in iter_samples(tmp_path / "ds", split)
    ] == []


def test_png_over_the_limit_fails_instead_of_being_downscaled():
    with pytest.raises(PromptExtractionError) as info:
        encode_png(photo((64, 64)), max_bytes=10)
    assert info.value.kind == "image_too_large"
