"""画像だけでCaptionとTagを独立生成し、任意N段とカテゴリ別整形を試す。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from image_drawer.prompting import (
    ImagePromptPipeline,
    PromptExtractionError,
    VisionResponse,
)
from image_drawer.prompting.registry import (
    ImageTrainingPreprocessor,
    PreprocessorRegistry,
)
from image_drawer.prompting.tagging import TagFormatter, TagNormalizer, TaggingPipeline


@dataclass
class FakeTagger:
    provider_id: str
    model_id: str
    responses: list[object] = field(default_factory=list)
    calls: int = 0

    def tag(self, image: bytes, mime_type: str) -> object:
        self.calls += 1
        assert image == b"image bytes"
        assert mime_type == "image/png"
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@dataclass
class FakeCaption:
    provider_id: str
    model_id: str
    replies: list[object] = field(default_factory=list)
    calls: int = 0

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        self.calls += 1
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def sample_caption() -> VisionResponse:
    return VisionResponse(
        json.dumps(
            {
                "summary": "赤いリンゴ",
                "short_prompt": "赤いリンゴ",
                "standard_prompt": "白背景にリンゴが一つ",
                "detailed_prompt": "中央の赤いリンゴ、一つ、白背景",
                "subjects": [{"name": "リンゴ", "count": 1, "attributes": ["赤"]}],
                "relations": [],
                "tags": [],
            },
            ensure_ascii=False,
        )
    )


def test_tagger_normalizes_pixai_and_keeps_confidence_and_emoticons():
    normalizer = TagNormalizer(min_confidence=0.35)
    tagged = normalizer.normalize(
        {"tag": {"red_dress": 0.9, "red dress": 0.65, "^_^": 0.9, "blurry": 0.2}}
    )
    labels = [tag.label for tag in tagged.tags]
    assert labels == ["red_dress", "red dress", "^_^"]
    assert tagged.tags[0].confidence == 0.9
    assert tagged.tags[0].category == "clothes"
    prepared = TagFormatter().format(tagged)
    assert prepared.positive_tags == ("red dress", "^_^")
    assert "red dress" in prepared.category_tags["clothes"]


def test_categorized_tagger_separates_rating_character_copyright_style():
    payload = {
        "tags": {
            "general": {"blue_dress": 0.91, "smile": 0.94, "standing": 0.93},
            "character": {"named_character": 0.91, "false_character": 0.4},
            "copyright": {"some_series": 0.99},
            "style": {"watercolor": 0.88},
        },
        "rating": {"general": 0.7, "explicit": 0.3},
        "caption": "元サービスが付けた文章（Captionerの代わりには使わない）",
    }
    normalized = TagNormalizer().normalize(payload)
    assert normalized.caption.startswith("元サービス")
    assert normalized.rating == "general"
    grouped = normalized.grouped()
    assert set(grouped) == {"clothes", "expression", "pose", "character", "copyright", "style"}
    assert [t.label for t in grouped["character"]] == ["named_character"]
    formatted = TagFormatter().format(normalized)
    assert "some series" not in formatted.positive_tags
    assert "general" not in formatted.positive_tags
    assert "watercolor" not in formatted.positive_tags
    assert "blue dress" in formatted.positive_tags
    assert "some series" in formatted.category_tags["copyright"]
    assert TagFormatter(include_style=True).format(normalized).positive_tags[-1] == "watercolor"


def test_category_override_supports_user_defined_names():
    normalizer = TagNormalizer(
        extra_categories=("appearance",),
        category_overrides={"red_hair": "appearance"},
    )
    result = normalizer.normalize({"general": {"red_hair": 0.91, "sky": 0.6}})
    assert result.tags[0].category == "appearance"
    assert result.tags[1].category == "background"
    with pytest.raises(ValueError):
        TagNormalizer(category_overrides={"red_hair": "unregistered_custom"})


@pytest.mark.parametrize(
    "invalid",
    [
        {"tag": {"x": True}},
        {"tag": {"x": float("nan")}},
        {"tags": [{"name": "x", "confidence": 2}]},
        {"tags": []},
        {"error": "not supported"},
    ],
)
def test_bad_tagger_results_are_not_silently_accepted(invalid):
    with pytest.raises(PromptExtractionError):
        TagNormalizer().normalize(invalid)


def test_empty_then_refused_then_good_tagger_third_priority_is_used():
    taggers = [
        FakeTagger("t1", "tag-accurate", [{"tags": []}]),
        FakeTagger("t2", "tag-balanced", [PromptExtractionError("refused", "no")]),
        FakeTagger("t3", "tag-wide-coverage", [{"tags": ["red_dress", "standing"]}]),
    ]
    result = TaggingPipeline(taggers).extract(b"image bytes", "image/png")
    assert result.fallback_used is True
    assert result.model_id == "tag-wide-coverage"
    assert [a.status for a in result.attempts] == ["empty", "refused", "ok"]
    assert [tag.category for tag in result.prepared.normalized.tags] == ["clothes", "pose"]
    assert [tagger.calls for tagger in taggers] == [1, 1, 1]


def test_caption_models_are_unlimited_in_number_and_not_qwen_specific():
    models = [
        FakeCaption("c1", "excellent", [VisionResponse(refusal=True)]),
        FakeCaption("c2", "balanced", [VisionResponse("")]),
        FakeCaption("c3", "broad", [PromptExtractionError("unavailable", "down")]),
        FakeCaption("c4", "local-anything", [sample_caption()]),
    ]
    result = ImagePromptPipeline(providers=models).extract(b"image bytes", "image/png")
    assert result.selected_model == "local-anything"
    assert [a.status for a in result.attempts] == [
        "refused",
        "empty",
        "unavailable",
        "ok",
    ]
    assert all(model.calls == 1 for model in models)


def test_both_caption_and_tagger_run_on_one_image_without_sidecar():
    caption = FakeCaption("c1", "caption-model", [sample_caption()])
    tagger = FakeTagger("t1", "tag-model", [{"general": ["red_dress", "standing"]}])
    pipeline = ImageTrainingPreprocessor(
        ImagePromptPipeline(providers=[caption]),
        TaggingPipeline([tagger]),
    )
    result = pipeline.preprocess(b"image bytes", "image/png")
    assert caption.calls == tagger.calls == 1
    assert result.training_prompt(1)
    assert result.formatted_tags == ("red dress", "standing")
    assert result.tag_categories["pose"] == ("standing",)


def test_independent_job_failure_does_not_skip_other_job():
    caption = FakeCaption("c1", "restricted", [VisionResponse(refusal=True)])
    tagger = FakeTagger("t1", "tag-model", [{"tag": {"cat": 0.9}}])
    pipeline = ImageTrainingPreprocessor(
        ImagePromptPipeline(providers=[caption]),
        TaggingPipeline([tagger]),
    )
    with pytest.raises(PromptExtractionError, match="caption:all_models_failed"):
        pipeline.preprocess(b"image bytes", "image/png")
    assert caption.calls == tagger.calls == 1


def test_user_runtime_registry_can_register_models_and_custom_formatters():
    registry = PreprocessorRegistry()
    registry.register_caption("caption", FakeCaption("c1", "v1", [sample_caption()]))
    registry.register_tagger("tags", FakeTagger("t1", "tagger", [{"tags": ["apple"]}]))
    registry.register_tag_formatter("style", TagFormatter(include_style=True))
    registry.register_prompt_formatter(
        "custom",
        lambda scene, tags, seed: scene.short + " / " + ", ".join(tags.positive_tags),
    )
    registry.configure(
        caption_chain=["caption"],
        tag_chain=["tags"],
        tag_formatter_id="style",
        prompt_formatter_id="custom",
    )
    result = registry.build().preprocess(b"image bytes", "image/png")
    assert result.training_prompt(5) == "赤いリンゴ / apple"


def test_json_user_settings_allow_three_caption_models_two_taggers_and_verifier(tmp_path):
    settings = {
        "schema_version": "idraw-preprocessor-registry/v1",
        "providers": [
            *[
                {
                    "id": f"caption{i}",
                    "kind": "caption",
                    "adapter": "openai_compatible",
                    "model_id": f"model{i}",
                    "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                }
                for i in range(3)
            ],
            {
                "id": "checker",
                "kind": "verifier",
                "adapter": "openai_compatible",
                "model_id": "checker-model",
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            },
            *[
                {
                    "id": f"tagger{i}",
                    "kind": "tagger",
                    "adapter": "tagger_http",
                    "model_id": f"tags{i}",
                    "endpoint": "http://127.0.0.1:7861/tagger/v1/interrogate",
                }
                for i in range(2)
            ],
        ],
        "caption_chain": ["caption0", "caption1", "caption2"],
        "tag_chain": ["tagger0", "tagger1"],
        "verifier_id": "checker",
        "tag_normalizer": {
            "category_overrides": {"red_hair": "appearance"},
            "extra_categories": ["appearance"],
        },
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    registry = PreprocessorRegistry.from_settings_file(path)
    pipeline = registry.build()
    assert len(pipeline.caption_pipeline.providers) == 3
    assert len(pipeline.tag_pipeline.providers) == 2
    assert pipeline.caption_pipeline.verifier.model_id == "checker-model"
    assert registry.normalizers["user"].classify("red_hair", "general") == "appearance"


def test_plain_json_config_cannot_import_arbitrary_code_or_contain_api_key():
    registry = PreprocessorRegistry()
    with pytest.raises(ValueError, match="APIキー"):
        registry.load_settings(
            {
                "schema_version": "idraw-preprocessor-registry/v1",
                "providers": [
                    {
                        "id": "bad",
                        "adapter": "openai_compatible",
                        "api_key": "do_not_embed_keys",
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="未登録"):
        registry.load_settings(
            {
                "schema_version": "idraw-preprocessor-registry/v1",
                "providers": [
                    {
                        "id": "module",
                        "adapter": "python:arbitrary.module",
                        "kind": "caption",
                    }
                ],
            }
        )
