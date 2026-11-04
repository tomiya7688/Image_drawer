"""PR #78 のレビューで残った指摘(想定外例外・設定エラー・全か無かの読込・rating重複)の回帰テスト。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from image_drawer.prompting import ImagePromptPipeline, PromptExtractionError, VisionResponse
from image_drawer.prompting.cli import main as preprocess_main
from image_drawer.prompting.registry import PreprocessorRegistry
from image_drawer.prompting.tagging import ScoredTag, TagFormatter, TagSet, TaggingPipeline

SCENE = json.dumps(
    {
        "summary": "赤いリンゴ",
        "short_prompt": "赤いリンゴ",
        "standard_prompt": "白い背景の赤いリンゴ",
        "detailed_prompt": "中央に赤いリンゴ",
    },
    ensure_ascii=False,
)


@dataclass
class Vision:
    provider_id: str
    model_id: str
    outcomes: list[object] = field(default_factory=list)
    calls: int = 0

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class Tagger:
    provider_id: str
    model_id: str
    outcomes: list[object] = field(default_factory=list)
    calls: int = 0

    def tag(self, image: bytes, mime_type: str) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.mark.parametrize(
    "error", [ValueError("x"), AttributeError("x"), KeyError("x"), TypeError("x")]
)
def test_caption_pipeline_falls_back_when_a_provider_raises_an_unexpected_exception(error):
    first = Vision("a", "m1", [error])
    second = Vision("b", "m2", [VisionResponse(SCENE)])
    result = ImagePromptPipeline(providers=[first, second]).extract(b"png", "image/png")
    assert result.selected_model == "m2" and result.fallback_used is True
    assert [item.status for item in result.attempts] == ["unexpected_error", "ok"]


def test_caption_pipeline_reports_all_models_failed_when_every_provider_raises_unexpectedly():
    pipeline = ImagePromptPipeline(
        providers=[Vision("a", "m1", [ValueError("x")]), Vision("b", "m2", [OSError("y")])]
    )
    with pytest.raises(PromptExtractionError) as info:
        pipeline.extract(b"png", "image/png")
    assert info.value.kind == "all_models_failed"


@pytest.mark.parametrize("error", [ValueError("x"), AttributeError("x")])
def test_tag_pipeline_falls_back_when_a_provider_raises_an_unexpected_exception(error):
    first = Tagger("a", "t1", [error])
    second = Tagger("b", "t2", [{"tags": {"apple": 0.9}}])
    result = TaggingPipeline([first, second]).extract(b"png", "image/png")
    assert result.model_id == "t2" and result.fallback_used is True
    assert [item.status for item in result.attempts] == ["unexpected_error", "ok"]


def settings(**extra) -> dict:
    document = {
        "schema_version": "idraw-preprocessor-registry/v1",
        "providers": [
            {
                "id": "vlm",
                "adapter": "openai_compatible",
                "model_id": "m",
                "endpoint": "http://127.0.0.1:9/v1/chat/completions",
            },
            {
                "id": "tagger",
                "adapter": "tagger_http",
                "model_id": "t",
                "endpoint": "http://127.0.0.1:9/tag",
            },
        ],
        "caption_chain": ["vlm"],
        "tag_chain": ["tagger"],
    }
    document.update(extra)
    return document


def test_missing_required_provider_key_is_reported_as_a_configuration_error():
    document = settings()
    del document["providers"][1]["model_id"]
    with pytest.raises(ValueError, match="model_id"):
        PreprocessorRegistry().load_settings(document)


def test_load_settings_is_all_or_nothing_when_a_later_step_fails():
    registry = PreprocessorRegistry()
    document = settings(caption_chain=["vlm", "missing"])
    with pytest.raises(ValueError):
        registry.load_settings(document)
    assert registry.caption_models == {} and registry.tag_models == {}
    assert registry.verifiers == {} and "user" not in registry.normalizers
    assert "user" not in registry.formatters and registry.caption_chain == ()
    registry.load_settings(settings())
    assert list(registry.caption_models) == ["vlm"]


def test_load_settings_rolls_back_when_a_provider_entry_is_missing_its_id():
    registry = PreprocessorRegistry()
    document = settings()
    document["providers"].append({"adapter": "openai_compatible", "model_id": "m"})
    with pytest.raises(ValueError, match="id"):
        registry.load_settings(document)
    assert registry.caption_models == {} and registry.tag_models == {}


def test_cli_reports_a_missing_key_as_a_settings_error(tmp_path, capsys):
    document = settings()
    del document["providers"][0]["id"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(SystemExit) as info:
        preprocess_main(["--settings", str(path), "--image", str(image)])
    assert info.value.code in (1, 2)
    assert "KeyError" not in capsys.readouterr().err


def test_prompt_formatter_can_be_selected_in_settings():
    registry = PreprocessorRegistry()
    registry.register_prompt_formatter("custom", lambda scene, tags, seed: "x")
    registry.load_settings(settings(prompt_formatter="custom"))
    assert registry.prompt_formatter_id == "custom"
    assert registry.build().prompt_formatter is registry.prompt_formatters["custom"]


def test_unregistered_prompt_formatter_in_settings_is_rejected_and_rolled_back():
    registry = PreprocessorRegistry()
    with pytest.raises(ValueError, match="prompt formatter"):
        registry.load_settings(settings(prompt_formatter="nope"))
    assert registry.caption_models == {}
    with pytest.raises(ValueError, match="prompt_formatter"):
        registry.load_settings(settings(prompt_formatter=3))


def tag_set(rating: str, *labels: tuple[str, str]) -> TagSet:
    return TagSet(
        tuple(ScoredTag(label, 0.9, category, category) for label, category in labels),
        rating=rating,
    )


def test_formatter_does_not_duplicate_rating_already_present_as_a_tag():
    prepared = TagFormatter(include_rating=True).format(
        tag_set("safe", ("safe", "general"), ("apple", "general"))
    )
    assert prepared.positive_tags.count("safe") == 1
    assert prepared.category_tags["rating"] == ("safe",)


def test_formatter_does_not_add_a_rating_category_tag_twice_or_when_disabled():
    tags = tag_set("safe", ("safe", "rating"), ("apple", "general"))
    on = TagFormatter(include_rating=True).format(tags)
    assert on.positive_tags.count("safe") == 1
    off = TagFormatter(include_rating=False).format(tags)
    assert "safe" not in off.positive_tags
