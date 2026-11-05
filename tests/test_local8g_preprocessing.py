"""8GB通常設定のモデル列と、同一VLMを使う独立Taggerを検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_drawer.prompting import PromptExtractionError, VisionResponse
from image_drawer.prompting.openai_compatible import OpenAICompatibleVisionProvider
from image_drawer.prompting.registry import PreprocessorRegistry
from image_drawer.prompting.vision_tagger import TAGGING_INSTRUCTION, VisionTaggerProvider

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "docs" / "users" / "preprocessor_settings.example.json"
HIGH = ROOT / "docs" / "users" / "preprocessor_settings.high.example.json"
FIRST = "qwen3vl4b-q4"
SECOND = "joycaption-beta-one-q3"
THIRD = "qwen3vl4b-uncensored-q4"


def test_default_profile_is_8gb_not_high_profile():
    settings = json.loads(STANDARD.read_text(encoding="utf-8"))
    hardware = settings["hardware_profile"]
    assert hardware["target_vram_gb"] == 8
    assert hardware["target_ram_gb"] == 32
    assert hardware["max_loaded_models"] == 1
    assert hardware["context_tokens"] == 4096
    assert hardware["parallel_requests"] == 1
    assert settings["caption_chain"] == [
        "accurate-vlm",
        "joycaption-fallback",
        "nsfw-caption-last",
    ]
    assert settings["tag_chain"] == ["content-tagger", "backup-tagger"]
    assert {model["endpoint"] for model in settings["providers"]} == {
        "http://127.0.0.1:8080/v1/chat/completions"
    }
    registry = PreprocessorRegistry.from_settings_file(STANDARD)
    pipeline = registry.build()
    assert [m.model_id for m in pipeline.caption_pipeline.providers] == [
        FIRST,
        SECOND,
        THIRD,
    ]
    assert [m.model_id for m in pipeline.tag_pipeline.providers] == [FIRST, THIRD]
    assert all(
        isinstance(model, VisionTaggerProvider)
        for model in pipeline.tag_pipeline.providers
    )
    high = PreprocessorRegistry.from_settings_file(HIGH)
    assert high.caption_models["accurate-vlm"].model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert high.caption_models["nsfw-caption-last"].model_id == (
        "Felldude/Qwen3-VL-8B-Instruct-Uncensored-V2"
    )


def test_caption_fallback_and_independent_tagger_in_single_router(monkeypatch):
    requests: list[tuple[str, bool]] = []

    def fake_describe(self, image, mime_type, instruction):
        is_tag = instruction == TAGGING_INSTRUCTION
        requests.append((self.model_id, is_tag))
        if is_tag:
            if self.model_id == FIRST:
                return VisionResponse(refusal=True)
            return VisionResponse(
                json.dumps(
                    {
                        "general": {"red_apple": 0.9, "standing": 0.6},
                        "rating": {"general": 0.99},
                    }
                )
            )
        if self.model_id in (FIRST, SECOND):
            return VisionResponse(refusal=True)
        return VisionResponse("A red apple sits on a white table.")

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    result = PreprocessorRegistry.from_settings_file(STANDARD).build().preprocess(
        b"test image", "image/png"
    )
    assert result.caption.selected_model == THIRD
    assert [a.status for a in result.caption.attempts] == [
        "refused",
        "refused",
        "ok",
    ]
    assert result.tags.model_id == THIRD
    assert [a.status for a in result.tags.attempts] == ["refused", "ok"]
    assert result.formatted_tags == ("red apple", "standing")
    assert result.tags.prepared.normalized.rating == "general"
    assert requests == [
        (FIRST, False),
        (SECOND, False),
        (THIRD, False),
        (FIRST, True),
        (THIRD, True),
    ]


@pytest.mark.parametrize(
    "tag_payload",
    ["", "not-json", '{"general":{"red_apple":2}}'],
)
def test_tagger_invalid_payload_falls_back_without_fake_tags(monkeypatch, tag_payload):
    def fake_describe(self, image, mime_type, instruction):
        if instruction == TAGGING_INSTRUCTION:
            if self.model_id == FIRST:
                return VisionResponse(tag_payload)
            return VisionResponse(refusal=True)
        return VisionResponse(refusal=True)

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    pipeline = PreprocessorRegistry.from_settings_file(STANDARD).build()
    with pytest.raises(PromptExtractionError, match="preprocessing_failed"):
        pipeline.preprocess(b"test image", "image/png")


def test_changing_tagger_instruction_or_model_changes_cache_identity():
    registry = PreprocessorRegistry.from_settings_file(STANDARD)
    taggers = registry.build().tag_pipeline.providers
    assert len(taggers) == 2
    assert taggers[0].config_fingerprint != taggers[1].config_fingerprint
    assert taggers[0].model_id == FIRST
    assert taggers[1].model_id == THIRD
