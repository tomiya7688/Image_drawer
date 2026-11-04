"""JoyCaption自然文のフォールバック、独立Tagger、無効応答を検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_drawer.prompting import (
    ImagePromptPipeline,
    JoyCaptionVisionProvider,
    PromptExtractionError,
    VisionResponse,
)
from image_drawer.prompting.openai_compatible import OpenAICompatibleVisionProvider
from image_drawer.prompting.registry import PreprocessorRegistry
from image_drawer.prompting.tagger_http import HTTPTagProvider

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "docs/users/preprocessor_settings.example.json"
JOYCAPTION_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def test_settings_select_real_joycaption_model():
    registry = PreprocessorRegistry.from_settings_file(SETTINGS)
    assert registry.caption_chain == ("accurate-vlm", "joycaption-fallback")
    primary = registry.caption_models["accurate-vlm"]
    assert primary.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    fallback = registry.caption_models["joycaption-fallback"]
    assert isinstance(fallback, JoyCaptionVisionProvider)
    assert fallback.model_id == JOYCAPTION_ID
    assert fallback.backend.json_mode is False


def test_prose_fallback_and_tagger_run_independently(monkeypatch):
    invoked: list[tuple[str, str]] = []

    def fake_describe(self, image, mime_type, instruction):
        invoked.append((self.model_id, instruction))
        if self.model_id == "Qwen/Qwen3-VL-8B-Instruct":
            return VisionResponse(refusal=True)
        caption = "A red apple rests on a white table. The background is plain white."
        return VisionResponse(caption)

    def fake_tag(self, image, mime_type):
        invoked.append((self.model_id, "Tagger独立処理"))
        return {"tags": {"general": {"red_apple": 0.95}}}

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    monkeypatch.setattr(HTTPTagProvider, "tag", fake_tag)

    registry = PreprocessorRegistry.from_settings_file(SETTINGS)
    result = registry.build().preprocess(b"image bytes", "image/png")
    assert result.caption.selected_model == JOYCAPTION_ID
    assert result.caption.fallback_used is True
    statuses = [attempt.status for attempt in result.caption.attempts]
    assert statuses == ["refused", "ok"]
    assert result.caption.scene.short == "A red apple rests on a white table."
    assert result.caption.scene.standard.endswith("The background is plain white.")
    assert result.caption.scene.detailed == result.caption.scene.standard
    # 自然文から個数や位置関係を独断で構造化せず、原文を保持する。
    assert result.caption.scene.subjects == ()
    assert result.caption.scene.relations == ()
    assert "red apple" in result.formatted_tags
    assert len(invoked) == 3
    assert invoked[1][1].startswith("Write a straightforward caption")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I cannot assist with this request.",
        "{not-json}",
        "descriptive text " * 130,
    ],
)
def test_invalid_caption_is_not_training_label(monkeypatch, raw):
    def fake_describe(self, image, mime_type, instruction):
        return VisionResponse(raw)

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    registry = PreprocessorRegistry.from_settings_file(SETTINGS)
    provider = registry.caption_models["joycaption-fallback"]
    with pytest.raises(PromptExtractionError) as failure:
        ImagePromptPipeline(providers=[provider]).extract(b"image", "image/png")
    assert failure.value.kind == "all_models_failed"


def test_joycaption_disallows_json_mode():
    registry = PreprocessorRegistry()
    with pytest.raises(ValueError, match="json_mode"):
        registry.load_settings(
            {
                "schema_version": "idraw-preprocessor-registry/v1",
                "providers": [
                    {
                        "id": "joy",
                        "kind": "caption",
                        "adapter": "joycaption",
                        "model_id": JOYCAPTION_ID,
                        "endpoint": "http://localhost:8001/v1/chat/completions",
                        "json_mode": True,
                    }
                ],
            }
        )


def test_fingerprint_tracks_model_revision():
    base = {
        "provider_id": "joy",
        "model_id": JOYCAPTION_ID,
        "endpoint": "http://localhost:8001/v1/chat/completions",
    }
    old_backend = OpenAICompatibleVisionProvider(**base, model_revision="a")
    new_backend = OpenAICompatibleVisionProvider(**base, model_revision="b")
    old = JoyCaptionVisionProvider(old_backend)
    new = JoyCaptionVisionProvider(new_backend)
    assert old.config_fingerprint != new.config_fingerprint


def test_json_scene_still_validates(monkeypatch):
    original = {
        "summary": "a red apple",
        "short_prompt": "red apple",
        "standard_prompt": "one red apple on a table",
        "detailed_prompt": "one red apple on a white table",
    }

    def fake_describe(self, image, mime_type, instruction):
        return VisionResponse(json.dumps(original))

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    registry = PreprocessorRegistry.from_settings_file(SETTINGS)
    provider = registry.caption_models["joycaption-fallback"]
    result = ImagePromptPipeline(providers=[provider]).extract(b"image", "image/png")
    assert result.scene.standard == original["standard_prompt"]
