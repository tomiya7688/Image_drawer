"""実際のJoyCaptionの自然文形式を既存のCaption契約に適合させる検証。"""

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

SETTINGS = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "users"
    / "preprocessor_settings.example.json"
)


def test_standard_settings_select_real_joycaption_model():
    registry = PreprocessorRegistry.from_settings_file(SETTINGS)
    assert registry.caption_chain == ("accurate-vlm", "joycaption-fallback")
    assert registry.caption_models["accurate-vlm"].model_id == "Qwen/Qwen3-VL-8B-Instruct"
    joycaption = registry.caption_models["joycaption-fallback"]
    assert isinstance(joycaption, JoyCaptionVisionProvider)
    assert joycaption.model_id == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    assert joycaption.backend.json_mode is False


def test_joycaption_prose_fallback_and_tagger_run_independently(monkeypatch):
    invoked: list[tuple[str, str]] = []

    def fake_describe(self, image, mime_type, instruction):
        invoked.append((self.model_id, instruction))
        if self.model_id == "Qwen/Qwen3-VL-8B-Instruct":
            return VisionResponse(refusal=True)
        return VisionResponse(
            "A red apple rests on a white table. The background is plain white."
        )

    def fake_tag(self, image, mime_type):
        invoked.append((self.model_id, "Tagger独立処理"))
        return {"tags": {"general": {"red_apple": 0.95}}}

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    monkeypatch.setattr(HTTPTagProvider, "tag", fake_tag)

    result = PreprocessorRegistry.from_settings_file(SETTINGS).build().preprocess(
        b"image bytes", "image/png"
    )
    assert result.caption.selected_model == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    assert result.caption.fallback_used is True
    assert [attempt.status for attempt in result.caption.attempts] == ["refused", "ok"]
    assert result.caption.scene.short == "A red apple rests on a white table."
    assert result.caption.scene.standard.endswith("The background is plain white.")
    assert result.caption.scene.detailed == result.caption.scene.standard
    # 説明文からsubjects等の「事実」を独断で追加しない。
    assert result.caption.scene.subjects == ()
    assert result.caption.scene.relations == ()
    assert "red apple" in result.formatted_tags
    assert len(invoked) == 3
    assert invoked[1][1].startswith("Write a straightforward caption")


@pytest.mark.parametrize(
    ("raw", "expected_kind"),
    [
        ("", "all_models_failed"),
        ("I cannot assist with this request.", "all_models_failed"),
        ("{not-json}", "all_models_failed"),
        ("descriptive text " * 130, "all_models_failed"),
    ],
)
def test_invalid_joycaption_responses_do_not_become_training_labels(
    monkeypatch, raw, expected_kind
):
    def fake_describe(self, image, mime_type, instruction):
        return VisionResponse(raw)

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    registry = PreprocessorRegistry.from_settings_file(SETTINGS)
    provider = registry.caption_models["joycaption-fallback"]
    with pytest.raises(PromptExtractionError) as failure:
        ImagePromptPipeline(providers=[provider]).extract(b"image", "image/png")
    assert failure.value.kind == expected_kind


def test_joycaption_rejects_json_mode_settings():
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
                        "model_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
                        "endpoint": "http://localhost:8001/v1/chat/completions",
                        "json_mode": True,
                    }
                ],
            }
        )


def test_joycaption_fingerprint_distinguishes_model_revision():
    base = {
        "provider_id": "joy",
        "model_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "endpoint": "http://localhost:8001/v1/chat/completions",
    }
    old = JoyCaptionVisionProvider(OpenAICompatibleVisionProvider(**base, model_revision="a"))
    new = JoyCaptionVisionProvider(OpenAICompatibleVisionProvider(**base, model_revision="b"))
    assert old.config_fingerprint != new.config_fingerprint


def test_joycaption_adapter_preserves_existing_json_scene(monkeypatch):
    original = {
        "summary": "a red apple",
        "short_prompt": "red apple",
        "standard_prompt": "one red apple on a table",
        "detailed_prompt": "one red apple on a white table",
    }

    def fake_describe(self, image, mime_type, instruction):
        return VisionResponse(json.dumps(original))

    monkeypatch.setattr(OpenAICompatibleVisionProvider, "describe", fake_describe)
    provider = PreprocessorRegistry.from_settings_file(SETTINGS).caption_models[
        "joycaption-fallback"
    ]
    result = ImagePromptPipeline(providers=[provider]).extract(b"image", "image/png")
    assert result.scene.standard == original["standard_prompt"]
