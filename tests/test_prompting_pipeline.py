"""画像だけのPrompt生成、拒否時のQwen切替、テキスト整合を検証する。"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from urllib.error import HTTPError

import pytest

from image_drawer.prompting import (
    ImagePromptPipeline,
    OpenAICompatibleVisionProvider,
    PromptExtractionError,
    VisionResponse,
    normalize_prompt,
)
import image_drawer.prompting.openai_compatible as compat


def caption_json(**overrides) -> str:
    doc = {
        "summary": "白い背景の赤いリンゴ",
        "subjects": [
            {"name": "リンゴ", "count": 1, "attributes": ["赤い"]},
        ],
        "relations": [],
        "background": "白い背景",
        "composition": "中央",
        "style": "シンプルなイラスト",
        "tags": ["apple", "red apple", "apple"],
        "short_prompt": "赤いリンゴ",
        "standard_prompt": "白い背景に赤いリンゴが一つ",
        "detailed_prompt": "中央に一つの赤いリンゴ、白背景のシンプルなイラスト",
    }
    doc.update(overrides)
    return json.dumps(doc, ensure_ascii=False)


@dataclass
class FakeVision:
    provider_id: str
    model_id: str
    responses: list[VisionResponse | Exception] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        assert image == b"image bytes"
        assert mime_type == "image/png"
        self.instructions.append(instruction)
        if not self.responses:
            raise AssertionError("予期しないモデル呼び出し")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_pipeline(
    primary: VisionResponse | Exception,
    fallback: VisionResponse | Exception | None = None,
    verifier: FakeVision | None = None,
) -> tuple[ImagePromptPipeline, FakeVision, FakeVision]:
    a = FakeVision("accurate", "best-vlm", [primary])
    b = FakeVision(
        "qwen_local", "Qwen3-VL-8B-Instruct", [fallback or VisionResponse(caption_json())]
    )
    return ImagePromptPipeline(a, b, verifier=verifier), a, b


def test_primary_is_preferred_and_prompt_lengths_are_available():
    router, primary, fallback = make_pipeline(VisionResponse(caption_json()))
    result = router.extract(b"image bytes", "image/png")
    assert result.selected_model == "best-vlm"
    assert result.fallback_used is False
    assert [x.status for x in result.attempts] == ["ok"]
    assert len(primary.instructions) == 1
    assert fallback.instructions == []
    assert set(result.scene.tags) == {"apple", "red apple"}
    outputs = {result.training_prompt(seed) for seed in range(100)}
    assert outputs == {
        result.scene.short,
        result.scene.standard,
        result.scene.detailed,
    }


@pytest.mark.parametrize(
    "primary_reply",
    [
        VisionResponse(refusal=True),
        VisionResponse(finish_reason="content_filter"),
        VisionResponse("申し訳ありませんが、この画像を解析できません"),
        VisionResponse(""),
        VisionResponse("not json"),
        VisionResponse(caption_json(standard_prompt="an image")),
    ],
)
def test_primary_refusal_or_failure_uses_qwen_automatically(primary_reply):
    router, primary, fallback = make_pipeline(primary_reply)
    result = router.extract(b"image bytes", "image/png")
    assert result.fallback_used is True
    assert result.selected_model == "Qwen3-VL-8B-Instruct"
    assert len(primary.instructions) == len(fallback.instructions) == 1
    assert result.attempts[0].status in {"refused", "empty", "invalid_format"}
    assert result.attempts[1].status == "ok"


def test_quality_verifier_rejection_triggers_fallback():
    verifier = FakeVision(
        "independent_checker",
        "checker-vlm",
        [
            VisionResponse('{"supported":false,"unsupported_claims":["wrong count"]}'),
            VisionResponse('{"supported":true,"unsupported_claims":[],"missing_facts":[]}'),
        ],
    )
    router, _, _ = make_pipeline(VisionResponse(caption_json()), verifier=verifier)
    result = router.extract(b"image bytes", "image/png")
    assert result.fallback_used is True
    assert result.verifier_status == "verified_by_model"
    assert [a.status for a in result.attempts] == ["unverified_facts", "ok"]
    assert len(verifier.instructions) == 2


def test_both_models_failed_never_generates_generic_fake_prompt():
    router, _, _ = make_pipeline(
        VisionResponse(refusal=True),
        VisionResponse(finish_reason="content_filter"),
    )
    with pytest.raises(PromptExtractionError) as exc:
        router.extract(b"image bytes", "image/png")
    assert exc.value.kind == "all_models_failed"
    assert "accurate:refused" in str(exc.value)
    assert "qwen_local:refused" in str(exc.value)


def test_unavailable_fallback_is_configurable():
    primary = FakeVision("accurate", "best-vlm", [PromptExtractionError("unavailable", "timeout")])
    backup = FakeVision("qwen", "qwen-vl", [VisionResponse(caption_json())])
    router = ImagePromptPipeline(primary, backup, fallback_on_unavailable=False)
    with pytest.raises(PromptExtractionError, match="accurate:unavailable"):
        router.extract(b"image bytes", "image/png")
    assert backup.instructions == []


def test_cache_key_tracks_model_and_extraction_profile():
    a, _, _ = make_pipeline(VisionResponse(caption_json()))
    b, _, _ = make_pipeline(VisionResponse(caption_json()))
    c = ImagePromptPipeline(
        FakeVision("accurate", "best-vlm", [VisionResponse(caption_json())]),
        FakeVision("qwen", "qwen-vl", [VisionResponse(caption_json())]),
    )
    assert a.extract(b"image bytes", "image/png").cache_key == (
        b.extract(b"image bytes", "image/png").cache_key
    )
    assert a._cache_key(b"image bytes", "image/png") != c._cache_key(b"image bytes", "image/png")


def test_invalid_subject_count_cannot_enter_training():
    router, _, _ = make_pipeline(
        VisionResponse(caption_json(subjects=[{"name": "猫", "count": True}]))
    )
    result = router.extract(b"image bytes", "image/png")
    assert result.fallback_used is True
    assert result.attempts[0].status == "invalid_format"


def test_human_prompt_normalization_preserves_negation_and_order():
    assert normalize_prompt("  青い犬  が   猫の左にいる。  影なし  ") == (
        "青い犬 が 猫の左にいる。 影なし"
    )
    with pytest.raises(ValueError):
        normalize_prompt("   ")


def test_only_image_bytes_and_mime_are_needed():
    router, _, _ = make_pipeline(VisionResponse(caption_json()))
    for invalid in [(b"", "image/png"), (b"a", "text/plain")]:
        with pytest.raises(ValueError):
            router.extract(*invalid)


def test_openai_compatible_endpoint_restricts_non_tls_remote_hosts():
    with pytest.raises(ValueError):
        OpenAICompatibleVisionProvider(
            "primary",
            "some-vlm",
            "http://example.com/v1/chat/completions",
        )
    local = OpenAICompatibleVisionProvider(
        "qwen_local",
        "qwen3-vl:8b",
        "http://127.0.0.1:11434/v1/chat/completions",
    )
    assert local.model_id == "qwen3-vl:8b"


def test_http_adapter_sends_image_and_recognizes_structured_refusal(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n):
            return json.dumps(
                {"choices": [{"message": {"refusal": "content_filter"}, "finish_reason": "stop"}]}
            ).encode()

    captured = {}

    def fake_open(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(compat, "urlopen", fake_open)
    backend = OpenAICompatibleVisionProvider(
        "primary",
        "accurate-vlm",
        "https://example.com/v1/chat/completions",
        json_mode=True,
    )
    response = backend.describe(b"image bytes", "image/png", "describe")
    assert response.refusal is True
    assert captured["timeout"] == 120
    payload = captured["body"]
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert payload["response_format"] == {"type": "json_object"}


def test_http_adapter_http_content_filter_is_detected(monkeypatch):
    def fake_open(request, timeout):
        data = json.dumps(
            {"error": {"code": "content_policy_violation", "message": "refused"}}
        ).encode()
        raise HTTPError(
            url=request.full_url,
            code=400,
            msg="rejected",
            hdrs={},
            fp=io.BytesIO(data),
        )

    monkeypatch.setattr(compat, "urlopen", fake_open)
    backend = OpenAICompatibleVisionProvider(
        "primary",
        "accurate-vlm",
        "https://example.com/v1/chat/completions",
    )
    assert backend.describe(b"image bytes", "image/png", "describe").refusal is True


def test_subject_color_and_count_evidence_are_preserved():
    router, _, _ = make_pipeline(VisionResponse(caption_json()))
    scene = router.extract(b"image bytes", "image/png").scene
    assert scene.subjects[0].name == "リンゴ"
    assert scene.subjects[0].count == 1
    assert scene.subjects[0].attributes == ("赤い",)


def test_vision_provider_connection_settings_change_cache_fingerprint():
    a = OpenAICompatibleVisionProvider(
        "primary", "accurate-vlm", "https://example.com/v1/chat/completions"
    )
    b = OpenAICompatibleVisionProvider(
        "primary",
        "accurate-vlm",
        "https://example.com/v1/chat/completions",
        model_revision="revision-2",
    )
    assert a.config_fingerprint != b.config_fingerprint
    assert "api_key" not in a.config_fingerprint
