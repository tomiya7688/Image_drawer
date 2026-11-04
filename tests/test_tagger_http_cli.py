"""ユーザー設定で登録したTagger HTTPアダプタと画像だけのCLI入出力。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import URLError

import pytest

import image_drawer.prompting.cli as cli
import image_drawer.prompting.tagger_http as http
from image_drawer.prompting import (
    ImagePromptPipeline,
    ImageTrainingPreprocessor,
    PromptExtractionError,
    TaggingPipeline,
    VisionResponse,
)
from image_drawer.prompting.tagger_http import HTTPTagProvider


class FakeHTTPResponse:
    def __init__(self, body: object) -> None:
        self.payload = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size: int) -> bytes:
        return self.payload[:size]


def test_tagger_http_adapter_sends_image_and_category_response(monkeypatch):
    captured: dict = {}

    def fake_open(request, timeout):
        captured["request"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeHTTPResponse({"tags": {"general": {"red_dress": 0.92}}})

    monkeypatch.setattr(http, "urlopen", fake_open)
    provider = HTTPTagProvider(
        "t1", "pixai-model", "http://127.0.0.1:7862/pixai/v1/interrogate",
        backend="pixai_http",
    )
    result = TaggingPipeline([provider]).extract(b"\x89PNG\r\n\x1a\nbytes", "image/png")
    assert captured["request"]["model"] == "pixai-model"
    assert captured["request"]["threshold"] == 0.35
    assert captured["request"]["character_threshold"] == 0.85
    assert result.prepared.positive_tags == ("red dress",)
    assert result.prepared.normalized.tags[0].confidence == 0.92


def test_tagger_http_disallows_remote_http_and_handles_unavailable(monkeypatch):
    with pytest.raises(ValueError):
        HTTPTagProvider("x", "tagger", "http://public.example/tagger/v1/interrogate")

    def unavailable(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(http, "urlopen", unavailable)
    tagger = HTTPTagProvider(
        "x", "tagger", "http://localhost:7861/tagger/v1/interrogate"
    )
    with pytest.raises(PromptExtractionError, match="x:unavailable"):
        TaggingPipeline([tagger]).extract(b"image", "image/png")


@dataclass
class FakeVision:
    provider_id: str = "v"
    model_id: str = "caption-v1"

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        return VisionResponse(
            json.dumps(
                {
                    "summary": "赤いリンゴ",
                    "standard_prompt": "白背景に赤いリンゴが1つ",
                    "short_prompt": "赤いリンゴ",
                    "detailed_prompt": "中央に赤いリンゴが1つ、白い背景",
                    "subjects": [{"name": "リンゴ", "count": 1}],
                },
                ensure_ascii=False,
            )
        )


@dataclass
class FakeTagger:
    provider_id: str = "t"
    model_id: str = "tagger-v1"

    def tag(self, image: bytes, mime_type: str) -> object:
        return {
            "general": {"red_apple": 0.95, "standing": 0.38},
            "rating": {"general": 0.7},
        }


def test_cli_image_only_outputs_separate_caption_and_classified_tag(monkeypatch, tmp_path, capsys):
    image = tmp_path / "example.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"non-actual-png-test-fixture")
    pipeline = ImageTrainingPreprocessor(
        ImagePromptPipeline(providers=[FakeVision()]),
        TaggingPipeline([FakeTagger()]),
    )

    class FakeRegistry:
        def build(self):
            return pipeline

    monkeypatch.setattr(
        cli.PreprocessorRegistry,
        "from_settings_file",
        lambda path: FakeRegistry(),
    )
    code = cli.main(["--settings", "user-settings.json", "--image", str(image)])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["caption"]["model_id"] == "caption-v1"
    assert result["tagger"]["model_id"] == "tagger-v1"
    assert result["tagger"]["rating"] == "general"
    assert "red apple" in result["tagger"]["positive_tags"]
    assert result["training_prompt"]
    assert "api_key" not in json.dumps(result)
