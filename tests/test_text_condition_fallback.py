"""ユーザー定義テキスト条件化とGPU→CPU自動メモリ解放をテストする。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from image_drawer.prompting.text_encoder_registry import TextEncoderRegistry
from image_drawer.prompting.text_condition import (
    HashedNgramTextEncoder,
    TextConditionEncoder,
    TextEncodingError,
    load_text_encoder_from_settings,
)

NORMAL_SETTINGS = "docs/users/preprocessor_settings.example.json"


@dataclass
class FakeEncoder:
    encoder_id: str
    behavior: str
    events: list[str]
    model_id: str = "same-model"
    model_revision: str = "commit-sha"
    dimension: int = 3
    feature_space_id: str = "sentence-transformer:same-model@commit-sha"

    def encode(self, text: str):
        self.events.append(self.encoder_id + ":encode")
        if self.behavior == "oom":
            raise RuntimeError("CUDA out of memory")
        if self.behavior == "bad":
            return [float("nan"), 0, 0]
        return [3.0, 4.0, 0.0]

    def release(self) -> None:
        self.events.append(self.encoder_id + ":release")


def test_text_encoder_offloads_gpu_before_cpu_fallback():
    events: list[str] = []
    gpu = FakeEncoder("gpu", "oom", events)
    cpu = FakeEncoder("cpu", "ok", events)
    encoder = TextConditionEncoder([gpu, cpu])
    result = encoder.encode("赤いリンゴを描く")
    assert result.values == pytest.approx((0.6, 0.8, 0.0))
    assert result.encoder_id == "cpu"
    assert [attempt.status for attempt in result.attempts] == ["oom", "ok"]
    assert events == ["gpu:encode", "gpu:release", "cpu:encode", "cpu:release"]


def test_incompatible_embedding_spaces_cannot_fallback_or_load_checkpoint():
    events: list[str] = []
    gpu = FakeEncoder("gpu", "oom", events)
    other = FakeEncoder("other", "ok", events, model_id="different-model")
    with pytest.raises(ValueError, match="互換性"):
        TextConditionEncoder([gpu, other])
    encoder = TextConditionEncoder([gpu])
    with pytest.raises(TextEncodingError) as failure:
        encoder.validate_checkpoint_space("other-model", 3)
    assert failure.value.kind == "incompatible_checkpoint"
    with pytest.raises(TextEncodingError):
        encoder.validate_checkpoint_space(gpu.feature_space_id, 512)


def test_all_encoders_fail_or_emit_invalid_values():
    events: list[str] = []
    encoder = TextConditionEncoder(
        [FakeEncoder("gpu", "oom", events), FakeEncoder("cpu", "bad", events)]
    )
    with pytest.raises(TextEncodingError) as failure:
        encoder.encode("A red apple")
    assert failure.value.kind == "all_encoders_failed"
    assert "gpu:oom" in str(failure.value)
    assert "cpu:invalid_output" in str(failure.value)
    assert events == ["gpu:encode", "gpu:release", "cpu:encode", "cpu:release"]


def test_blank_prompt_is_invalid_before_running_any_encoder():
    events: list[str] = []
    encoder = TextConditionEncoder([FakeEncoder("cpu", "ok", events)])
    with pytest.raises(ValueError, match="空"):
        encoder.encode("  ")
    assert events == []


def test_local8g_profile_selects_same_model_gpu_then_cpu():
    chain = load_text_encoder_from_settings(NORMAL_SETTINGS)
    assert len(chain.providers) == 2
    first, second = chain.providers
    assert first.device == "cuda"
    assert second.device == "cpu"
    assert first.model_id == second.model_id
    assert first.model_revision == second.model_revision
    assert first.dimension == second.dimension == 384
    assert first.feature_space_id == second.feature_space_id


def test_hashed_ngram_is_deterministic_but_not_compatible_with_sentence_model():
    baseline = HashedNgramTextEncoder(dimension=64)
    model = TextConditionEncoder([baseline])
    first = model.encode("赤いリンゴを描く")
    second = model.encode("赤いリンゴを描く")
    assert first.values == second.values
    assert len(first.values) == 64
    assert first.model_id == "hashed-ngram-v1"
    assert abs(sum(v * v for v in first.values) - 1) < 1e-7


def test_developer_can_register_custom_text_encoder_without_arbitrary_json_import():
    events: list[str] = []
    registry = TextEncoderRegistry()
    registry.register_adapter(
        "custom-test-provider",
        lambda entry: FakeEncoder(entry["id"], "ok", events),
    )
    encoder = registry.build(
        {
            "text_encoder": {
                "providers": [
                    {"id": "custom", "adapter": "custom-test-provider"}
                ]
            }
        }
    )
    assert encoder.encode("赤いリンゴ").encoder_id == "custom"
    with pytest.raises(ValueError, match="未登録"):
        TextEncoderRegistry().build(
            {
                "text_encoder": {
                    "providers": [
                        {"id": "bad", "adapter": "python:unsafe.module"}
                    ]
                }
            }
        )
