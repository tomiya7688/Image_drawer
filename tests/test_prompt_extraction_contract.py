"""高精度caption設定と画像のみ学習の整合を監査する設計契約。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs" / "developers" / "prompt_extraction_v1.json"
LEARNING = ROOT / "docs" / "developers" / "training_input_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_priority_fallback_and_failure_are_declared():
    c = _config()
    routing = c["model_routing"]
    assert c["schema_version"] == "idraw-prompt-extraction/v1"
    assert c["external_training_input"] == ["image"]
    assert routing["priority"] == ["quality_primary", "qwen_local_fallback"]
    assert routing["qwen_local_fallback"]["local_model_required"] is True
    assert routing["qwen_local_fallback"]["cannot_guarantee_all_content_is_supported"] is True
    for kind in ("refused", "empty", "invalid_format", "unverified_facts"):
        assert kind in routing["primary_failure_triggers"]
    assert "never_write_generic_caption" in routing["both_failed"]


def test_human_prompt_alignment_and_no_hallucination():
    c = _config()
    generation = c["generation"]
    assert generation["prompt_variants"] == ["short", "standard", "detailed"]
    assert generation["sampling_weights"] == [40, 40, 20]
    assert generation["optional_independent_verifier"] is True
    assert "no_unobserved_details" in generation["quality_rules"]
    assert "no_hallucinated_negative_prompt" in generation["quality_rules"]
    assert generation["human_prompt_alignment"]["inference_uses_same_text_encoder"] is True
    assert {"subject", "count", "color", "left_right", "short_prompt_following"} <= set(
        generation["human_prompt_alignment"]["evaluations"]
    )


def test_pipeline_is_internal_and_no_user_caption_file_required():
    c = _config()
    d = json.loads(LEARNING.read_text(encoding="utf-8"))
    assert d["external_required_inputs"] == ["image"]
    assert d["external_auxiliary_files_required"] is False
    assert d["prompt_extraction"]["profile_id"] == c["profile_id"]
    assert d["prompt_extraction"]["no_user_extra_files"] is True
    assert d["prompt_extraction"]["pipeline"] == "image_drawer.prompting.ImagePromptPipeline"


def test_related_research_and_support_tools_are_documented():
    c = _config()
    assert "tomiya7688/comfyUI_support_tools" in c["related_repository"]
    assert len(c["research"]) >= 3
    d = (ROOT / "docs" / "developers" / "prompt_extraction_v1.md").read_text(
        encoding="utf-8"
    )
    assert "Qwen" in d
    assert "短文" in d
    assert "Completeness" in d
    assert "Correctness" in d
