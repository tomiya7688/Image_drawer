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
    assert routing["priority"] == "user_registered_ordered_chains"
    assert routing["caption_chain"]["minimum"] == 1
    assert routing["caption_chain"]["maximum"] == 256
    assert routing["caption_chain"]["requires_qwen"] is False
    assert routing["caption_chain"]["requires_exactly_two"] is False
    assert routing["tag_chain"]["requires_caption_model_for_tags"] is False
    assert routing["qwen_is_optional_example"] is True
    assert routing["default_img2txt_fallback"]["model_id"] == (
        "fancyfeast/llama-joycaption-beta-one-hf-llava"
    )
    assert routing["default_img2txt_fallback"]["adapter"] == "joycaption"
    assert "joycaption" in c["user_registration"]["adapters"]
    for kind in ("refused", "empty", "invalid_format", "unverified_facts"):
        assert kind in routing["primary_failure_triggers"]
    assert (
        "never_require_manual_text" in routing["all_failed"]
        or "without_creating" in routing["all_failed"]
    )


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
    assert d["prompt_extraction"]["pipeline"] == "image_drawer.prompting.ImageTrainingPreprocessor"
    assert d["prompt_extraction"]["caption_and_tag_run_independently"] is True
    assert d["prompt_extraction"]["tag_pipeline"] == "image_drawer.prompting.TaggingPipeline"


def test_related_research_and_support_tools_are_documented():
    c = _config()
    assert "meta" in c["tag_extraction"]["categories"]
    assert "meta" in c["tag_extraction"]["no_implicit_prompt_merge_categories"]
    assert "tomiya7688/comfyUI_support_tools" in c["related_repository"]
    assert len(c["research"]) >= 3
    d = (ROOT / "docs" / "developers" / "prompt_extraction_v1.md").read_text(encoding="utf-8")
    assert "Qwen" in d
    assert "短文" in d
    assert "Completeness" in d
    assert "Correctness" in d
