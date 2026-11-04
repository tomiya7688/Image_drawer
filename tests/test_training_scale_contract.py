"""学習データの暫定枚数、固定分割、再現試験、NSFW評価の契約。"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "developers"
    / "training_scale_test_v1.json"
)


def _plan() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_tiers_are_explicit_and_not_empirically_guaranteed():
    plan = _plan()
    assert plan["schema_version"] == "idraw-training-scale-test/v1"
    assert "暫定" in plan["status"]
    assert plan["external_training_input"] == ["image"]
    assert plan["initial_recommended_image_count"] == 1000
    stages = plan["stages"]
    assert [stage["id"] for stage in stages] == [
        "smoke",
        "pilot",
        "limited_illustration",
        "scaling",
        "research_large",
    ]
    assert [stage["target_images"] for stage in stages] == [
        100,
        1000,
        10000,
        50000,
        100000,
    ]
    assert stages[0]["min_images"] == 10
    assert stages[-1]["max_images"] is None
    assert all(stage["min_images"] <= stage["target_images"] for stage in stages)
    assert all(stage["graduation"] and stage["goal"] for stage in stages)


def test_split_counts_and_holdout_are_defined():
    plan = _plan()
    split = plan["split_policy"]
    assert split["ratio"] == {"train": 80, "validation": 10, "test": 10}
    assert split["split_before_extractor_selection"] is True
    assert split["holdout_is_never_used_for_tuning"] is True
    assert split["target_is_evaluator_only"] is True
    assert split["counts_are_targets_not_guarantees_when_groups_are_uneven"]

    for stage in plan["stages"]:
        counts = stage["reference_split"]
        assert set(counts) == {"train", "validation", "test"}
        assert sum(counts.values()) == stage["target_images"]
        assert all(count > 0 for count in counts.values())
        assert len(stage["seeds"]) >= (1 if stage["id"] == "smoke" else 3)


def test_learning_uses_real_rollouts_not_image_count_only():
    plan = _plan()
    required = set(plan["minimum_report_fields"])
    assert {
        "registered_image_count",
        "usable_image_count",
        "rollouts_per_image",
        "total_rollout_count",
        "policy_checkpoint_before",
        "policy_checkpoint_after",
        "reward_before",
        "reward_after",
        "unseen_image_quality",
        "unseen_prompt_following",
        "caption_failed_count",
        "caption_fallback_selected_model_counts",
        "confidence_interval_95",
    } <= required
    gate = plan["learning_acceptance"]
    assert all(gate.values())
    assert gate["no_quality_claim_from_ci_only"] is True


def test_nsfw_caption_benchmark_preserves_failure_denominators():
    benchmark = _plan()["caption_benchmark"]
    assert benchmark["min_images_per_stratum"] == 100
    assert benchmark["strata"] == ["sfw", "lawful_adult_nsfw"]
    assert "当該モデル" in benchmark["per_model_refusal_denominator"]
    assert "前処理" in benchmark["pipeline_completion_denominator"]
    assert benchmark["real_model_inference_required_for_quality_claim"] is True
    assert "評価不可" in benchmark["if_stratum_unavailable"]
    assert {
        "refusal_count",
        "fallback_selection_count",
        "all_failed_count",
        "caption_factual_error",
        "caption_coverage",
    } <= set(benchmark["required_metrics"])
