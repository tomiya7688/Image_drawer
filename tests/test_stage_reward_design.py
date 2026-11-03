"""工程別Rewardの静的設計契約。画像比較の画素Goldenではない。"""

from __future__ import annotations

import json
from pathlib import Path

EVALUATION = (
    Path(__file__).resolve().parents[1] / "docs" / "developers" / "stage_reward_v2.json"
)
STAGES = (
    Path(__file__).resolve().parents[1] / "docs" / "developers" / "drawing_stage_workflow_v2.json"
)


def test_reward_is_stage_specific_and_normalized():
    reward = json.loads(EVALUATION.read_text(encoding="utf-8"))
    stages = json.loads(STAGES.read_text(encoding="utf-8"))
    stage_ids = [stage["id"] for stage in stages["stages"]]
    assert reward["schema_version"] == "idraw-stage-reward/v2"
    assert reward["stage_ids"] == stage_ids
    assert set(reward["stage_potential_weights"]) == set(stage_ids)
    assert reward["feature_score_range"] == [0, 1]
    features = set(reward["features"])
    for stage_weights in reward["stage_potential_weights"].values():
        assert set(stage_weights) <= features
        assert all(0 <= value <= 1 for value in stage_weights.values())
        assert abs(sum(stage_weights.values()) - 1.0) < 1e-9
    final_weights = reward["final_quality_weights"]
    assert set(final_weights) <= features
    assert abs(sum(final_weights.values()) - 1.0) < 1e-9


def test_target_is_not_exposed_to_policy_during_trial():
    reward = json.loads(EVALUATION.read_text(encoding="utf-8"))
    policy = set(reward["policy_observation_excludes"])
    assert reward["prohibit_reward_as_policy_input_same_trial"] is True
    assert reward["reward_update_timing"] == "after_rollout_only"
    assert {"target_image", "target_embedding", "target_dependent_reward"} <= policy
    assert reward["target_image_access"] == [
        "evaluator",
        "trainer_reward_calculation",
    ]


def test_action_cost_and_failure_reward_are_explicit():
    reward = json.loads(EVALUATION.read_text(encoding="utf-8"))
    assert reward["step_reward_formula"] == (
        "stage_potential_after - stage_potential_before - 0.05/max_actions"
    )
    assert reward["final_bonus_formula"] == "2.0*final_quality"
    assert reward["invalid_action_penalty_formula"] == "-0.05/max_actions"
    assert reward["episode_failure_penalty"] < 0
    assert reward["weights_are_initial_hypothesis_not_guarantee"] is True
    assert reward["feature_numeric_renderer"].startswith("pending-")

def test_temporary_ruff_difference():
    import subprocess

    result = subprocess.run(
        ["ruff", "format", "--diff", __file__],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
