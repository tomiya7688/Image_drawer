"""RewardEvaluator・工程別Reward仕様のテスト(設計書8.2節)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from training_helpers import complete_script, run_scripted, small_config, stroke  # noqa: E402

from image_drawer.dataset import RewardTarget  # noqa: E402
from image_drawer.drawing import DrawingAction  # noqa: E402
from image_drawer.training import REWARD_COMPONENTS, RewardConfig  # noqa: E402
from image_drawer.training.reward import RewardEvaluator, TargetFeatures  # noqa: E402
from image_drawer.training.reward_spec import (  # noqa: E402
    EMBEDDED_SPEC,
    default_stage_reward_spec,
    load_stage_reward_spec,
    parse_stage_reward_spec,
)

SPEC_JSON = Path(__file__).resolve().parents[2] / "docs" / "developers" / "stage_reward_v2.json"
SIDE = 16


def band_target(horizontal: bool = True) -> TargetFeatures:
    rgb = np.ones((SIDE, SIDE, 3))
    if horizontal:
        rgb[6:10, 1:15] = 0.0
    else:
        rgb[1:15, 6:10] = 0.0
    return TargetFeatures("t", SIDE, rgb)


def evaluator(**config) -> RewardEvaluator:
    return RewardEvaluator(RewardConfig(**config), SIDE)


PREPARE = [
    DrawingAction("select_layer", {"layer_id": "sys_rough_001"}),
    DrawingAction("select_color", {"color_id": 0}),
    DrawingAction("select_brush", {"brush_id": "ink_gpen"}),
    DrawingAction("set_brush_params", {"size_level_id": 6, "opacity_level_id": 10}),
]


def test_stage_weights_sum_to_one_and_unknown_feature_rejected() -> None:
    spec = load_stage_reward_spec(SPEC_JSON)
    assert spec == default_stage_reward_spec()
    document = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    for key, value in EMBEDDED_SPEC.items():
        assert document[key] == value, key
    for weights in spec.potential_weights.values():
        assert sum(weights.values()) == pytest.approx(1.0)
    bad = json.loads(json.dumps(EMBEDDED_SPEC))
    bad["stage_potential_weights"]["rough"] = {"edge_16": 1}
    with pytest.raises(ValueError):
        parse_stage_reward_spec(bad)
    bad["stage_potential_weights"]["rough"] = {"edge_32": 0.5}
    with pytest.raises(ValueError):
        parse_stage_reward_spec(bad)


def test_reward_blank_vs_blank_target_is_not_rewarded_for_blank_when_target_has_edges() -> None:
    target = band_target()
    ev = evaluator()
    blank = np.ones((SIDE, SIDE, 3))
    blank_level = ev._level("rough", blank, target)
    perfect_level = ev._level("rough", target.rgb, target)
    assert blank_level.total < 0.01 and perfect_level.total > 0.99
    assert ev._level("rough", blank, TargetFeatures("b", SIDE, blank)).total == pytest.approx(1.0)


def test_reward_progress_improves_when_stroke_matches_target() -> None:
    target = band_target(horizontal=True)
    ev = evaluator()

    def progress_of(line) -> float:
        trial = run_scripted([*PREPARE, stroke(*line)], stop_after_stage=None,
                             config=small_config(max_actions=5))  # fmt: skip
        reward = ev.evaluate_trial(trial, target, trial.training_prompt)
        return sum(b.components["progress"] for b in reward.steps)

    matching = progress_of((0.1, 0.5, 0.9, 0.5))
    mismatching = progress_of((0.5, 0.1, 0.5, 0.9))
    assert matching > mismatching
    assert matching > 0.1


def test_reward_stage_boundary_does_not_mix_potentials() -> None:
    target = band_target()
    trial = run_scripted(complete_script())
    reward = evaluator().evaluate_trial(trial, target, "x")
    boundaries = [b for b in reward.steps if b.kind == "boundary"]
    assert len(boundaries) == 5
    cost = 0.05 / 60
    for breakdown in boundaries:
        assert breakdown.components["progress"] == 0.0
        assert breakdown.total == pytest.approx(-cost)
    # 次の工程の最初の描画のpotential_beforeは、境界で確定した新しい基準点と一致する。
    steps = list(reward.steps)
    for index, breakdown in enumerate(steps[:-1]):
        if breakdown.kind == "boundary":
            following = next(b for b in steps[index + 1 :] if b.kind == "step")
            assert following.potential_before == pytest.approx(breakdown.potential_after)


def test_reward_invalid_attempt_penalty_and_no_pixel_change() -> None:
    trial = run_scripted([DrawingAction("advance_stage", {}), *complete_script()])
    reward = evaluator().evaluate_trial(trial, band_target(), "x")
    first = reward.steps[0]
    assert first.kind == "invalid" and first.potential_before is None
    assert first.total == pytest.approx(-0.05 / 60)
    assert {k: v for k, v in first.components.items() if v} == {"action_cost": -0.05 / 60}


def test_reward_failure_penalty_on_budget_exceeded() -> None:
    trial = run_scripted(
        [DrawingAction("select_color", {"color_id": 1})], small_config(max_actions=5)
    )
    reward = evaluator().evaluate_trial(trial, band_target(), "x")
    last = reward.steps[-1]
    assert trial.end_reason == "action_budget" and last.kind == "failure"
    assert last.components["failure_penalty"] == -1.0
    assert last.total == pytest.approx(-1.0 - 0.05 / 5)
    assert reward.final_quality is None
    finished = evaluator().evaluate_trial(run_scripted(complete_script()), band_target(), "x")
    assert all(b.kind != "failure" for b in finished.steps)


def test_reward_final_bonus_uses_final_quality() -> None:
    trial = run_scripted(complete_script())
    reward = evaluator().evaluate_trial(trial, band_target(), "x")
    assert reward.final_quality is not None and 0.0 <= reward.final_quality <= 1.0
    last = reward.steps[-1]
    assert last.kind == "final"
    assert last.components["final_bonus"] == pytest.approx(2.0 * reward.final_quality)
    assert reward.total_return == pytest.approx(sum(b.total for b in reward.steps))


def test_reward_breakdown_contains_all_components() -> None:
    reward = evaluator().evaluate_trial(run_scripted(complete_script()), band_target(), "x")
    for breakdown in reward.steps:
        assert set(breakdown.components) == set(REWARD_COMPONENTS)
        assert breakdown.reward_impl_version == "stage-reward-impl-v0"
        assert breakdown.reward_profile_id == "image-only-stage-reward-baseline-v2"
    painted = [b for b in reward.steps if b.kind == "step" and b.components["progress"]]
    for breakdown in painted:
        parts = sum(breakdown.components[n] for n in ("outline", "color", "perceptual"))
        assert parts == pytest.approx(breakdown.components["progress"])


def test_reward_resolution_is_capped_to_canvas_side() -> None:
    reward = evaluator().evaluate_trial(run_scripted(complete_script()), band_target(), "x")
    resolutions = {k: v for b in reward.steps for k, v in b.effective_resolution.items()}
    assert resolutions["edge_128"] == SIDE and resolutions["edge_64"] == SIDE


def test_prepare_target_reads_image_once_and_resizes(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "t.png"
    Image.new("RGB", (32, 24), (255, 0, 0)).save(path)
    features = evaluator().prepare_target(RewardTarget("i", str(path), "sha"))
    assert features.rgb.shape == (SIDE, SIDE, 3)
    assert features.rgb[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        evaluator().prepare_target(RewardTarget("i", str(tmp_path / "none.png"), "sha"))
