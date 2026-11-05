"""TrainConfigの検証・JSON往復のテスト。"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from image_drawer.training import CurriculumConfig, RewardConfig, TrainConfig  # noqa: E402


def test_train_config_json_roundtrip_and_hash_is_stable() -> None:
    config = TrainConfig(master_seed=7, curriculum=CurriculumConfig(enabled=True))
    restored = TrainConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert restored.config_hash() == config.config_hash()
    assert TrainConfig(master_seed=8).config_hash() != config.config_hash()


def test_train_config_rejects_invalid_values() -> None:
    for kwargs in (
        {"canvas_side": 0},
        {"gamma": float("nan")},
        {"sigma_start": 0.01, "sigma_end": 0.05},
        {"learning_rate": 0},
        {"stroke_samples": 1},
    ):
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"unknown_key": 1})


def test_max_attempts_follows_controller_contract() -> None:
    assert TrainConfig(max_actions=96).max_attempts == 480
    assert TrainConfig(max_actions=10).max_attempts == 100


def test_sigma_decays_linearly() -> None:
    config = TrainConfig(iterations=11)
    assert config.sigma_at(0) == pytest.approx(0.15)
    assert config.sigma_at(10) == pytest.approx(0.05)
    assert config.sigma_at(5) == pytest.approx(0.10)


def test_reward_config_rejects_unknown_component_and_negative_weight() -> None:
    weights = dict(RewardConfig().training_weights)
    assert all(weights[n] == 0 for n in ("shape", "outline", "color", "perceptual", "text_match"))
    assert weights["progress"] == 1
    with pytest.raises(ValueError):
        RewardConfig(training_weights={**weights, "target_diff": 1.0})
    with pytest.raises(ValueError):
        RewardConfig(training_weights={**weights, "progress": -1.0})


def test_curriculum_stops_early_stages_and_ends_with_all_stages() -> None:
    curriculum = CurriculumConfig(enabled=True, stage_counts=(1, 7), iterations_per_stage=(2, 3))
    assert [curriculum.stop_after_stage(i) for i in range(6)] == [
        "rough", "rough", None, None, None, None,
    ]  # fmt: skip
    assert CurriculumConfig().stop_after_stage(0) is None
    with pytest.raises(ValueError):
        CurriculumConfig(stage_counts=(1, 3), iterations_per_stage=(1, 1))
