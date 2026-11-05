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
        {"softmax_temperature": 0.0},
        {"softmax_temperature": float("nan")},
        {"stroke_grid": 0},
        {"stroke_max_segments": 0},
        {"stroke_lengths": ()},
        {"stroke_lengths": (1, 1)},
        {"stroke_max_segments": 6, "stroke_samples": 6},
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


def test_stroke_settings_roundtrip_and_have_no_gaussian_sigma() -> None:
    config = TrainConfig(stroke_grid=16, stroke_lengths=(1, 4), softmax_temperature=0.7)
    data = json.loads(json.dumps(config.to_dict()))
    assert data["stroke_lengths"] == [1, 4]
    assert TrainConfig.from_dict(data) == config
    assert not {"sigma_start", "sigma_end"} & set(data)
    assert config.stroke_shape.grid == 16 and config.stroke_shape.lengths == (1, 4)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"sigma_start": 0.15})


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
