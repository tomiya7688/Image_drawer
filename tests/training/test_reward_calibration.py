"""reward_calibration: 色の誤りに反応するよう重みを自動校正することを確認する。"""

import numpy as np
import pytest

from image_drawer.training.reward_calibration import (
    COLOR_RATIO,
    calibrate_spec,
    calibrate_weights,
    measure_sensitivity,
)
from image_drawer.training.reward_features import compute_features
from image_drawer.training.reward_spec import default_stage_reward_spec

SIDE = 32


def make_targets(count: int = 6) -> list[np.ndarray]:
    """白地に色付きの四角を置いた目標(色の入れ替えで色が変わる)。"""
    targets = []
    for index in range(count):
        image = np.ones((SIDE, SIDE, 3))
        top = 6 + index
        image[top : top + 12, 8:22] = (0.9, 0.2 + 0.05 * index, 0.1)
        targets.append(image)
    return targets


def test_measure_sensitivity_sees_edges_as_color_blind_and_rgb_as_color_aware():
    sensitivity = measure_sensitivity(make_targets(), SIDE, ["edge_32", "rgb_32", "rgb_hist_16"])
    assert sensitivity["edge_32"].color_drop < sensitivity["rgb_32"].color_drop
    assert sensitivity["rgb_hist_16"].color_drop > 0
    assert all(s.blank_drop > 0 for s in sensitivity.values())


def test_measure_sensitivity_rejects_empty_targets():
    with pytest.raises(ValueError):
        measure_sensitivity([], SIDE, ["rgb_32"])


def test_calibrate_weights_raises_color_drop_to_the_ratio_and_sums_to_one():
    names = ["rgb_128", "edge_128", "rgb_hist_16"]
    sensitivity = measure_sensitivity(make_targets(), SIDE, names)
    base = {"rgb_128": 0.6, "edge_128": 0.3, "rgb_hist_16": 0.1}

    def ratio(weights):
        color = sum(w * sensitivity[n].color_drop for n, w in weights.items())
        blank = sum(w * sensitivity[n].blank_drop for n, w in weights.items())
        return color / blank

    after, reached = calibrate_weights(base, sensitivity)
    assert sum(after.values()) == pytest.approx(1.0)
    assert ratio(after) >= ratio(base)
    assert reached == (ratio(after) >= COLOR_RATIO)


def test_calibrate_weights_keeps_edge_only_stages_unchanged():
    sensitivity = measure_sensitivity(make_targets(), SIDE, ["edge_32", "edge_64"])
    weights = {"edge_32": 0.3, "edge_64": 0.7}
    after, reached = calibrate_weights(weights, sensitivity)
    assert after == weights and reached is True


def test_calibrate_spec_returns_a_valid_spec_with_a_new_hash_and_a_report():
    default = default_stage_reward_spec()
    spec, report, document = calibrate_spec(default, make_targets(), SIDE)
    assert spec.profile_id.endswith("-calibrated") and spec.sha256 != default.sha256
    assert set(report) == {*default.stage_ids, "final_quality"}
    assert document["final_quality_weights"] == dict(spec.final_weights)
    # 辺だけの工程は変わらない
    assert dict(spec.potential_weights["sketch"]) == dict(default.potential_weights["sketch"])
    # 校正後のfinal_qualityは、色だけ違う画像でも既定より大きく下がる
    target = make_targets(1)[0]
    wrong = target[:, :, ::-1].copy()

    def quality(weights):
        scores, _ = compute_features(wrong, target, SIDE, list(weights))
        return sum(weights[n] * scores[n] for n in weights)

    assert quality(spec.final_weights) < quality(default.final_weights)
