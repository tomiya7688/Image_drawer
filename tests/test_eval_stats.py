"""汎化評価の統計(seed、クラスタ・ブートストラップ、勝率、Holm、κ、ITT)を確認する。"""

import hashlib

import pytest

from image_drawer.generalization.stats import (
    aggregate_scores,
    cluster_bootstrap,
    effect_size,
    fleiss_kappa,
    holm_adjust,
    inference_seed,
    sign_test_p,
    win_rate,
)


def test_inference_seed_derivation_is_stable_and_paired_across_conditions():
    expected = int(hashlib.sha256(b"set1:img1:3").hexdigest()[:8], 16)
    assert inference_seed("set1", "img1", 3) == expected
    # 条件を引数に持たないので、beforeとafterで同じ値になる。k違い・画像違いでは変わる。
    assert inference_seed("set1", "img1", 3) == inference_seed("set1", "img1", 3)
    assert inference_seed("set1", "img1", 4) != expected
    assert inference_seed("set1", "img2", 3) != expected


def test_cluster_bootstrap_resamples_by_group_id_not_by_image():
    # グループAは100枚が全て1、グループBは1枚が0。画像単位ならほぼ1に張り付くが、
    # グループ単位ではBだけが2回選ばれる標本(平均0)と、Aだけの標本(平均1)が現れる。
    values = [1.0] * 100 + [0.0]
    groups = ["A"] * 100 + ["B"]
    _, lower, upper = cluster_bootstrap(values, groups, resamples=2000, rng_seed=1)
    assert (lower, upper) == (0.0, 1.0)


def test_bootstrap_is_deterministic_for_fixed_rng_seed():
    values = [0.1, 0.4, -0.2, 0.3, 0.0, 0.5]
    groups = ["a", "a", "b", "c", "c", "d"]
    first = cluster_bootstrap(values, groups, resamples=500, rng_seed=9)
    assert first == cluster_bootstrap(values, groups, resamples=500, rng_seed=9)
    assert first != cluster_bootstrap(values, groups, resamples=500, rng_seed=10)


def test_bootstrap_interval_contains_zero_for_identical_conditions():
    gains = [0.0] * 20
    mean_gain, lower, upper = cluster_bootstrap(gains, [f"g{i}" for i in range(20)], resamples=200)
    assert lower <= 0.0 <= upper and mean_gain == 0.0


def test_bootstrap_rejects_invalid_input():
    with pytest.raises(ValueError):
        cluster_bootstrap([], [])
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0], ["a", "b"])
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0], ["a"], ci=1.5)


def test_win_rate_counts_ties_as_half():
    assert win_rate([1.0, 0.0, -1.0, 2.0]) == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert effect_size([1.0, 3.0], [0.0, 2.0]) == pytest.approx(2.0)
    assert effect_size([1.0], [1.0, 1.0]) is None


def test_holm_adjustment_matches_reference_values():
    # 参照: p=[0.01, 0.04, 0.03] -> 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04(単調化して0.06)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert sign_test_p(9, 1) == pytest.approx(22 / 1024)
    assert sign_test_p(0, 0) == 1.0


def test_fleiss_kappa_matches_reference_values():
    # 参照: Fleiss(1971)の例として広く引用される表。10対象・14評価者・5選択肢でκ=0.210。
    table = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=0.001)
    assert fleiss_kappa([[3, 0], [0, 3]]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fleiss_kappa([[3, 0], [1, 1]])


def test_itt_assigns_blank_score_to_failed_generations():
    scores = [0.8, None, 0.6]
    blanks = [0.1, 0.2, 0.1]
    value, denominator = aggregate_scores(scores, blanks, reporting="itt")
    assert value == pytest.approx((0.8 + 0.2 + 0.6) / 3) and denominator == 3


def test_conditional_and_itt_aggregates_differ_when_failures_exist():
    scores = [0.8, None, 0.6]
    blanks = [0.1, 0.2, 0.1]
    conditional, denominator = aggregate_scores(scores, blanks, reporting="conditional")
    assert conditional == pytest.approx(0.7) and denominator == 2
    assert conditional != aggregate_scores(scores, blanks, reporting="itt")[0]
    assert aggregate_scores([None], [0.1], reporting="conditional") == (None, 0)
