"""Reward特徴(設計書8.2節 test_reward_features_golden_values ほか)のテスト。"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from image_drawer.training import reward_features as rf  # noqa: E402


def _square_image(side: int, low: int, high: int, color: tuple[float, float, float]):
    image = np.ones((side, side, 3), dtype=np.float64)
    image[low:high, low:high] = color
    return image


def test_box_downscale_averages_cells_and_handles_non_divisible_sizes() -> None:
    gradient = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert rf.box_downscale(gradient, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    assert np.array_equal(rf.box_downscale(gradient, 4), gradient)
    # 3->2: 出力セルは入力区間[0,1.5)と[1.5,3)を覆い、重なり幅で平均する。
    line = np.array([[0.0, 3.0, 6.0]] * 3)
    assert rf.box_downscale(line, 2)[0].tolist() == pytest.approx([1.0, 5.0])
    with pytest.raises(ValueError):
        rf.box_downscale(gradient, 5)


def test_sobel_magnitude_saturates_on_step_edge_and_is_zero_on_flat() -> None:
    step = np.zeros((4, 4))
    step[:, 2:] = 1.0
    magnitude = rf.sobel_magnitude(step)
    assert magnitude[:, 1:3].tolist() == [[1.0, 1.0]] * 4
    assert magnitude[:, 0].tolist() == [0.0] * 4 and magnitude[:, 3].tolist() == [0.0] * 4
    assert not rf.sobel_magnitude(np.full((4, 4), 0.7)).any()


def test_soft_dice_edge_cases() -> None:
    edges = np.zeros((4, 4))
    edges[1, :] = 1.0
    assert rf.soft_dice(edges, edges) == pytest.approx(1.0)
    assert rf.soft_dice(np.zeros((4, 4)), np.zeros((4, 4))) == pytest.approx(1.0)
    assert rf.soft_dice(np.zeros((4, 4)), edges) < 1e-5
    other = np.zeros((4, 4))
    other[2, :] = 1.0
    assert rf.soft_dice(edges, other) < 1e-5


def test_similarity_bounds_for_extreme_images() -> None:
    white = np.ones((8, 8, 3))
    black = np.zeros((8, 8, 3))
    gray = np.full((8, 8, 3), 0.5)
    assert rf.rgb_similarity(white, white, 8) == 1.0
    assert rf.rgb_similarity(white, black, 8) == 0.0
    assert rf.rgb_similarity(white, gray, 8) == pytest.approx(0.5)
    assert rf.luma_similarity(white, black, 8) == pytest.approx(0.0)
    assert rf.rgb_hist_similarity(white, white) == 1.0
    assert rf.rgb_hist_similarity(white, black) == 0.0
    # 白(1.0)は最上位binに入り、境界値でindex 16にならない。
    assert rf.rgb_hist_similarity(white, np.full((8, 8, 3), 15.5 / 16)) == 1.0


def test_reward_features_golden_values() -> None:
    # 固定入力: 8x8の白紙と、左上4x4が赤(1,0,0)の画像を比べる暫定数値化のGolden。
    blank = np.ones((8, 8, 3))
    red = _square_image(8, 0, 4, (1.0, 0.0, 0.0))
    names = ["edge_32", "edge_64", "edge_128", "rgb_32", "rgb_128", "luma_64", "rgb_hist_16"]
    scores, resolutions = rf.compute_features(blank, red, 8, names)
    assert resolutions == {name: 8 for name in names}
    assert scores["rgb_32"] == scores["rgb_128"] == pytest.approx(1 - (2 / 3) * (1 / 4))
    assert scores["luma_64"] == pytest.approx(1 - 0.7874 * 0.25)
    assert scores["rgb_hist_16"] == pytest.approx((1.0 + 0.75 + 0.75) / 3)
    # 白紙は輪郭が無く、目標には輪郭があるので輪郭類似度はほぼ0。
    assert scores["edge_32"] == scores["edge_64"] == scores["edge_128"]
    assert scores["edge_32"] < 1e-5
    same, _ = rf.compute_features(red, red, 8, names)
    assert all(value == pytest.approx(1.0) for value in same.values())


def test_reward_resolution_is_capped_to_canvas_side() -> None:
    blank = np.ones((32, 32, 3))
    scores, resolutions = rf.compute_features(blank, blank, 32, ["edge_128", "edge_64", "rgb_128"])
    assert resolutions == {"edge_128": 32, "edge_64": 32, "rgb_128": 32}
    assert all(value == pytest.approx(1.0) for value in scores.values())
    _, large = rf.compute_features(
        np.ones((256, 256, 3)), np.ones((256, 256, 3)), 256, ["edge_32", "edge_128"]
    )
    assert large == {"edge_32": 32, "edge_128": 128}
    assert rf.effective_resolution(128, 32) == 32


def test_compute_features_rejects_unknown_feature_and_bad_shape() -> None:
    image = np.ones((8, 8, 3))
    with pytest.raises(ValueError):
        rf.compute_features(image, image, 8, ["edge_16"])
    with pytest.raises(ValueError):
        rf.compute_features(image, image[:4], 8, ["edge_32"])


def test_composite_over_white_blends_alpha() -> None:
    rgba = bytes([255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 128, 10, 20, 30, 0])
    image = rf.composite_over_white(rgba, 2)
    assert image[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert image[0, 1].tolist() == [1.0, 1.0, 1.0]
    assert image[1, 0].tolist() == pytest.approx([1 - 128 / 255] * 3)
    with pytest.raises(ValueError):
        rf.composite_over_white(rgba[:-1], 2)


def test_feature_names_match_stage_reward_json() -> None:
    import json
    from pathlib import Path

    spec = Path(__file__).resolve().parents[2] / "docs" / "developers" / "stage_reward_v2.json"
    assert set(json.loads(spec.read_text(encoding="utf-8"))["features"]) == rf.FEATURE_NAMES
