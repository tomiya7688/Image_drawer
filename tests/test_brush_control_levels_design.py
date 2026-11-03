"""サイズ・不透明度・筆圧の固定段階表と筆圧パターンを独立に検査する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "docs" / "developers" / "brush_control_levels_v2.json"
CATALOG = ROOT / "docs" / "developers" / "brush_presets_v2.json"


def load_levels() -> dict:
    return json.loads(CONTROLS.read_text(encoding="utf-8"))


def nearest_level(values: list[int], raw: float) -> int:
    """一番近い候補。ちょうど中間なら値の小さい候補を選ぶ。"""
    return min(range(len(values)), key=lambda i: (abs(values[i] - raw), values[i]))


def round_half_up(numerator: int, denominator: int) -> int:
    """非負有理数の厳密な四捨五入。"""
    return (2 * numerator + denominator) // (2 * denominator)


def pressure_pattern(level_ids: list[int | None], config: dict) -> str | None:
    """教師用。サンプル番号で進行度を定義し、圧力の全体形を分類する。"""
    if len(level_ids) < config["pressure_pattern_min_valid_samples"]:
        return None
    if any(value is None for value in level_ids):
        return None
    values = [config["pressure_percent"][value] for value in level_ids]
    peak = max(values)
    if peak == 0:
        return "constant"
    # 5つの位置 0%,25%,50%,75%,100% を有理数で補間する。
    observations = []
    for j in range(5):
        t = j * (len(values) - 1)
        index, rem = divmod(t, 4)
        left = values[index]
        right = values[min(index + 1, len(values) - 1)]
        interpolated_numerator = (4 - rem) * left + rem * right
        observations.append(round_half_up(interpolated_numerator * 100, 4 * peak))
    profiles = config["pressure_patterns"][:-1]
    errors = [
        sum(abs(a - b) for a, b in zip(observations, p["normalized_pressure_percent"]))
        for p in profiles
    ]
    best = min(range(len(errors)), key=lambda i: (errors[i], i))
    if errors[best] > 5 * config["pressure_pattern_other_threshold_mae_percent"]:
        return "other"
    return profiles[best]["id"]


def test_levels_are_stable_and_ordered():
    c = load_levels()
    assert c["schema_version"] == "idraw-brush-control-levels/2"
    assert c["control_set_id"] == "brush-controls-v2-levels-20261103"
    assert c["size_px"] == [
        1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128, 200, 256
    ]
    assert c["opacity_percent"] == [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert c["pressure_percent"] == list(range(0, 101, 10))
    assert c["tilt_cdeg"] == list(range(-9000, 9001, 1500))
    for key in ("size_px", "opacity_percent", "pressure_percent", "tilt_cdeg"):
        values = c[key]
        assert all(type(v) is int for v in values)
        assert values == sorted(set(values))
    assert c["quantize_tie_break"] == "choose_lower_level_value"


@pytest.mark.parametrize(
    ("key", "input_value", "expected_value"),
    [
        ("size_px", 14, 12),
        ("size_px", 3.2, 3),
        ("size_px", 50, 50),
        ("size_px", 101, 100),
        ("opacity_percent", 82, 80),
        ("opacity_percent", 15, 10),
        ("pressure_percent", 73, 70),
        ("pressure_percent", 0, 0),
        ("tilt_cdeg", 750, 0),
    ],
)
def test_nearest_level_and_tie_break(key: str, input_value: float, expected_value: int):
    values = load_levels()[key]
    assert values[nearest_level(values, input_value)] == expected_value


def test_presets_reference_valid_level_ids():
    c = load_levels()
    preset = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert preset["control_set_id"] == c["control_set_id"]
    for b in preset["presets"]:
        assert c["size_px"][b["default_size_level_id"]] > 0
        assert 5 <= c["opacity_percent"][b["default_opacity_level_id"]] <= 100


def test_pressure_profile_labels_are_deterministic():
    c = load_levels()
    assert [p["id"] for p in c["pressure_patterns"]] == [
        "constant", "increase", "decrease", "taper_both", "other"
    ]
    assert pressure_pattern([8, 8, 8, 8, 8], c) == "constant"
    assert pressure_pattern([2, 4, 7, 9, 10], c) == "increase"
    assert pressure_pattern([10, 9, 7, 4, 2], c) == "decrease"
    assert pressure_pattern([2, 7, 10, 7, 2], c) == "taper_both"
    assert pressure_pattern([1, 10, 1, 10, 1], c) == "other"
    assert pressure_pattern([2, None, 8, 10], c) is None
    assert pressure_pattern([2, 8], c) is None
    assert pressure_pattern([0, 0, 0], c) == "constant"
