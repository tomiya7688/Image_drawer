"""S1b-3: Budget（Action数・試行数の上限と既定値）。"""

from __future__ import annotations

import pytest

from image_drawer.drawing.stage import StageConfigError
from image_drawer.drawing.stage.budget import MAX_ACTIONS_LIMIT, Budget


def test_budget_default_attempt_limit_floor_is_100():
    assert Budget(10).max_attempts == 100
    assert Budget(1).max_attempts == 100


def test_budget_default_attempt_limit_is_five_times_actions():
    assert Budget(1000).max_attempts == 5000
    assert Budget(20).max_attempts == 100
    assert Budget(21).max_attempts == 105


def test_budget_explicit_attempt_limit_is_kept():
    assert Budget(10, 10).max_attempts == 10
    assert Budget(10, 777).max_attempts == 777


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, 10.5, None, "10"])
def test_budget_rejects_zero_negative_bool_float_and_none_actions(bad):
    with pytest.raises(StageConfigError):
        Budget(bad)


def test_budget_accepts_boundary_max_actions_19900():
    assert Budget(MAX_ACTIONS_LIMIT).max_actions == 19_900
    assert Budget(1).max_actions == 1


def test_budget_rejects_max_actions_above_engine_limit():
    with pytest.raises(StageConfigError):
        Budget(MAX_ACTIONS_LIMIT + 1)


@pytest.mark.parametrize("attempts", [9, 0, -5, True, 12.0, "20"])
def test_budget_rejects_attempt_limit_below_action_limit(attempts):
    with pytest.raises(StageConfigError):
        Budget(10, attempts)
