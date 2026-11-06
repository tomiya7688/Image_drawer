"""受理されたstrokeの形(直線率・曲がり角)の集計テスト。"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from image_drawer.training.action_codec import ActionChoice  # noqa: E402
from image_drawer.training.stroke_stats import stroke_stats  # noqa: E402


def stroke(dirs: tuple[int, ...], segments: int | None = None) -> ActionChoice:
    return ActionChoice(
        kind_index=0,
        stroke_mode=0,
        stroke_start=0,
        stroke_segments=segments or len(dirs),
        stroke_dirs=dirs,
        stroke_lens=(2,) * len(dirs),
    )


def test_empty_and_non_stroke_choices_give_none_rates() -> None:
    stats = stroke_stats([])
    assert (stats.count, stats.single, stats.truncated) == (0, 0, 0)
    assert stats.single_rate is None and stats.mean_turn_deg is None
    assert stroke_stats([ActionChoice(kind_index=1, color_id=3)]).count == 0


def test_single_segment_strokes_have_no_turn() -> None:
    stats = stroke_stats([stroke((5,)), stroke((0,))])
    assert stats.count == 2 and stats.single == 2 and stats.single_rate == 1.0
    assert stats.mean_turn_deg is None


def test_turn_angle_uses_shorter_way_around_in_degrees() -> None:
    # 相対方向 0=直進0度, 1=22.5度, 15=22.5度(反対回り), 8=反転180度
    stats = stroke_stats([stroke((3, 0, 1, 15, 8))])
    assert stats.turns_deg == (0.0, 22.5, 22.5, 180.0)
    assert stats.mean_turn_deg == pytest.approx(56.25)
    assert stats.single_rate == 0.0


def test_truncated_count_sums_dropped_segments() -> None:
    stats = stroke_stats([stroke((2, 1), segments=4), stroke((6,), segments=1)])
    assert stats.truncated == 2 and stats.count == 2 and stats.single == 1
