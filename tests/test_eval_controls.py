"""対照指標(prompt_shuffle_delta)の配線のテスト。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from image_drawer.generalization.cli_report import _controls_of
from image_drawer.generalization.controls import (
    PROBES,
    probe_hit_rate_control,
    prompt_shuffle_control,
    shuffle_targets,
)
from image_drawer.generalization.gates import GateInputs
from image_drawer.generalization.report import _unverified
from image_drawer.generalization.runner import GenerationResult, ProbeResult, RunOutput

IDS = [f"img{i}" for i in range(6)]


def generation(
    image_id: str, k: int, score: float | None, failure: str | None = None
) -> GenerationResult:
    metrics = {} if score is None else {"rgb_128": score}
    return GenerationResult("after", image_id, f"g-{image_id}", k, 0, failure, metrics)


def output_with(own: dict[str, float | None], other: dict[str, float | None]) -> RunOutput:
    own_rows = [generation(i, k, s, None if s is not None else "policy_error")
                for i, s in own.items() for k in range(2)]  # fmt: skip
    other_rows = [generation(i, k, s, None if s is not None else "policy_error")
                  for i, s in other.items() for k in range(2)]  # fmt: skip
    blank = {i: {"rgb_128": 0.1} for i in own}
    return RunOutput("set", "test", own_rows, blank, {}, {}, {}, {}, shuffled=other_rows)


def test_shuffle_targets_never_map_an_item_to_itself_and_are_deterministic() -> None:
    targets = shuffle_targets(IDS, "set1")
    assert set(targets) == set(IDS) and set(targets.values()) == set(IDS)
    assert all(src != dst for src, dst in targets.items())
    assert targets == shuffle_targets(list(reversed(IDS)), "set1")
    with pytest.raises(ValueError, match="at least two"):
        shuffle_targets(["only"], "set1")


def test_control_is_positive_when_correct_prompt_scores_higher() -> None:
    own = {i: 0.9 for i in IDS}
    other = {i: 0.3 for i in IDS}
    control = prompt_shuffle_control(output_with(own, other), "after", "rgb_128", resamples=200)
    assert control["delta"] == pytest.approx(0.6)
    assert control["ci95"][0] > 0 and control["n"] == 12 and control["metric"] == "rgb_128"


def test_control_is_zero_for_a_prompt_ignoring_policy() -> None:
    same = {i: 0.5 for i in IDS}
    control = prompt_shuffle_control(
        output_with(same, dict(same)), "after", "rgb_128", resamples=200
    )
    assert control["delta"] == 0.0 and control["ci95"] == [0.0, 0.0]


def test_failed_generations_use_the_blank_score_and_stay_in_the_denominator() -> None:
    own = {i: 0.9 for i in IDS}
    other = {i: (None if i == "img0" else 0.9) for i in IDS}  # img0の入れ替え生成だけ失敗
    control = prompt_shuffle_control(output_with(own, other), "after", "rgb_128", resamples=200)
    assert control["n"] == 12
    assert control["delta"] == pytest.approx((0.9 - 0.1) * 2 / 12)  # 失敗は白紙0.1の得点


def test_control_is_none_without_shuffled_generations_or_for_other_conditions() -> None:
    output = output_with({i: 0.9 for i in IDS}, {i: 0.3 for i in IDS})
    assert prompt_shuffle_control(output, "before", "rgb_128", resamples=50) is None
    output.shuffled = []
    assert prompt_shuffle_control(output, "after", "rgb_128", resamples=50) is None


def test_report_controls_use_the_after_condition_and_list_only_the_missing_ones() -> None:
    output = output_with({i: 0.9 for i in IDS}, {i: 0.3 for i in IDS})
    controls = _controls_of(output, {"bootstrap": {"resamples": 50, "rng_seed": 3}})
    assert set(controls) == {"prompt_shuffle_delta"} and controls["prompt_shuffle_delta"]["n"] == 12
    notes = " ".join(_unverified(GateInputs(policy_kind="trained", controls=controls), output))
    assert "probe_hit_rate・nearest_train_hit_rate(未計算)" in notes
    assert "prompt_shuffle_delta" not in notes
    output.shuffled = []
    assert _controls_of(output, {}) == {}
    both = " ".join(_unverified(GateInputs(policy_kind="trained"), output))
    assert "prompt_shuffle_delta・probe_hit_rate・nearest_train_hit_rate(未計算)" in both


def probe_row(condition: str, prompt: str, k: int, hit: bool, failure: str | None = None):
    return ProbeResult(condition, prompt, k, 0, 180, hit, failure)


def test_probe_hit_rate_counts_failures_as_misses_and_needs_both_conditions() -> None:
    output = output_with({"a": 0.5}, {"a": 0.5})
    output.probes = [
        probe_row("before", "赤い", 0, False),
        probe_row("before", "赤い", 1, True),
        probe_row("after", "赤い", 0, True),
        probe_row("after", "赤い", 1, False, "policy_error"),
    ]
    assert probe_hit_rate_control(output) == {"before": 0.5, "after": 0.5, "n": 2}
    output.probes = [p for p in output.probes if p.condition == "after"]
    assert probe_hit_rate_control(output) is None
    output.probes = []
    assert probe_hit_rate_control(output) is None


def test_probe_set_uses_distinct_palette_colors_of_the_named_hues() -> None:
    from image_drawer.drawing.palette import palette_rgb

    colors = {prompt: palette_rgb(color) for prompt, color in PROBES}
    assert colors == {
        "赤い": (255, 0, 0),
        "青い": (0, 0, 255),
        "緑の": (0, 255, 0),
        "黄色い": (255, 255, 0),
    }


def test_report_controls_include_nearest_train_hit_rate_when_memorization_was_checked() -> None:
    output = output_with({i: 0.9 for i in IDS}, {i: 0.3 for i in IDS})
    before = [replace(g, condition="before") for g in output.generations]
    output.generations = [*before, *output.generations]
    for item in output.generations:
        item.nearest_train_distance = 0.5 if item.condition == "before" else 0.01
    output.memorization = {"tau": 0.1, "train_count": 3}
    controls = _controls_of(output, {"bootstrap": {"resamples": 50, "rng_seed": 3}})
    near = controls["nearest_train_hit_rate"]
    assert (near["before"], near["after"], near["tau"]) == (0.0, 1.0, 0.1)
    notes = " ".join(_unverified(GateInputs(policy_kind="trained", controls=controls), output))
    assert "nearest_train_hit_rate" not in notes
