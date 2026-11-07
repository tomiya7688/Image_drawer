"""汎化評価の工程・stroke・拒否・コスト指標を確認する(設計 6.2〜6.5節)。"""

import subprocess
import sys

import numpy as np
import pytest
from stage_helpers import (
    accept_all,
    advance,
    draw_actions,
    full_cover_actions,
    full_run,
    make_controller,
    model,
    sample,
    setup_actions,
)

from image_drawer.generalization.metrics_quality import (
    blank_image,
    component_scores,
    heldout_scores,
    memorization_tau,
    nearest_train_distance,
    nearest_train_hit_rate,
    probe_hit,
    prompt_shuffle_delta,
    reward_gain_without_heldout_gain,
)
from image_drawer.generalization.metrics_process import (
    GenerationCost,
    ProcessSample,
    compute_process_metrics,
    make_process_sample,
    summarize_costs,
)


# {
#   責務: [ sample_of: Controllerの実行結果から指標の入力を作る ]
# }
def sample_of(controller) -> ProcessSample:
    return make_process_sample(controller.episode(), controller.attempts())


# {
#   責務: [ finished_controller: 全工程を描いてfinishまで進めたController ]
# }
def finished_controller(prompt: str = "猫"):
    controller = make_controller(prompt)
    full_run(controller)
    return controller


def test_effective_stroke_rate_ignores_strokes_that_change_no_pixel():
    # 画布全体を塗った後に同じ列を繰り返すと、2本目は受理されても画素を変えない。
    controller = make_controller()
    accept_all(controller, full_cover_actions("sys_rough_001"))
    accept_all(controller, full_cover_actions("sys_rough_001"))
    metrics = compute_process_metrics([sample_of(controller)])
    assert metrics.accepted_stroke_rate == 1.0
    assert metrics.effective_stroke_rate == pytest.approx(0.5)


def test_dot_stroke_rate_detects_sub_pixel_paths():
    controller = make_controller()
    accept_all(controller, setup_actions("sys_rough_001", "ink_gpen", 0))
    accept_all(controller, [model("stroke_v2", {"samples": [sample(5, 5)]})])
    accept_all(controller, [model("stroke_v2", {"samples": [sample(2, 2, 0), sample(9, 6, 1)]})])
    metrics = compute_process_metrics([sample_of(controller)])
    assert metrics.dot_stroke_rate == pytest.approx(0.5)
    # 1画素の点だけで通過しようとしても、必須工程の変更画素数の中央値に現れる。
    assert metrics.pixels_changed_per_required_stage["rough"] is not None


def test_rejection_rate_breaks_down_by_reason_and_stage():
    controller = make_controller()
    controller.propose(model("skip_stage"))
    controller.propose(advance())
    controller.propose(advance())
    full_run(controller)
    metrics = compute_process_metrics([sample_of(controller)])
    assert metrics.rejection_by_reason == {"action_not_allowed": 1, "stage_not_ready": 2}
    assert metrics.rejection_by_stage == {"rough": 3}
    assert metrics.max_consecutive_rejections == 3
    # 3件の拒否はすべて工程の規則による。
    total = len(controller.attempts())
    assert metrics.rejection_rate == pytest.approx(3 / total)
    assert metrics.stage_violation_rate == pytest.approx(3 / total)


def test_proposal_legal_rate_uses_all_attempts_as_denominator():
    clean = sample_of(finished_controller())
    noisy_controller = make_controller()
    noisy_controller.propose(model("skip_stage"))
    full_run(noisy_controller)
    noisy = sample_of(noisy_controller)
    metrics = compute_process_metrics([clean, noisy])
    total = len(clean.attempts) + len(noisy.attempts)
    assert metrics.proposal_legal_rate == pytest.approx((total - 1) / total)
    assert metrics.completion_rate == 1.0 and metrics.audit_pass_rate == 1.0
    assert metrics.auditor_controller_disagreements == 0
    # 失敗した生成(Episodeなし)は分母に残り、完成率が下がる。
    failed = ProcessSample(attempts=(), audit=None, is_complete=False, attempt_limit_hit=True)
    with_failure = compute_process_metrics([clean, noisy, failed])
    assert with_failure.completion_rate == pytest.approx(2 / 3)
    assert with_failure.attempt_limit_hit_rate == pytest.approx(1 / 3)


def test_cost_fields_use_null_when_memory_is_unavailable():
    costs = [
        GenerationCost(10, 12, 20, 100, 400, 0.5, 0.25, 0.75),
        GenerationCost(20, 24, 40, 300, 800, 1.5, 0.75, 2.25),
    ]
    summary = summarize_costs(costs)
    assert summary["peak_memory_mb"] == {"median": None, "p90": None}
    assert summary["accepted_actions"] == {"median": 15.0, "p90": 20.0}
    assert summarize_costs([])["attempts"] == {"median": None, "p90": None}


RED = np.full((16, 16, 3), [0.85, 0.1, 0.1])
BLUE = np.full((16, 16, 3), [0.1, 0.1, 0.85])
TARGETS = {"a": RED, "b": BLUE, "c": RED, "d": BLUE}
PROMPTS = {"a": "赤", "b": "青", "c": "赤", "d": "青"}
SHUFFLE = {"a": "b", "b": "a", "c": "d", "d": "c"}


# {
#   責務: [ closeness: 画像が目標へ近いほど1に近い試験用の得点 ]
# }
def closeness(image_id: str, image) -> float:
    return float(1.0 - np.abs(image - TARGETS[image_id]).mean())


def test_prompt_shuffle_delta_is_positive_for_prompt_dependent_toy_policy():
    def generate(_image_id: str, prompt: str):
        return RED if prompt == "赤" else BLUE

    delta = prompt_shuffle_delta(list(TARGETS), SHUFFLE, PROMPTS, generate, closeness)
    assert delta > 0.3


def test_prompt_shuffle_delta_is_near_zero_for_prompt_ignoring_policy():
    delta = prompt_shuffle_delta(
        list(TARGETS), SHUFFLE, PROMPTS, lambda _i, _p: np.full((16, 16, 3), 0.5), closeness
    )
    assert delta == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        prompt_shuffle_delta(["a"], {"a": "a"}, PROMPTS, lambda _i, _p: RED, closeness)


def test_nearest_train_hit_rate_flags_copied_train_target():
    def distance(first, second) -> float:
        return float(np.abs(first - second).mean())

    train = [RED, BLUE]
    other = np.full((16, 16, 3), 0.5)
    assert nearest_train_hit_rate([RED.copy()], train, distance, tau=0.01) == 1.0
    assert nearest_train_hit_rate([other], train, distance, tau=0.01) == 0.0
    assert nearest_train_hit_rate([RED, other], train, distance, tau=0.01) == 0.5


def test_quality_metrics_report_each_reward_component_separately():
    names = ["rgb_128", "edge_128", "rgb_hist_16"]
    scores = component_scores(RED, RED, names)
    assert set(scores) == set(names) and "total" not in scores
    worse = component_scores(blank_image(16), RED, names)
    assert all(0.0 <= worse[n] <= scores[n] + 1e-9 for n in names)
    # 独立指標は別名で返し、同一画像では1になる。
    held = heldout_scores(RED, RED)
    assert held["heldout_ssim64"] == pytest.approx(1.0)
    assert heldout_scores(blank_image(16), RED)["heldout_ssim64"] < 0.9
    from image_drawer.drawing.palette import PALETTE

    color_id = 100
    swatch = np.full((16, 16, 3), PALETTE[color_id], dtype=np.float64) / 255.0
    assert probe_hit(swatch, color_id) is True
    assert probe_hit(swatch, (color_id + 1) % 256) is False
    assert probe_hit(blank_image(16), color_id) is False


def test_reward_gain_without_heldout_gain_warning_is_raised():
    assert reward_gain_without_heldout_gain({"rgb": 0.1}, {"heldout_ssim64": -0.2}) is True
    assert reward_gain_without_heldout_gain({"rgb": 0.1}, {"heldout_ssim64": 0.05}) is False
    assert reward_gain_without_heldout_gain({"rgb": -0.1}, {"heldout_ssim64": -0.2}) is False


def test_metrics_never_load_target_inside_policy_process_boundary():
    # Policy側の経路(rollout・policy・observation)を読み込んでも、評価の指標モジュールは読み込まれない。
    code = (
        "import sys; import image_drawer.training.rollout, image_drawer.training.policy, "
        "image_drawer.training.observation; "
        "bad=[m for m in sys.modules if m.startswith('image_drawer.generalization')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_nearest_train_distance_is_zero_for_a_copy_and_positive_for_others():
    stripes = np.zeros((16, 16, 3))
    stripes[:, ::2] = 1.0
    train = [RED, stripes]
    assert nearest_train_distance(stripes.copy(), train) == pytest.approx(0.0, abs=1e-9)
    assert nearest_train_distance(BLUE, [stripes]) > 0.0
    with pytest.raises(ValueError):
        nearest_train_distance(RED, [])


def test_memorization_tau_is_a_lower_quantile_of_validation_to_train_distances():
    stripes = np.zeros((16, 16, 3))
    stripes[:, ::2] = 1.0
    train = [stripes]
    validation = [stripes.copy(), BLUE, RED]
    distances = sorted(nearest_train_distance(v, train) for v in validation)
    assert memorization_tau(validation, train, quantile=0.0) == pytest.approx(distances[0])
    assert memorization_tau(validation, train, quantile=1.0) == pytest.approx(distances[-1])
    with pytest.raises(ValueError):
        memorization_tau([], train)
    with pytest.raises(ValueError):
        memorization_tau(validation, train, quantile=1.5)
