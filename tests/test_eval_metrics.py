"""汎化評価の工程・stroke・拒否・コスト指標を確認する(設計 6.2〜6.5節)。"""

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
