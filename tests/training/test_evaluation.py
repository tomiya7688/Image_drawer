"""ClaimGate・対応ブートストラップ・測定ハーネスのテスト(設計書8.4節)。"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from training_helpers import ScriptedPolicy, complete_script, make_dataset, small_config  # noqa: E402

from image_drawer.dataset import iter_samples  # noqa: E402
from image_drawer.training import HashedNgramTextEncoder  # noqa: E402
from image_drawer.training.evaluation import (  # noqa: E402
    ClaimGate,
    EvalRecord,
    UpdateEvidence,
    action_signature,
    paired_bootstrap_ci,
    run_evaluation,
    swap_prompt,
)
from image_drawer.training.observation import ObservationBuilder  # noqa: E402
from image_drawer.training.reward import RewardEvaluator  # noqa: E402
from image_drawer.training.rollout import RolloutRunner  # noqa: E402

EVIDENCE = UpdateEvidence(param_delta_norm=0.5, log_prob_changed=True)


def table(
    completed: float = 0.2,
    quality: float = 0.1,
    swapped: float = 0.0,
    distance: float = 0.1,
    images: int = 8,
    seeds: int = 5,
    signatures: int = 1000,
    jitter: float = 0.0,
) -> list[EvalRecord]:
    records = []
    for image in range(images):
        shift = jitter if image % 2 == 0 else -jitter
        for seed in range(seeds):
            done = seed < round(completed * seeds)
            records.append(
                EvalRecord(
                    f"img{image}",
                    seed,
                    done,
                    not done and seed % 2 == 0,
                    (quality + shift) if done else 0.0,
                    swapped,
                    distance,
                    f"sig{(image * seeds + seed) % signatures}",
                )  # fmt: skip
            )
    return records


def synthetic_good() -> tuple[list[EvalRecord], list[EvalRecord], list[EvalRecord]]:
    before = table(completed=0.2, quality=0.3)
    after = table(completed=1.0, quality=0.6, swapped=0.3)
    baseline = table(completed=0.4, quality=0.35)
    return before, after, baseline


def test_claim_gate_marks_learned_only_when_every_criterion_passes() -> None:
    before, after, baseline = synthetic_good()
    report = ClaimGate().evaluate(before, after, baseline, EVIDENCE, integrity_ok=True)
    assert report.verdict == "learned" and all(report.passed.values())
    assert set(report.passed) == {"M1", "M2", "M3", "M4", "M5", "M6"}
    not_integer = ClaimGate().evaluate(before, after, baseline, EVIDENCE, integrity_ok=False)
    assert not_integer.verdict == "not_learned"
    frozen = UpdateEvidence(0.0, False)
    assert ClaimGate().evaluate(before, after, baseline, frozen, True).verdict == "not_learned"


def test_claim_gate_marks_not_learned_when_no_improvement() -> None:
    same = table(completed=0.2, quality=0.3, swapped=0.3)
    report = ClaimGate().evaluate(same, same, same, EVIDENCE, integrity_ok=True)
    assert report.verdict == "not_learned"
    assert not report.passed["M2"] and not report.passed["M3"] and not report.passed["M4"]


def test_claim_gate_requires_confidence_interval_lower_bound_above_zero() -> None:
    before = table(completed=0.2, quality=0.3)
    after = table(completed=1.0, quality=0.35, swapped=0.0, jitter=0.5)
    report = ClaimGate().evaluate(before, after, table(0.4, 0.1), EVIDENCE, True)
    assert report.values["m3_mean_gain"] >= 0.05 and report.values["m3_ci_lower"] <= 0
    assert not report.passed["M3"] and report.verdict == "not_learned"


def test_claim_gate_returns_inconclusive_with_too_few_images_or_seeds() -> None:
    few_images = ClaimGate().evaluate(
        table(images=7), table(images=7, completed=1.0), table(images=7), EVIDENCE, True
    )
    few_seeds = ClaimGate().evaluate(
        table(seeds=4), table(seeds=4, completed=1.0), table(seeds=4), EVIDENCE, True
    )
    mismatched = ClaimGate().evaluate(
        table(images=9), table(images=8), table(images=8), EVIDENCE, True
    )
    for report in (few_images, few_seeds, mismatched):
        assert report.verdict == "inconclusive" and report.notes


def test_claim_gate_flags_scripted_suspect() -> None:
    before, after, baseline = synthetic_good()
    scripted = [EvalRecord(**{**r.__dict__, "action_signature": "same"}) for r in after]
    report = ClaimGate().evaluate(before, scripted, baseline, EVIDENCE, True)
    assert report.verdict == "not_learned" and not report.passed["M6"]
    assert any("scripted_suspect" in note for note in report.notes)


def test_paired_bootstrap_ci_is_deterministic_for_seed() -> None:
    differences = list(np.random.default_rng(0).normal(0.1, 0.3, 25))
    first = paired_bootstrap_ci(differences, seed=3)
    assert first == paired_bootstrap_ci(differences, seed=3)
    assert first != paired_bootstrap_ci(differences, seed=4)
    mean, low, high = first
    assert low <= mean <= high
    assert paired_bootstrap_ci([0.2] * 5, seed=1) == pytest.approx((0.2, 0.2, 0.2))
    with pytest.raises(ValueError):
        paired_bootstrap_ci([])


def test_swap_prompt_picks_next_different_prompt() -> None:
    assert swap_prompt(0, ["a", "a", "b"]) == "b"
    assert swap_prompt(2, ["a", "a", "b"]) == "a"
    assert swap_prompt(0, ["a", "a"]) is None


def test_run_evaluation_records_failures_and_swap_metrics(tmp_path: Path) -> None:
    dataset_dir = make_dataset(tmp_path, count=12)
    config = small_config()
    evaluator = RewardEvaluator(config.reward, config.canvas_side)
    runner = RolloutRunner(
        ScriptedPolicy(complete_script()),
        HashedNgramTextEncoder(dim=8),
        ObservationBuilder(config.tail_length, config.summary_side),
        config,
    )
    samples = list(iter_samples(dataset_dir, "train"))[:3]
    records = run_evaluation(runner, evaluator, samples, seeds=[1, 2])
    assert len(records) == 6 and all(r.completed for r in records)
    assert all(0.0 <= r.final_quality <= 1.0 for r in records)
    assert all(r.swapped_quality is not None and r.pixel_distance == 0.0 for r in records)
    assert len({r.action_signature for r in records}) == 1  # 台本Policyは固定Script
    failing = RolloutRunner(
        ScriptedPolicy(complete_script(), raise_at=0),
        HashedNgramTextEncoder(dim=8),
        ObservationBuilder(config.tail_length, config.summary_side),
        config,
    )
    failed = run_evaluation(failing, evaluator, samples, seeds=[1])
    assert len(failed) == 3 and not any(r.completed for r in failed)
    assert action_signature(runner.run_trial(samples[0][0], 1, None)) == records[0].action_signature


@pytest.mark.training_experiment
@pytest.mark.skipif(
    os.environ.get("IMAGE_DRAWER_TRAINING_EXPERIMENT") != "1",
    reason="数十分かかる実測。IMAGE_DRAWER_TRAINING_EXPERIMENT=1で手動実行する",
)
def test_small_training_improves_heldout_quality_vs_untrained(tmp_path: Path) -> None:
    from image_drawer.training.experiment import write_record
    from image_drawer.training.optimizer import mean_log_prob
    from image_drawer.training.policy import ParametricPolicy, RandomLegalPolicy
    from image_drawer.training.trainer import Trainer

    iterations = int(os.environ.get("IMAGE_DRAWER_EXPERIMENT_ITERATIONS", "200"))
    dataset_dir = make_dataset(tmp_path, count=150)
    config = small_config(canvas_side=32, iterations=iterations, max_actions=96, summary_side=8)
    held_out = (
        list(iter_samples(dataset_dir, "test"))[:8]
        or list(iter_samples(dataset_dir, "validation"))[:8]
    )
    encoder = HashedNgramTextEncoder(dim=64)
    evaluator = RewardEvaluator(config.reward, config.canvas_side)
    policy = ParametricPolicy(config, encoder.dim, np.random.default_rng(config.master_seed))

    def make_runner(active) -> RolloutRunner:
        builder = ObservationBuilder(config.tail_length, config.summary_side)
        return RolloutRunner(active, encoder, builder, config)

    seeds = list(range(5))
    untrained = run_evaluation(make_runner(policy), evaluator, held_out, seeds)
    baseline = run_evaluation(make_runner(RandomLegalPolicy(config)), evaluator, held_out, seeds)
    trainer = Trainer(config, policy, encoder, evaluator, tmp_path / "out")
    probe = trainer.collect_rollout(0, list(iter_samples(dataset_dir, "train"))[:1])
    recorded = [s.sample for t in probe.trials for s in t.steps][:500]
    logp_before = mean_log_prob(policy, recorded)
    start = policy.state_dict()
    record = trainer.train(dataset_dir)
    end = policy.state_dict()
    delta = float(np.sqrt(sum(float(((end[k] - start[k]) ** 2).sum()) for k in start)))
    trained = run_evaluation(make_runner(policy), evaluator, held_out, seeds)
    evidence = UpdateEvidence(delta, mean_log_prob(policy, recorded) != logp_before)
    claim = ClaimGate().evaluate(untrained, trained, baseline, evidence, integrity_ok=True)
    final = dataclasses.replace(record, claim=claim.to_json())
    write_record(final, tmp_path / "out")
    # 閾値は事前登録済み。ここでは判定をそのまま保存し、改善は主張しない。
    assert claim.verdict in {"learned", "not_learned", "inconclusive"}


def test_swap_metrics_treats_missing_composite_as_missing_value() -> None:
    import dataclasses as dc

    from image_drawer.training.evaluation import _swap_metrics
    from training_helpers import run_scripted

    trial = run_scripted(complete_script())
    broken = dc.replace(trial, final_composite=None)
    config = small_config()
    evaluator = RewardEvaluator(config.reward, config.canvas_side)
    assert _swap_metrics(evaluator, None, broken, trial, 16) == (None, None)
    assert _swap_metrics(evaluator, None, trial, broken, 16) == (None, None)


def test_claim_gate_handles_empty_tables_and_mismatched_seed_sets() -> None:
    empty = ClaimGate().evaluate([], [], [], EVIDENCE, True)
    assert empty.verdict == "inconclusive" and empty.notes
    before, after, baseline = synthetic_good()
    shifted = [EvalRecord(**{**r.__dict__, "seed": r.seed + 100}) for r in after]
    report = ClaimGate().evaluate(before, shifted, baseline, EVIDENCE, True)
    assert report.verdict == "inconclusive"
    assert any("seed" in note for note in report.notes)
