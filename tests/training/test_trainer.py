"""Optimizer・checkpoint・Trainer・ExperimentRecordのテスト(設計書8.1・8.3節)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from training_helpers import (  # noqa: E402
    ScriptedPolicy,
    complete_script,
    make_dataset,
    run_scripted,
    small_config,
)

from image_drawer.dataset import RewardTarget, iter_samples  # noqa: E402
from image_drawer.training import (  # noqa: E402
    CheckpointMismatchError,
    HashedNgramTextEncoder,
    TrainConfig,
)
from image_drawer.training.checkpoint import load_checkpoint  # noqa: E402
from image_drawer.training.optimizer import (  # noqa: E402
    BaselineTable,
    discounted_returns,
    mean_log_prob,
    reinforce_update,
)
from image_drawer.training.policy import ParametricPolicy  # noqa: E402
from image_drawer.training.policy_params import AdamState, PolicyGrads  # noqa: E402
from image_drawer.training.reward import RewardEvaluator, TrialReward  # noqa: E402
from image_drawer.training.trainer import Trainer, derive_seed  # noqa: E402

ENCODER_DIM = 8


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("trainer_data"), count=12)


def make_config(**overrides) -> TrainConfig:
    values = dict(
        batch_images=2, seeds_per_image=2, iterations=2, max_actions=24, checkpoint_every=1,
        master_seed=11,
    )  # fmt: skip
    return small_config(**{**values, **overrides})


def make_trainer(config: TrainConfig, output_dir: Path, policy=None) -> Trainer:
    encoder = HashedNgramTextEncoder(dim=ENCODER_DIM)
    policy = policy or ParametricPolicy(
        config, ENCODER_DIM, np.random.default_rng(derive_seed(config.master_seed, 0))
    )
    evaluator = RewardEvaluator(config.reward, config.canvas_side)
    return Trainer(config, policy, encoder, evaluator, output_dir)


def test_trainer_updates_parameters(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(make_config(), tmp_path)
    before = trainer.policy.state_dict()
    samples = list(iter_samples(dataset_dir, "train"))[:2]
    rollout = trainer.collect_rollout(0, samples)
    recorded = [s.sample for t in rollout.trials for s in t.steps][:200]
    logp_before = mean_log_prob(trainer.policy, recorded)
    stats = trainer.update_policy(rollout)
    assert not stats.skipped and stats.param_delta_norm > 0
    after = trainer.policy.state_dict()
    delta = np.sqrt(sum(float(((after[k] - before[k]) ** 2).sum()) for k in before))
    assert delta > 0
    assert mean_log_prob(trainer.policy, recorded) != logp_before


def test_default_update_moves_logits_only_slightly(dataset_dir: Path, tmp_path: Path) -> None:
    # Adamの初回は全パラメータを約lr動かすため、連続特徴の多い方策ではlogitが
    # lr x (特徴のL1ノルム)だけ動く。lr=0.01では約1.8動いて方策が1回で崩れた(entropy 6.85->1.96)。
    config = make_config(canvas_side=32, summary_side=8, max_actions=96)
    trainer = make_trainer(config, tmp_path)
    samples = list(iter_samples(dataset_dir, "train"))[:2]
    rollout = trainer.collect_rollout(0, samples)
    features = [s.sample.features for t in rollout.trials for s in t.steps][:100]
    before = np.array([trainer.policy._logits("kind", f) for f in features])
    stats = trainer.update_policy(rollout)
    after = np.array([trainer.policy._logits("kind", f) for f in features])
    assert not stats.skipped
    assert np.abs(after - before).max() < 0.5


def test_trainer_skips_update_when_advantage_std_is_zero() -> None:
    config = make_config()
    policy = ParametricPolicy(config, ENCODER_DIM, np.random.default_rng(0))
    trial = run_scripted(complete_script())
    flat = TrialReward(tuple(), None, 0.0)
    object.__setattr__(flat, "steps", tuple(_zero_breakdown(step) for step in trial.steps))
    before = policy.params_sha256()
    stats = reinforce_update(
        policy, AdamState(policy.params), [(trial, flat)], BaselineTable(), config
    )
    assert stats.skipped and stats.skip_reason == "zero_advantage_std"
    assert policy.params_sha256() == before


def _zero_breakdown(step):
    from image_drawer.training.reward import RewardBreakdown

    return RewardBreakdown(step.stage_id, "step", {}, {}, None, None, {}, 0.0, "p", "v")


def test_trainer_clips_gradients() -> None:
    policy = ParametricPolicy(make_config(), ENCODER_DIM, np.random.default_rng(0))
    grads = PolicyGrads(policy.params)
    grads.arrays["kind"][:] = 1e6
    reported = grads.clip(5.0)
    assert reported > 5.0 and grads.norm() == pytest.approx(5.0)
    small = PolicyGrads(policy.params)
    small.arrays["kind"][0, 0] = 1.0
    assert small.clip(5.0) == 1.0 and small.norm() == 1.0
    assert discounted_returns([1.0, 1.0, 1.0], 0.5) == pytest.approx([1.75, 1.5, 1.0])


def test_trainer_aborts_when_most_trials_error(dataset_dir: Path, tmp_path: Path) -> None:
    config = make_config()
    policy = ScriptedPolicy(complete_script(), raise_at=0)
    trainer = make_trainer(config, tmp_path, policy=policy)
    record = trainer.train(dataset_dir)
    assert record.abort_reason and "policy_error" in record.abort_reason
    assert record.counts["end_policy_error"] == 4 and record.counts.get("updates_applied", 0) == 0
    assert (tmp_path / "experiment.json").is_file()


def test_checkpoint_roundtrip_and_sha_verification(dataset_dir: Path, tmp_path: Path) -> None:
    config = make_config()
    trainer = make_trainer(config, tmp_path)
    ref = trainer.save_checkpoint(3)
    sidecar = tmp_path / "checkpoints" / "iter_3.json"
    assert ref.sha256 == json.loads(sidecar.read_text(encoding="utf-8"))["npz_sha256"]
    restored = make_trainer(config, tmp_path / "other", policy=ParametricPolicy(
        config, ENCODER_DIM, np.random.default_rng(99)))  # fmt: skip
    assert restored.policy.params_sha256() != trainer.policy.params_sha256()
    iteration = load_checkpoint(
        sidecar, restored.policy, restored.adam, restored.baselines, restored.text_encoder, config
    )
    assert iteration == 3 and restored.policy.params_sha256() == trainer.policy.params_sha256()
    # 改ざん(1バイト書換え)はSHA-256不一致になる。
    npz = tmp_path / "checkpoints" / "iter_3.npz"
    data = bytearray(npz.read_bytes())
    data[-10] ^= 0xFF
    npz.write_bytes(bytes(data))
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(sidecar, restored.policy, restored.adam, restored.baselines,
                        restored.text_encoder)  # fmt: skip
    # 別のencoder次元・設定は拒否する。
    trainer.save_checkpoint(4)
    other_encoder = HashedNgramTextEncoder(dim=16)
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(tmp_path / "checkpoints" / "iter_4.json", restored.policy,
                        restored.adam, restored.baselines, other_encoder)  # fmt: skip
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(tmp_path / "checkpoints" / "iter_4.json", restored.policy,
                        restored.adam, restored.baselines, restored.text_encoder,
                        make_config(master_seed=1))  # fmt: skip


def test_checkpoint_rejects_pickle(tmp_path: Path) -> None:
    import hashlib

    config = make_config()
    trainer = make_trainer(config, tmp_path)
    trainer.save_checkpoint(1)
    npz = tmp_path / "checkpoints" / "iter_1.npz"
    sidecar = tmp_path / "checkpoints" / "iter_1.json"
    np.savez(npz, **{"p/kind": np.array([{"evil": 1}], dtype=object)})
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(sidecar, trainer.policy, trainer.adam, trainer.baselines,
                        trainer.text_encoder)  # fmt: skip


def test_training_smoke_updates_parameters_and_writes_records(
    dataset_dir: Path, tmp_path: Path
) -> None:
    trainer = make_trainer(make_config(), tmp_path)
    before = trainer.policy.params_sha256()
    record = trainer.train(dataset_dir)
    assert trainer.policy.params_sha256() != before
    assert record.policy_before.sha256 != record.policy_after.sha256
    rows = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and json.loads(rows[0])["iteration"] == 0
    assert (tmp_path / "experiment.json").is_file()
    assert (tmp_path / "checkpoints" / "iter_2.npz").is_file()


def test_experiment_record_contains_required_fields(dataset_dir: Path, tmp_path: Path) -> None:
    record = make_trainer(make_config(), tmp_path).train(dataset_dir)
    assert record.missing_fields() == []
    saved = json.loads((tmp_path / "experiment.json").read_text(encoding="utf-8"))
    assert saved["reward"]["reward_impl_version"] == "stage-reward-impl-v0"
    assert saved["extractors"]["prompt_builder_version"] and saved["dataset_hash"]
    assert saved["renderer_profile"] == "brush-raster-v2" and saved["environment"]["numpy"]
    assert saved["config"]["max_actions"] == 24


def test_experiment_record_counts_failures_in_denominator(
    dataset_dir: Path, tmp_path: Path
) -> None:
    record = make_trainer(make_config(), tmp_path).train(dataset_dir)
    counts = record.counts
    ends = sum(v for k, v in counts.items() if k.startswith("end_"))
    assert counts["trials"] == ends == 8
    assert counts.get("end_action_budget", 0) + counts.get("end_attempt_budget", 0) > 0
    assert counts["accepted_attempts"] + counts.get("rejected_attempts", 0) > 0


def test_training_is_reproducible_with_same_seed(dataset_dir: Path, tmp_path: Path) -> None:
    runs = []
    for name in ("a", "b"):
        trainer = make_trainer(make_config(), tmp_path / name)
        record = trainer.train(dataset_dir)
        rollout = trainer.collect_rollout(0, list(iter_samples(dataset_dir, "train"))[:1])
        actions = [s.proposal for t in rollout.trials for s in t.steps]
        runs.append((trainer.policy.params_sha256(), record.policy_after.sha256, actions))
    assert runs[0] == runs[1]


def test_resume_matches_uninterrupted_training(dataset_dir: Path, tmp_path: Path) -> None:
    config = make_config(iterations=3)
    straight = make_trainer(config, tmp_path / "straight")
    straight.train(dataset_dir)
    first = make_trainer(config, tmp_path / "resume")
    first.train(dataset_dir, stop_after_iteration=1)
    second = make_trainer(
        config, tmp_path / "resume",
        policy=ParametricPolicy(config, ENCODER_DIM, np.random.default_rng(1234)),
    )  # fmt: skip
    second.train(dataset_dir, resume_from=tmp_path / "resume" / "checkpoints" / "iter_1.json")
    assert second.policy.params_sha256() == straight.policy.params_sha256()
    assert second.baselines.values == straight.baselines.values


def test_swapping_reward_target_keeps_all_proposals_identical(
    dataset_dir: Path, tmp_path: Path
) -> None:
    trainer = make_trainer(make_config(), tmp_path)
    (policy_input, target), (_other_input, other_target) = list(iter_samples(dataset_dir, "train"))[
        :2
    ]
    swapped = RewardTarget(
        policy_input.image_id, other_target.image_path, other_target.image_sha256
    )
    first = trainer.collect_rollout(0, [(policy_input, target)])
    second = trainer.collect_rollout(0, [(policy_input, swapped)])
    proposals = lambda r: [[s.proposal for s in t.steps] for t in r.trials]  # noqa: E731
    assert proposals(first) == proposals(second)
    assert [t.policy_sha256 for t in first.trials] == [t.policy_sha256 for t in second.trials]
    # 学習前のPolicyは描かないことが多いので、Rewardが変わることは描画する台本Trialで確かめる。
    drawn = run_scripted(complete_script())
    rewards = [
        trainer.evaluator.evaluate_trial(
            drawn, trainer.evaluator.prepare_target(t), "x"
        ).total_return
        for t in (target, swapped)
    ]
    assert rewards[0] != rewards[1]


def test_reward_is_computed_only_after_trial_ends(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(make_config(), tmp_path)
    inner = trainer.policy
    observed: list[int] = []
    original_act = inner.act

    def spy(observation, rng):
        observed.append(trainer.evaluator.calls)
        return original_act(observation, rng)

    inner.act = spy  # type: ignore[method-assign]
    samples = list(iter_samples(dataset_dir, "train"))[:1]
    trainer.collect_rollout(0, samples)
    # trial k の実行中は、評価がk回(それ以前のtrial分)だけ完了している。
    expected = sorted(set(observed))
    assert expected == list(range(len(expected)))
    assert trainer.evaluator.calls == 2


def _saved(tmp_path: Path, config: TrainConfig):
    trainer = make_trainer(config, tmp_path)
    trainer.save_checkpoint(1)
    return trainer, tmp_path / "checkpoints"


def _rewrite_npz(directory: Path, mutate) -> Path:
    import hashlib

    npz, sidecar = directory / "iter_1.npz", directory / "iter_1.json"
    with np.load(npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    mutate(arrays)
    np.savez(npz, **arrays)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    return sidecar


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("v/stroke_start"),
        lambda a: a.__setitem__("v/kind", a["v/kind"][:, :-1]),
        lambda a: a.__setitem__("adam_step", np.array([], dtype=np.int64)),
        lambda a: a.__setitem__("baseline_values", np.array([1.0, 2.0])),
    ],
)
def test_checkpoint_load_is_atomic_and_validates_all_arrays(tmp_path: Path, mutate) -> None:
    config = make_config()
    trainer, directory = _saved(tmp_path / "src", config)
    sidecar = _rewrite_npz(directory, mutate)
    target = make_trainer(config, tmp_path / "dst", policy=ParametricPolicy(
        config, ENCODER_DIM, np.random.default_rng(77)))  # fmt: skip
    policy_hash, adam_step = target.policy.params_sha256(), target.adam.step_count
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(sidecar, target.policy, target.adam, target.baselines, target.text_encoder)
    # 失敗しても、policy・Adam・baselineのどれも書き換わらない。
    assert target.policy.params_sha256() == policy_hash
    assert target.adam.step_count == adam_step and target.baselines.values == {}
    assert policy_hash != trainer.policy.params_sha256()


def test_checkpoint_sidecar_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    import os

    config = make_config()
    trainer = make_trainer(config, tmp_path)
    replaced: list[str] = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (replaced.append(Path(b).name), real(a, b))[1])
    trainer.save_checkpoint(2)
    assert "iter_2.json" in replaced and "iter_2.npz" in replaced
    assert not list((tmp_path / "checkpoints").glob("*.tmp*"))
