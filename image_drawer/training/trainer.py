"""Trainer: rolloutの収集、採点、更新、checkpoint保存を統括する。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from image_drawer.dataset import PolicyInput, RewardTarget, iter_samples
from image_drawer.training.checkpoint import CheckpointRef, load_checkpoint, save_checkpoint
from image_drawer.training.config import TrainConfig
from image_drawer.training.experiment import (
    ExperimentRecord,
    append_metrics,
    build_record,
    write_record,
)
from image_drawer.training.integrity import verify_allowed_actions, verify_trial_replay
from image_drawer.training.observation import ObservationBuilder
from image_drawer.training.optimizer import (
    NON_UPDATING_END_REASONS,
    BaselineTable,
    UpdateStats,
    reinforce_update,
)
from image_drawer.training.policy import DrawingPolicy
from image_drawer.training.policy_params import AdamState
from image_drawer.training.reward import RewardEvaluator, TargetFeatures
from image_drawer.training.rollout import RolloutRunner, TrainingRollout
from image_drawer.training.stroke_stats import stroke_stats
from image_drawer.training.text_encoder import TextEncoder, get_text_encoder

STREAM_INIT, STREAM_SPLIT, STREAM_TRIAL = 0, 1, 2


# {
#   責務: [ derive_seed: master_seedと経路から独立な32bit乱数seedを決定的に作る ]
#   処理: [ 状態を持たず(iteration, 試行番号)だけで決まるため、再開しても同じ列になる ]
# }
def derive_seed(master_seed: int, *path: int) -> int:
    return int(np.random.SeedSequence([master_seed, *path]).generate_state(1)[0])


# {
#   責務: [
#     Trainer: rollout->採点->REINFORCE更新->checkpointの1周を回す。Policyは目標を見ない
#   ]
#   フィールド: [ config, policy, text_encoder, evaluator, output_dir, checkpoint_dir, adam, baselines ]
# }
class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        policy: DrawingPolicy,
        text_encoder: TextEncoder,
        evaluator: RewardEvaluator,
        output_dir: Path,
        checkpoint_dir: Path | None = None,
        verify_integrity: bool = True,
    ) -> None:
        self.config = config
        self.policy = policy
        self.text_encoder = text_encoder
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir or output_dir / "checkpoints"
        self.verify_integrity = verify_integrity
        self.adam = AdamState(policy.state_dict())
        self.baselines = BaselineTable()
        self.runner = RolloutRunner(
            policy,
            text_encoder,
            ObservationBuilder(config.tail_length, config.summary_side),
            config,
        )
        self._targets: dict[tuple[str, str], TargetFeatures] = {}

    # {
    #   責務: [ _target_features: 目標特徴を画像ごとに1回だけ作って再利用する ]
    # }
    def _target_features(self, target: RewardTarget) -> TargetFeatures:
        key = (target.image_id, target.image_path)
        if key not in self._targets:
            self._targets[key] = self.evaluator.prepare_target(target)
        return self._targets[key]

    # {
    #   責務: [ collect_rollout: batchの各画像にK個のseedでtrialを実行し、終了後に採点する ]
    #   処理: [
    #     1: カリキュラムの打切り工程をiterationから決める
    #     2: 各trialをRolloutRunnerで実行する(目標は渡さない)
    #     3: 整合検査の後、RewardEvaluatorで採点する(trial終了後にだけ)
    #   ]
    #   引数: [ iteration, samples: (PolicyInput, RewardTarget)の列 ]
    #   戻り値: [ rollout: trialと報酬(同順) ]
    # }
    def collect_rollout(
        self, iteration: int, samples: Sequence[tuple[PolicyInput, RewardTarget]]
    ) -> TrainingRollout:
        stop = self.config.curriculum.stop_after_stage(iteration)
        trials, rewards = [], []
        for sample_index, (policy_input, target) in enumerate(samples):
            for seed_index in range(self.config.seeds_per_image):
                seed = derive_seed(
                    self.config.master_seed, STREAM_TRIAL, iteration, sample_index, seed_index
                )
                trial = self.runner.run_trial(policy_input, seed, stop)
                if self.verify_integrity and trial.end_reason not in NON_UPDATING_END_REASONS:
                    verify_trial_replay(trial)
                    verify_allowed_actions(trial)
                trials.append(trial)
                rewards.append(
                    self.evaluator.evaluate_trial(
                        trial, self._target_features(target), policy_input.prompt
                    )
                )
        return TrainingRollout(f"rollout-{iteration}", iteration, tuple(trials), tuple(rewards))

    # {
    #   責務: [ update_policy: rolloutの全trialでPolicyを1回更新する ]
    # }
    def update_policy(self, rollout: TrainingRollout) -> UpdateStats:
        pairs = list(zip(rollout.trials, rollout.rewards))
        return reinforce_update(self.policy, self.adam, pairs, self.baselines, self.config)

    # {
    #   責務: [ save_checkpoint: 現在の状態を保存する ]
    # }
    def save_checkpoint(self, iteration: int) -> CheckpointRef:
        return save_checkpoint(
            self.checkpoint_dir, iteration, self.policy, self.adam, self.baselines,
            self.config, self.text_encoder,
        )  # fmt: skip

    # {
    #   責務: [ _pick_samples: iterationから決定的にB枚を選ぶ(画像が足りなければ重複を許す) ]
    # }
    def _pick_samples(self, iteration: int, pool: list) -> list:
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([self.config.master_seed, STREAM_SPLIT, iteration])
            )
        )
        batch = self.config.batch_images
        indices = rng.choice(len(pool), size=batch, replace=len(pool) < batch)
        return [pool[int(i)] for i in indices]

    # {
    #   責務: [ train: 学習ループを回してExperimentRecordを書き出す ]
    #   処理: [
    #     1: train splitを読む 2: 開始(または再開)時のcheckpointを保存/読込
    #     3: iterationごとにrollout->更新->metrics追記。半数超のtrialが例外なら中断
    #     4: checkpoint_everyごとと終了時に保存し、experiment.jsonを書く
    #   ]
    #   引数: [ dataset_dir, resume_from: iter_N.json, stop_after_iteration: 途中停止(テスト・中断再開用) ]
    #   戻り値: [ record ]
    #   エラー: [ ValueError: train splitが空 ]
    # }
    def train(
        self,
        dataset_dir: Path,
        resume_from: Path | None = None,
        stop_after_iteration: int | None = None,
    ) -> ExperimentRecord:
        pool = list(iter_samples(dataset_dir, "train"))
        if not pool:
            raise ValueError("train splitに学習できる画像がありません")
        metrics_path = self.output_dir / "metrics.jsonl"
        start = 0
        if resume_from is not None:
            start = load_checkpoint(
                resume_from, self.policy, self.adam, self.baselines, self.text_encoder, self.config
            )
            sidecar = json.loads(resume_from.read_text(encoding="utf-8"))
            before = CheckpointRef(
                str(resume_from.parent / sidecar["npz"]), sidecar["npz_sha256"],
                sidecar["policy_version"],
            )  # fmt: skip
        else:
            before = self.save_checkpoint(0)
        end = min(self.config.iterations, stop_after_iteration or self.config.iterations)
        counts: Counter[str] = Counter()
        abort_reason: str | None = None
        for iteration in range(start, end):
            rollout = self.collect_rollout(iteration, self._pick_samples(iteration, pool))
            errors = sum(t.end_reason in NON_UPDATING_END_REASONS for t in rollout.trials)
            self._count(counts, rollout)
            if errors * 2 > len(rollout.trials):
                abort_reason = (
                    f"iteration {iteration}: {errors}/{len(rollout.trials)} trialsが"
                    "policy_error/engine_errorで終了したため中断"
                )
                break
            stats = self.update_policy(rollout)
            counts["updates_skipped" if stats.skipped else "updates_applied"] += 1
            append_metrics(metrics_path, self._metrics_row(iteration, rollout, stats))
            if (iteration + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(iteration + 1)
            counts["iterations_completed"] += 1
        final_iteration = start + counts["iterations_completed"]
        after = self.save_checkpoint(final_iteration) if final_iteration != 0 else before
        experiment_id = f"exp-{self.config.config_hash()[:8]}"
        record = build_record(
            dataset_dir, self.config, self.text_encoder, experiment_id, before, after,
            dict(counts), str(metrics_path), abort_reason=abort_reason,
            reward_sha256=self.evaluator.spec.sha256,
        )  # fmt: skip
        write_record(record, self.output_dir)
        return record

    # {
    #   責務: [ _count: trial数・受理/拒否・終了理由を集計する(失敗も分母に残す) ]
    # }
    @staticmethod
    def _count(counts: Counter, rollout: TrainingRollout) -> None:
        for trial in rollout.trials:
            counts["trials"] += 1
            counts[f"end_{trial.end_reason}"] += 1
            for step in trial.steps:
                counts["accepted_attempts" if step.accepted else "rejected_attempts"] += 1

    # {
    #   責務: [ _metrics_row: 1 iterationの集計(finish率、収益、更新量) ]
    # }
    def _metrics_row(
        self, iteration: int, rollout: TrainingRollout, stats: UpdateStats
    ) -> dict[str, object]:
        qualities = [r.final_quality for r in rollout.rewards if r.final_quality is not None]
        trials = rollout.trials
        # 受理されたストロークの形(直線率・曲がり角。design_stroke_action.md 6節4)
        shape = stroke_stats(
            step.sample.choice for trial in trials for step in trial.steps if step.accepted
        )
        return {
            "iteration": iteration,
            "trials": len(trials),
            "finish_rate": sum(t.end_reason == "finish" for t in trials) / len(trials),
            "mean_final_quality": float(np.mean(qualities)) if qualities else None,
            "mean_total_return": float(np.mean([r.total_return for r in rollout.rewards])),
            "update_skipped": stats.skipped,
            "skip_reason": stats.skip_reason,
            "grad_norm": stats.grad_norm,
            "param_delta_norm": stats.param_delta_norm,
            "advantage_std": stats.advantage_std,
            "mean_entropy": stats.mean_entropy,
            "stroke_count": shape.count,
            "stroke_single_segment_rate": shape.single_rate,
            "stroke_mean_turn_deg": shape.mean_turn_deg,
            "stroke_truncated_count": shape.truncated,
            "policy_sha256": rollout.trials[0].policy_sha256,
        }


# {
#   責務: [ default_text_encoder: configのtext_encoder_idからencoderを生成する ]
# }
def default_text_encoder(config: TrainConfig) -> TextEncoder:
    return get_text_encoder(config.text_encoder_id)
