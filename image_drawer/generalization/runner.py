"""評価ランナー。同じ評価集合・推論seed・予算・canvasで各条件を生成し、指標と失敗を集める。

失敗した生成は分母に残し、品質はITT(失敗にblank得点)と条件付きの両方で集計できる形で保存する。
目標画像を読むのは評価器(RewardEvaluator)だけで、RolloutRunnerには渡さない。
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from image_drawer.core import SerializableModel
from image_drawer.dataset import PolicyInput, RewardTarget
from image_drawer.drawing.replay import replay
from image_drawer.drawing.stage.controller import AttemptRecord
from image_drawer.generalization.controls import PROBES, shuffle_targets
from image_drawer.generalization.metrics_process import (
    GenerationCost,
    ProcessSample,
    compute_process_metrics,
    make_process_sample,
    summarize_costs,
)
from image_drawer.generalization.metrics_quality import (
    blank_image,
    component_scores,
    heldout_scores,
    probe_hit,
)
from image_drawer.generalization.stats import inference_seed
from image_drawer.training.reward import RewardEvaluator
from image_drawer.training.reward_features import composite_over_white
from image_drawer.training.rollout import RolloutRunner, Trial

BLANK = "blank"
FAILURE_KINDS = (
    "draft_budget_exhausted",
    "draft_attempt_limit",
    "draft_event_limit",
    "draft_cancelled",
    "policy_error",
    "engine_error",
    "timeout",
    "reward_eval_error",
    "replay_mismatch",
    "dataset_exclusion",
)
_FAILURE_BY_END_REASON = {
    "action_budget": "draft_budget_exhausted",
    "attempt_budget": "draft_attempt_limit",
    "policy_error": "policy_error",
    "engine_error": "engine_error",
    "curriculum_stop": "draft_cancelled",
}


# {
#   責務: [
#     GenerationResult: 1回の生成(条件×画像×推論seed)の結果。失敗でも1件として残す
#   ]
#   フィールド: [
#     condition, image_id, group_id, k, seed: 生成の識別
#     failure_kind: 失敗の種類(完成ならNone)
#     metrics: 指標名 -> 得点(Reward成分と独立指標)。失敗した生成は空
#     audit_passed: 監査の合否。Episodeが無ければNone
#     replay_exact: 再生した画像が生成時の画像とバイト単位で一致したか
#     cost: GenerationCostの辞書
#   ]
# }
@dataclass(slots=True)
class GenerationResult(SerializableModel):
    condition: str
    image_id: str
    group_id: str
    k: int
    seed: int
    failure_kind: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    audit_passed: bool | None = None
    replay_exact: bool | None = None
    cost: dict[str, Any] = field(default_factory=dict)


# {
#   責務: [
#     ProbeResult: probeのpromptで生成した1回の結果。失敗(完成しない)も命中しなかった1件として残す
#   ]
#   フィールド: [ condition, prompt, k, seed: 生成の識別, expected_color_id: 期待色, hit: 命中したか, failure_kind ]
# }
@dataclass(slots=True)
class ProbeResult(SerializableModel):
    condition: str
    prompt: str
    k: int
    seed: int
    expected_color_id: int
    hit: bool
    failure_kind: str | None = None


# {
#   責務: [
#     RunOutput: 1回の実行の全結果。JSONで保存し、compareとreportが読む
#   ]
#   フィールド: [
#     eval_set_id, split: 評価集合と分割
#     generations: 全生成。失敗を含む
#     blank_metrics: image_id -> 白紙の指標(ITTで失敗に割り当てる値)
#     process, failures, costs: 条件ごとの工程指標・失敗件数・コスト
#     hardware_ids: 条件ごとの実行環境
#     dataset_exclusions: 学習用promptが空で評価対象外になったtest画像の件数
#     exit_code: 0成功、1失敗の生成を含む、3 Replay不一致・監査器の不一致
#     shuffled: promptを別の画像のものに入れ替えた生成(対照指標prompt_shuffle_delta用)。
#     通常の生成・工程指標・コスト・終了コードには含めない
#     probes: 構成的なprobeのpromptで生成した結果(対照指標probe_hit_rate用)。同じく別枠
#   ]
# }
@dataclass(slots=True)
class RunOutput(SerializableModel):
    eval_set_id: str
    split: str
    generations: list[GenerationResult]
    blank_metrics: dict[str, dict[str, float]]
    process: dict[str, dict[str, Any]]
    failures: dict[str, dict[str, int]]
    costs: dict[str, dict[str, Any]]
    hardware_ids: dict[str, str]
    dataset_exclusions: int = 0
    exit_code: int = 0
    shuffled: list[GenerationResult] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)


# {
#   責務: [ describe_hardware: 実行環境の識別子(OS・Python・NumPy)を返す ]
# }
def describe_hardware() -> str:
    return f"{platform.platform()}|py{sys.version.split()[0]}|numpy{np.__version__}"


# {
#   責務: [ attempts_of: Trialの各stepから、監査器の指標が使う提案記録を作る ]
# }
def attempts_of(trial: Trial) -> tuple[AttemptRecord, ...]:
    return tuple(
        AttemptRecord(
            step.attempt_index,
            step.proposal,
            step.accepted,
            step.reject_reason,
            step.stage_id,
            tuple(event.seq for event in step.events),
        )
        for step in trial.steps
    )


# {
#   責務: [ _cost_of: Trialと実時間から、取得できる値だけのコストを作る(Engine内部の値と内訳は未取得) ]
# }
def _cost_of(trial: Trial, wall_total: float) -> GenerationCost:
    events = trial.episode.events
    return GenerationCost(
        accepted_actions=sum(1 for e in events if e.actor == "model"),
        attempts=len(trial.steps),
        system_events=sum(1 for e in events if e.actor == "system"),
        renderer_points_used=None,
        renderer_area_used=None,
        wall_policy_seconds=None,
        wall_engine_seconds=None,
        wall_total_seconds=wall_total,
    )


# {
#   責務: [
#     _score_generation: 完成した生成を採点し、再生の一致を確認する
#   ]
#   戻り値: [
#     (metrics, replay_exact, failure_kind): 失敗しなければfailure_kindはNone
#   ]
# }
def _score_generation(
    trial: Trial, evaluator: RewardEvaluator, features, side: int
) -> tuple[dict[str, float], bool, str | None]:
    try:
        rgb = composite_over_white(trial.final_composite, side)
        metrics = component_scores(rgb, features.rgb, list(evaluator.spec.final_weights))
        metrics.update(heldout_scores(rgb, features.rgb))
    except Exception:
        return {}, False, "reward_eval_error"
    try:
        exact = replay(trial.episode).image.rgba == trial.final_composite
    except Exception:
        exact = False
    return metrics, exact, (None if exact else "replay_mismatch")


# {
#   責務: [
#     _generate: 1条件・1画像・1seedを生成して結果を作る
#   ]
# }
def _generate(
    condition: str,
    runner: RolloutRunner,
    evaluator: RewardEvaluator,
    policy_input: PolicyInput,
    features,
    group_id: str,
    k: int,
    seed: int,
    sink: Callable[[GenerationResult, Trial], None] | None = None,
) -> tuple[GenerationResult, ProcessSample | None, GenerationCost]:
    started = time.perf_counter()
    trial = runner.run_trial(policy_input, seed, None)
    cost = _cost_of(trial, time.perf_counter() - started)
    failure = _FAILURE_BY_END_REASON.get(trial.end_reason)
    metrics: dict[str, float] = {}
    exact: bool | None = None
    if failure is None:
        metrics, exact, failure = _score_generation(
            trial, evaluator, features, runner.config.canvas_side
        )
    try:
        sample = make_process_sample(
            trial.episode,
            attempts_of(trial),
            attempt_limit_hit=trial.end_reason == "attempt_budget",
        )
    except Exception:
        sample = None
    result = GenerationResult(
        condition,
        policy_input.image_id,
        group_id,
        k,
        seed,
        failure,
        metrics if failure is None else {},
        sample.audit.passed if sample and sample.audit else None,
        exact,
        asdict(cost),
    )
    if sink is not None:
        sink(result, trial)
    return result, sample, cost


# {
#   責務: [
#     _run_probes: 各条件で、probeのpromptごとに同じ推論seedで生成し、期待色が最大面積かを判定する
#   ]
#   処理: [
#     目標画像は使わない(promptだけで生成する)。完成しなかった生成は命中に数えず、分母に残す。
#     seedは条件に依存しない(対応づけ)
#   ]
# }
def _run_probes(
    conditions: Mapping[str, RolloutRunner], eval_set_id: str, seeds: int, side: int
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for prompt, color_id in PROBES:
        probe_id = f"probe:{prompt}"
        for k in range(seeds):
            seed = inference_seed(eval_set_id, probe_id, k)
            for name, runner in conditions.items():
                trial = runner.run_trial(PolicyInput(probe_id, prompt), seed, None)
                failure = _FAILURE_BY_END_REASON.get(trial.end_reason)
                hit = False
                if failure is None:
                    try:
                        hit = probe_hit(composite_over_white(trial.final_composite, side), color_id)
                    except Exception:
                        failure = "reward_eval_error"
                results.append(ProbeResult(name, prompt, k, seed, color_id, hit, failure))
    return results


# {
#   責務: [
#     run_generations: 各条件を、同じ画像・推論seed・予算・canvasで生成し、指標と失敗を集める
#   ]
#   処理: [
#     1: 画像ごとに目標を1回だけ評価器へ読み込み、白紙の指標(ITTの基準)を求める
#     2: 推論seedごとに全条件を生成する。seedは条件に依存しない(対応づけ)
#     3: 条件ごとに工程指標・失敗件数・コストを集計し、終了コードを決める
#   ]
#   引数: [
#     conditions: 条件名 -> RolloutRunner(blankは渡さない。白紙は内部で扱う)
#     evaluator: RewardEvaluator, samples: (入力, 目標)の組, group_of: image_id -> group_id
#     eval_set_id, split, seeds_per_item: 推論seedの数K, dataset_exclusions: 対象外となった画像数
#     hardware_ids: 条件ごとの実行環境(省略時は現在の環境)
#     sink: 生成ごとにTrialを受け取る保存用の関数(人間評価用の画像やEpisodeの保存)
#   ]
# }
def run_generations(
    conditions: Mapping[str, RolloutRunner],
    evaluator: RewardEvaluator,
    samples: Sequence[tuple[PolicyInput, RewardTarget]],
    group_of: Mapping[str, str],
    eval_set_id: str,
    split: str,
    *,
    seeds_per_item: int = 5,
    dataset_exclusions: int = 0,
    hardware_ids: Mapping[str, str] | None = None,
    sink: Callable[[GenerationResult, Trial], None] | None = None,
    shuffle_prompts: bool = False,
    probe: bool = False,
) -> RunOutput:
    sides = {runner.config.canvas_side for runner in conditions.values()}
    budgets = {(r.config.max_actions, r.config.max_attempts) for r in conditions.values()}
    if len(sides) != 1 or len(budgets) != 1 or BLANK in conditions:
        raise ValueError("all conditions must share canvas and budget, and 'blank' is reserved")
    side = sides.pop()
    ordered = sorted(samples, key=lambda pair: pair[0].image_id)
    # 入れ替え先のpromptは、評価集合IDだけで決まる固定の対応で選ぶ(条件・seedに依存しない)
    prompts = {pair[0].image_id: pair[0].prompt for pair in ordered}
    targets = (
        shuffle_targets(list(prompts), eval_set_id) if shuffle_prompts else {}
    )  # fmt: skip
    shuffled: list[GenerationResult] = []
    generations: list[GenerationResult] = []
    process_samples: dict[str, list[ProcessSample]] = {name: [] for name in conditions}
    costs: dict[str, list[GenerationCost]] = {name: [] for name in conditions}
    blank_metrics: dict[str, dict[str, float]] = {}
    for policy_input, target in ordered:
        features = evaluator.prepare_target(target)
        group_id = group_of.get(policy_input.image_id, policy_input.image_id)
        blank = blank_image(side)
        blank_metrics[policy_input.image_id] = {
            **component_scores(blank, features.rgb, list(evaluator.spec.final_weights)),
            **heldout_scores(blank, features.rgb),
        }
        for k in range(seeds_per_item):
            seed = inference_seed(eval_set_id, policy_input.image_id, k)
            for name, runner in conditions.items():
                result, sample, cost = _generate(
                    name, runner, evaluator, policy_input, features, group_id, k, seed, sink
                )
                generations.append(result)
                costs[name].append(cost)
                if sample is not None:
                    process_samples[name].append(sample)
                if shuffle_prompts:
                    swapped = PolicyInput(
                        policy_input.image_id, prompts[targets[policy_input.image_id]]
                    )
                    swapped_result, _, _ = _generate(
                        name, runner, evaluator, swapped, features, group_id, k, seed
                    )
                    shuffled.append(swapped_result)
    probes = _run_probes(conditions, eval_set_id, seeds_per_item, side) if probe else []
    failures = {name: _count_failures(generations, name, dataset_exclusions) for name in conditions}
    process = {n: asdict(compute_process_metrics(s)) for n, s in process_samples.items()}
    output = RunOutput(
        eval_set_id=eval_set_id,
        split=split,
        generations=generations,
        blank_metrics=blank_metrics,
        process=process,
        failures=failures,
        costs={name: summarize_costs(c) for name, c in costs.items()},
        hardware_ids=dict(hardware_ids or {n: describe_hardware() for n in conditions}),
        dataset_exclusions=dataset_exclusions,
        shuffled=shuffled,
        probes=probes,
    )
    output.exit_code = decide_exit_code(output)
    return output


# {
#   責務: [ _count_failures: 条件ごとの失敗の種類別件数。0件の種類も含める ]
# }
def _count_failures(generations: list[GenerationResult], condition: str, exclusions: int) -> dict:
    counts = {kind: 0 for kind in FAILURE_KINDS}
    for generation in generations:
        if generation.condition == condition and generation.failure_kind:
            counts[generation.failure_kind] += 1
    counts["dataset_exclusion"] = exclusions
    return counts


# {
#   責務: [
#     decide_exit_code: 終了コードを決める
#   ]
#   処理: [ Replay不一致か、完成なのに監査が不合格なら3。失敗した生成があれば1。それ以外は0 ]
# }
def decide_exit_code(output: RunOutput) -> int:
    mismatch = any(
        g.failure_kind == "replay_mismatch" or (g.failure_kind is None and g.audit_passed is False)
        for g in output.generations
    )
    if mismatch:
        return 3
    return 1 if any(g.failure_kind for g in output.generations) else 0
