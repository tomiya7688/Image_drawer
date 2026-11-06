"""held-outでの測定、対応ブートストラップ、ClaimGate(学習できたかの3値判定)。

閾値は設計書1.4節で先に固定した値で、実験結果を見て変更しない。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from image_drawer.dataset import PolicyInput, RewardTarget
from image_drawer.training.reward import RewardEvaluator
from image_drawer.training.rollout import RolloutRunner, Trial

MIN_IMAGES = 8
MIN_SEEDS = 5
M2A_MIN_GAIN = 0.30
M3_MIN_GAIN = 0.05
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE_ALPHA = 0.05


# {
#   責務: [ EvalRecord: 1(画像, seed)の測定。失敗・予算超過も1件として残す ]
#   フィールド: [
#     completed: 7工程を通ってfinishしたか, budget_exceeded, final_quality: 未完成は0.0,
#     swapped_quality: 文章を入れ替えた出力を自分の目標で測った品質(入替先なしはNone),
#     pixel_distance: 入れ替え前後の出力の平均絶対画素差(0〜1), action_signature: Action列のハッシュ
#   ]
# }
@dataclass(frozen=True)
class EvalRecord:
    image_id: str
    seed: int
    completed: bool
    budget_exceeded: bool
    final_quality: float
    swapped_quality: float | None
    pixel_distance: float | None
    action_signature: str


# {
#   責務: [ UpdateEvidence: M1の根拠(パラメータ変化量と、記録済みActionの対数確率の変化) ]
# }
@dataclass(frozen=True)
class UpdateEvidence:
    param_delta_norm: float
    log_prob_changed: bool


# {
#   責務: [ ClaimReport: 判定(learned/not_learned/inconclusive)とM1〜M6の値・合否 ]
# }
@dataclass(frozen=True)
class ClaimReport:
    verdict: str
    passed: Mapping[str, bool]
    values: Mapping[str, float | None]
    notes: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


# {
#   責務: [ action_signature: modelのAction列(種別とpayload)の正規化ハッシュ ]
# }
def action_signature(trial: Trial) -> str:
    events = [
        [event.type, event.payload] for event in trial.episode.events if event.actor == "model"
    ]
    text = json.dumps(events, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# {
#   責務: [ swap_prompt: 文章を入れ替える先の固定規則。image_id順で次の、文章が異なる画像の文章 ]
# }
def swap_prompt(index: int, prompts: Sequence[str]) -> str | None:
    for offset in range(1, len(prompts)):
        candidate = prompts[(index + offset) % len(prompts)]
        if candidate != prompts[index]:
            return candidate
    return None


# {
#   責務: [ run_evaluation: 各画像x各seedで、自分の文章と入れ替えた文章の2回を描かせて測定する ]
#   処理: [
#     1: 同じ画像・seed・Budgetで自分の文章のtrialを実行して品質を測る
#     2: 同じseedで入れ替えた文章のtrialを実行し、出力画素距離と、自分の目標での品質を測る
#   ]
#   引数: [ runner: 評価するPolicyのRolloutRunner, evaluator, samples: held-outの(入力, 目標), seeds ]
#   戻り値: [ records ]
# }
def run_evaluation(
    runner: RolloutRunner,
    evaluator: RewardEvaluator,
    samples: Sequence[tuple[PolicyInput, RewardTarget]],
    seeds: Sequence[int],
) -> list[EvalRecord]:
    ordered = sorted(samples, key=lambda pair: pair[0].image_id)
    prompts = [pair[0].prompt for pair in ordered]
    records: list[EvalRecord] = []
    for index, (policy_input, target) in enumerate(ordered):
        features = evaluator.prepare_target(target)
        other_prompt = swap_prompt(index, prompts)
        for seed in seeds:
            trial = runner.run_trial(policy_input, seed, None)
            own = evaluator.evaluate_trial(trial, features, policy_input.prompt)
            swapped_quality = distance = None
            if other_prompt is not None:
                swapped = runner.run_trial(
                    PolicyInput(policy_input.image_id, other_prompt), seed, None
                )
                swapped_quality, distance = _swap_metrics(
                    evaluator, features, trial, swapped, runner.config.canvas_side
                )
            records.append(
                EvalRecord(
                    policy_input.image_id,
                    seed,
                    trial.end_reason == "finish",
                    trial.end_reason in ("action_budget", "attempt_budget"),
                    own.final_quality or 0.0,
                    swapped_quality,
                    distance,
                    action_signature(trial),
                )  # fmt: skip
            )
    return records


# {
#   責務: [ _swap_metrics: 入れ替えた出力の自分の目標での品質と、元の出力との画素距離を返す ]
# }
def _swap_metrics(
    evaluator: RewardEvaluator, features, own: Trial, swapped: Trial, side: int
) -> tuple[float | None, float | None]:
    from image_drawer.training.reward_features import composite_over_white

    # 描画が成立せず合成が無い試行は、測定できない欠損として扱う。
    if own.final_composite is None or swapped.final_composite is None:
        return None, None
    own_rgb = composite_over_white(own.final_composite, side)
    other_rgb = composite_over_white(swapped.final_composite, side)
    quality = (
        evaluator.final_quality(other_rgb, features) if swapped.end_reason == "finish" else 0.0
    )
    return quality, float(np.abs(own_rgb - other_rgb).mean())


# {
#   責務: [ per_image_mean: 画像ごとの平均(画像単位の対応比較用)。値がNoneの記録は除く ]
# }
def per_image_mean(records: Sequence[EvalRecord], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = getattr(record, field)
        if value is not None:
            grouped[record.image_id].append(float(value))
    return {image: float(np.mean(values)) for image, values in grouped.items()}


# {
#   責務: [ paired_bootstrap_ci: 画像単位の差のブートストラップ95%信頼区間。seedで再現できる ]
#   戻り値: [ (mean, lower, upper) ]
# }
def paired_bootstrap_ci(
    differences: Sequence[float], seed: int = 0, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("差が空です")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    low, high = np.quantile(means, [CONFIDENCE_ALPHA / 2, 1 - CONFIDENCE_ALPHA / 2])
    return float(values.mean()), float(low), float(high)


# {
#   責務: [ _rate: 条件を満たす記録の割合(分母は全記録) ]
# }
def _rate(records: Sequence[EvalRecord], field: str) -> float:
    if not records:
        return 0.0
    return sum(bool(getattr(r, field)) for r in records) / len(records)


# {
#   責務: [ _paired_gain: 画像単位の品質差(a-b)の平均と信頼区間の下限 ]
# }
def _paired_gain(a: Sequence[EvalRecord], b: Sequence[EvalRecord], field: str, seed: int):
    first, second = per_image_mean(a, field), per_image_mean(b, field)
    images = sorted(set(first) & set(second))
    return paired_bootstrap_ci([first[i] - second[i] for i in images], seed)


# {
#   責務: [
#     ClaimGate: M1〜M6を固定閾値で判定する。改善が見えなければnot_learnedを返す
#   ]
# }
class ClaimGate:
    # {
    #   責務: [ evaluate: 学習前・学習後・ランダムの測定表から3値の判定を作る ]
    #   処理: [
    #     1: 画像数・seed数・画像集合の一致を確認(不足はinconclusive)
    #     2: M1(更新)、M2(完走率+0.30、ただし余地(1-学習前)が小さければ余地まで・学習前を下回らない・
    #        予算超過が増えない)、M3(品質+0.05かつ下限>0)、
    #        M4(ランダム超え)、M5(文章の使用)、M6(固定Script化の検査)を判定
    #     3: 全て合格し、整合検査とtarget隔離が通っていればlearned
    #   ]
    #   引数: [ before, after, baseline: EvalRecord列, evidence: M1の根拠,
    #           integrity_ok: Replay一致とtarget隔離の検査が通ったか, bootstrap_seed ]
    # }
    def evaluate(
        self,
        before: Sequence[EvalRecord],
        after: Sequence[EvalRecord],
        baseline: Sequence[EvalRecord],
        evidence: UpdateEvidence,
        integrity_ok: bool,
        bootstrap_seed: int = 0,
    ) -> ClaimReport:
        notes = self._insufficiency_notes(before, after, baseline)
        if notes:
            return ClaimReport("inconclusive", {}, {}, tuple(notes))
        passed: dict[str, bool] = {}
        values: dict[str, float | None] = {}
        passed["M1"] = evidence.param_delta_norm > 0 and evidence.log_prob_changed
        values["param_delta_norm"] = evidence.param_delta_norm
        before_rate = _rate(before, "completed")
        gain = _rate(after, "completed") - before_rate
        increase = _rate(after, "budget_exceeded") - _rate(before, "budget_exceeded")
        # 学習前の完走率が高いと+0.30の余地が無い(合法マスク後は未学習でも完走率1.00)。
        # 余地を超える増分は求めず、学習前を下回らないことを求める
        required = min(M2A_MIN_GAIN, 1.0 - before_rate)
        passed["M2"] = gain >= required and gain >= 0 and increase <= 0
        values.update(
            m2a_completion_gain=gain,
            m2a_required_gain=required,
            m2b_budget_exceeded_increase=increase,
        )
        mean, low, _high = _paired_gain(after, before, "final_quality", bootstrap_seed)
        passed["M3"] = mean >= M3_MIN_GAIN and low > 0
        values.update(m3_mean_gain=mean, m3_ci_lower=low)
        mean4, low4, _ = _paired_gain(after, baseline, "final_quality", bootstrap_seed)
        passed["M4"] = mean4 > 0 and low4 > 0
        values.update(m4_mean_gain=mean4, m4_ci_lower=low4)
        passed["M5"], m5 = self._text_use(after, bootstrap_seed)
        values.update(m5)
        completed = {r.action_signature for r in after if r.completed}
        passed["M6"] = len(completed) >= 2
        values["m6_distinct_completed_signatures"] = float(len(completed))
        suspect = len({r.action_signature for r in after}) <= 1
        extra = []
        if suspect:
            extra.append("scripted_suspect: 全試行のAction列が同一")
        if not integrity_ok:
            extra.append("整合検査またはtarget隔離の検査が通っていない")
        learned = all(passed.values()) and integrity_ok and not suspect
        return ClaimReport("learned" if learned else "not_learned", passed, values, tuple(extra))

    # {
    #   責務: [ _insufficiency_notes: 判定できない理由(画像数・seed数・画像集合の不一致) ]
    # }
    @staticmethod
    def _insufficiency_notes(*tables: Sequence[EvalRecord]) -> list[str]:
        notes = []
        if not tables or any(len(table) == 0 for table in tables):
            return ["測定表が空です"]
        image_sets = [frozenset(r.image_id for r in table) for table in tables]
        if len(set(image_sets)) != 1:
            notes.append("測定表の画像集合が一致しない")
        if len(image_sets[0]) < MIN_IMAGES:
            notes.append(f"画像数が{MIN_IMAGES}未満: {len(image_sets[0])}")
        seed_sets = [frozenset((r.image_id, r.seed) for r in table) for table in tables]
        if len(set(seed_sets)) != 1:
            notes.append("測定表の(画像, seed)の集合が一致しない")
        for index, table in enumerate(tables):
            counts: dict[str, set[int]] = defaultdict(set)
            for record in table:
                counts[record.image_id].add(record.seed)
            if counts and min(len(seeds) for seeds in counts.values()) < MIN_SEEDS:
                notes.append(f"表{index}のseed数が{MIN_SEEDS}未満の画像がある")
        return notes

    # {
    #   責務: [ _text_use: 入替で出力が変わり、自分の文章の品質が入替文章より高い(下限>0)か ]
    # }
    @staticmethod
    def _text_use(after: Sequence[EvalRecord], seed: int) -> tuple[bool, dict[str, float | None]]:
        distances = [r.pixel_distance for r in after if r.pixel_distance is not None]
        own = per_image_mean(after, "final_quality")
        swapped = per_image_mean(after, "swapped_quality")
        images = sorted(set(own) & set(swapped))
        if not distances or not images:
            return False, {"m5_mean_pixel_distance": None, "m5_margin_ci_lower": None}
        mean, low, _ = paired_bootstrap_ci([own[i] - swapped[i] for i in images], seed)
        distance = float(np.mean(distances))
        values = {
            "m5_mean_pixel_distance": distance,
            "m5_margin_mean": mean,
            "m5_margin_ci_lower": low,
        }
        return distance > 0 and mean > 0 and low > 0, values
