"""実行結果の比較。項目ごとのgainをgroup_id単位のブートストラップで区間推定する。"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from image_drawer.generalization.runner import BLANK, GenerationResult, RunOutput
from image_drawer.generalization.stats import (
    aggregate_scores,
    cluster_bootstrap,
    effect_size,
    win_rate,
)


# {
#   責務: [
#     assert_same_hardware: 比較する2条件の実行環境が同じことを確認する
#   ]
#   エラー: [ ValueError: hardware_idが異なる(異なる環境のbefore/after比較は行わない) ]
# }
def assert_same_hardware(output: RunOutput, first: str, second: str) -> None:
    if output.hardware_ids.get(first) != output.hardware_ids.get(second):
        raise ValueError(f"hardware_id differs between {first} and {second}")


# {
#   責務: [
#     _itt_scores: 条件の生成ごとの得点を、画像ごとに集める。失敗には画像のblank得点を割り当てる
#   ]
#   戻り値: [ scores: image_id -> [k順の得点] ]
# }
def _itt_scores(output: RunOutput, condition: str, metric: str) -> dict[str, list[float]]:
    by_item: dict[str, list[GenerationResult]] = defaultdict(list)
    for generation in output.generations:
        if generation.condition == condition:
            by_item[generation.image_id].append(generation)
    scores: dict[str, list[float]] = {}
    for image_id, generations in by_item.items():
        generations.sort(key=lambda g: g.k)
        blank = output.blank_metrics[image_id][metric]
        scores[image_id] = [
            blank if g.failure_kind or metric not in g.metrics else g.metrics[metric]
            for g in generations
        ]
    return scores


# {
#   責務: [
#     metric_names: 実行結果に現れた品質指標名(Reward成分と独立指標)
#   ]
# }
def metric_names(output: RunOutput) -> list[str]:
    return sorted({m for blank in output.blank_metrics.values() for m in blank})


# {
#   責務: [
#     compare_conditions: 条件aと条件bの差を指標ごとに求める(ITT)
#   ]
#   処理: [
#     1: 項目iのgain(i)を、推論seedkの平均(a - b)として求める(同じseedで対応づけ)
#     2: gainをgroup_id単位でブートストラップして95%区間を求める
#     3: 勝率(引き分け0.5)と効果量(bの標準偏差で割る)を求める
#   ]
#   引数: [
#     output: 実行結果, a, b: 条件名(aが評価の対象), item_ids: 部分集合(省略は全項目)
#     resamples, rng_seed: ブートストラップの回数と評価seed
#   ]
#   戻り値: [ comparisons: 指標ごとのdelta・ci95・win_rate・effect_size・n_groups ]
# }
def compare_conditions(
    output: RunOutput,
    a: str,
    b: str,
    *,
    item_ids: set[str] | None = None,
    resamples: int = 10000,
    rng_seed: int = 0,
) -> list[dict[str, Any]]:
    assert_same_hardware(output, a, b)
    groups = {g.image_id: g.group_id for g in output.generations}
    comparisons = []
    for metric in metric_names(output):
        first, second = _itt_scores(output, a, metric), _itt_scores(output, b, metric)
        ids = sorted(set(first) & set(second) & (item_ids if item_ids is not None else set(first)))
        if not ids:
            continue
        gains = [mean(x - y for x, y in zip(first[i], second[i])) for i in ids]
        delta, lower, upper = cluster_bootstrap(
            gains, [groups[i] for i in ids], resamples=resamples, rng_seed=rng_seed
        )
        comparisons.append(
            {
                "a": a,
                "b": b,
                "metric": metric,
                "delta": delta,
                "ci95": [lower, upper],
                "win_rate": win_rate(gains),
                "effect_size": effect_size(gains, [mean(second[i]) for i in ids]),
                "n_groups": len({groups[i] for i in ids}),
                "n_items": len(ids),
            }
        )
    return comparisons


# {
#   責務: [
#     summarize_condition: 条件ごとの指標を、ITTと条件付きの両方で返す
#   ]
#   戻り値: [ metrics: 指標名 -> {mean, n, denominator, reporting} を ITT と conditional の2通り ]
# }
def summarize_condition(output: RunOutput, condition: str) -> dict[str, dict[str, Any]]:
    rows = [g for g in output.generations if g.condition == condition]
    summary: dict[str, dict[str, Any]] = {}
    for metric in metric_names(output):
        scores = [
            None if g.failure_kind or metric not in g.metrics else g.metrics[metric] for g in rows
        ]
        blanks = [output.blank_metrics[g.image_id][metric] for g in rows]
        for reporting in ("itt", "conditional"):
            value, denominator = aggregate_scores(scores, blanks, reporting=reporting)
            summary.setdefault(metric, {})[reporting] = {
                "mean": value,
                "denominator": denominator,
                "failures_included": reporting == "itt",
            }
    return summary


# {
#   責務: [
#     compare_all: 評価対象(after)と、before・random_valid・blankとの比較をまとめる
#   ]
#   戻り値: [ comparison: 条件ごとの要約と、比較の一覧 ]
# }
def compare_all(
    output: RunOutput,
    *,
    unseen_prompt_ids: set[str] | None = None,
    resamples: int = 10000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    conditions = sorted({g.condition for g in output.generations} - {BLANK})
    comparisons: list[dict[str, Any]] = []
    for other in ("before", "random_valid"):
        if "after" in conditions and other in conditions:
            subsets = {"test_unseen_image": None}
            if unseen_prompt_ids is not None:
                subsets["test_unseen_prompt"] = unseen_prompt_ids
            for subset, ids in subsets.items():
                for row in compare_conditions(
                    output, "after", other, item_ids=ids, resamples=resamples, rng_seed=rng_seed
                ):
                    comparisons.append({**row, "subset": subset})
    return {
        "eval_set_id": output.eval_set_id,
        "split": output.split,
        "by_condition": {c: summarize_condition(output, c) for c in conditions},
        "comparisons": comparisons,
    }
