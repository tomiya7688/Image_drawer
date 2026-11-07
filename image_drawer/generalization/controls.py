"""対照指標(prompt_shuffle_delta・probe_hit_rate)の計算。実行結果から報告書の入力を作る。"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from image_drawer.generalization.stats import cluster_bootstrap

# 構成的なprobe(設計書6.6節)。色名のpromptと、固定256色(6x6x6の立方体: 番号=36r+6g+b)での期待色。
# 個数・位置は後続のprobeで追加する
PROBES: tuple[tuple[str, int], ...] = (
    ("赤い", 180),  # (255, 0, 0)
    ("青い", 5),  # (0, 0, 255)
    ("緑の", 30),  # (0, 255, 0)
    ("黄色い", 210),  # (255, 255, 0)
)


# {
#   責務: [
#     probe_hit_rate_control: 条件ごとのprobe命中率(命中数 ÷ 全生成数)を作る。失敗は命中に数えず分母に残す
#   ]
#   引数: [ output: RunOutput(probesを持つもの), before, after: 比較する条件名 ]
#   戻り値: [ {"before": 率, "after": 率, "n": 条件あたりの生成数}。どちらかの条件の記録が無ければNone ]
# }
def probe_hit_rate_control(
    output: Any, before: str = "before", after: str = "after"
) -> dict[str, Any] | None:
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for condition in (before, after):
        rows = [p for p in getattr(output, "probes", []) if p.condition == condition]
        if not rows:
            return None
        counts[condition] = len(rows)
        rates[condition] = sum(1 for p in rows if p.hit) / len(rows)
    if counts[before] != counts[after]:
        return None
    return {"before": rates[before], "after": rates[after], "n": counts[after]}


# {
#   責務: [
#     nearest_train_hit_rate_control: 条件ごとに、trainの目標画像へτ未満まで近づいた生成の割合(記憶・コピーの疑い)を作る
#   ]
#   処理: [
#     命中は nearest_train_distance < τ。失敗した生成は画像が無く命中に数えず、分母に残す。
#     τはvalidationで決めて凍結した値で、実行時にRunOutput.memorizationへ保存される
#   ]
#   引数: [ output: RunOutput, before, after: 比較する条件名 ]
#   戻り値: [ {"before": 率, "after": 率, "tau": τ, "n": 条件あたりの生成数}。確認していない・片方の記録が無い・件数が違えばNone ]
# }
def nearest_train_hit_rate_control(
    output: Any, before: str = "before", after: str = "after"
) -> dict[str, Any] | None:
    memorization = getattr(output, "memorization", None)
    if not memorization:
        return None
    tau = float(memorization["tau"])
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for condition in (before, after):
        rows = [g for g in output.generations if g.condition == condition]
        if not rows:
            return None
        counts[condition] = len(rows)
        hits = sum(
            1
            for g in rows
            if g.nearest_train_distance is not None and g.nearest_train_distance < tau
        )
        rates[condition] = hits / len(rows)
    if counts[before] != counts[after]:
        return None
    return {"before": rates[before], "after": rates[after], "tau": tau, "n": counts[after]}


# {
#   責務: [
#     generalization_gap_control: gain_train − gain_test(gainはafter − beforeの画像ごとの平均の平均)を作る
#   ]
#   処理: [
#     失敗には画像のblank得点を割り当てる(ITT)。同じ指標で、trainの部分集合の実行とtestの実行を比べる。
#     大きいほどtrainだけ改善している(記憶・過学習の疑い)
#   ]
#   引数: [ test_output, train_output: RunOutput(beforeとafterを持つ), metric: 指標名, before, after ]
#   戻り値: [ {"gap","ci95","gain_train","gain_test","metric","n_train","n_test"}。ci95はtrainとtestをgroup単位で独立に復元抽出した差の95%区間。片方に条件の記録が無ければNone ]
# }
def generalization_gap_control(
    test_output: Any,
    train_output: Any,
    metric: str,
    before: str = "before",
    after: str = "after",
    *,
    resamples: int = 10000,
    rng_seed: int = 0,
) -> dict[str, Any] | None:
    from image_drawer.generalization.compare import _itt_scores  # 循環importを避ける

    def gains(output: Any) -> tuple[list[float], list[str]] | None:
        first, second = _itt_scores(output, before, metric), _itt_scores(output, after, metric)
        group_of = {g.image_id: g.group_id for g in output.generations}
        values: list[float] = []
        groups: list[str] = []
        for image_id in sorted(set(first) & set(second)):
            ks = sorted(set(first[image_id]) & set(second[image_id]))
            if ks:
                values.append(sum(second[image_id][k] - first[image_id][k] for k in ks) / len(ks))
                groups.append(group_of.get(image_id, image_id))
        return (values, groups) if values else None

    test, train = gains(test_output), gains(train_output)
    if test is None or train is None:
        return None
    gain_test, gain_train = sum(test[0]) / len(test[0]), sum(train[0]) / len(train[0])
    # trainとtestは別の画像集合なので、それぞれをgroup単位で独立に復元抽出して差の区間を求める
    rng = random.Random(rng_seed)

    def resampled_mean(values: list[float], groups: list[str]) -> float:
        by_group: dict[str, list[float]] = {}
        for value, group in zip(values, groups):
            by_group.setdefault(group, []).append(value)
        keys = sorted(by_group)
        picks = [rng.choice(keys) for _ in keys]
        flat = [v for key in picks for v in by_group[key]]
        return sum(flat) / len(flat)

    estimates = sorted(resampled_mean(*train) - resampled_mean(*test) for _ in range(resamples))
    low = estimates[min(resamples - 1, int(0.025 * resamples))]
    high = estimates[min(resamples - 1, int(0.975 * resamples))]
    return {
        "gap": gain_train - gain_test,
        "ci95": [low, high],
        "gain_train": gain_train,
        "gain_test": gain_test,
        "metric": metric,
        "n_train": len(train[0]),
        "n_test": len(test[0]),
    }


# {
#   責務: [
#     shuffle_targets: 各画像の入れ替え先の画像を、固定乱数で決める(自分自身にならない巡回)
#   ]
#   処理: [ 画像IDを並べて固定乱数で1〜n-1の距離を選び、その分だけずらした画像を入れ替え先にする ]
#   引数: [ image_ids: 入れ替える画像(2件以上), key: 乱数の種になる文字列(評価集合ID) ]
#   戻り値: [ targets: image_id -> 入れ替え先のimage_id ]
#   エラー: [ ValueError: 画像が2件未満(入れ替え先が作れない) ]
# }
def shuffle_targets(image_ids: Sequence[str], key: str) -> dict[str, str]:
    ordered = sorted(set(image_ids))
    if len(ordered) < 2:
        raise ValueError("prompt shuffle needs at least two distinct images")
    shift = random.Random(f"{key}:prompt-shuffle").randrange(1, len(ordered))
    return {item: ordered[(index + shift) % len(ordered)] for index, item in enumerate(ordered)}


# {
#   責務: [
#     prompt_shuffle_control: 正しいpromptと入れ替えたpromptの得点差を、group単位のブートストラップで集計する
#   ]
#   処理: [
#     1: (画像, 推論seed)ごとに score(正しいprompt) - score(入れ替えたprompt) を求める。失敗は白紙の得点
#     2: 近重複の画像は独立でないため、group_id単位で復元抽出して平均の95%区間を求める
#   ]
#   引数: [ output: RunOutput(shuffledを持つもの), condition: 条件名, metric: 指標名, resamples, rng_seed ]
#   戻り値: [ {"delta": 平均, "ci95": [下限, 上限], "metric": 指標名, "n": 件数}。入れ替え生成が無ければNone ]
# }
def prompt_shuffle_control(
    output: Any,
    condition: str,
    metric: str,
    *,
    resamples: int = 10000,
    rng_seed: int = 0,
) -> dict[str, Any] | None:
    shuffled = {
        (g.image_id, g.k): g
        for g in getattr(output, "shuffled", [])
        if g.condition == condition
    }  # fmt: skip
    if not shuffled:
        return None
    blank: Mapping[str, Mapping[str, float]] = output.blank_metrics
    deltas: list[float] = []
    groups: list[str] = []
    for own in output.generations:
        other = shuffled.get((own.image_id, own.k))
        if own.condition != condition or other is None:
            continue
        base = blank[own.image_id][metric]
        own_score = base if own.failure_kind or metric not in own.metrics else own.metrics[metric]
        other_score = (
            base if other.failure_kind or metric not in other.metrics else other.metrics[metric]
        )
        deltas.append(own_score - other_score)
        groups.append(own.group_id)
    if not deltas:
        return None
    mean, low, high = cluster_bootstrap(deltas, groups, resamples=resamples, rng_seed=rng_seed)
    return {"delta": mean, "ci95": [low, high], "metric": metric, "n": len(deltas)}
