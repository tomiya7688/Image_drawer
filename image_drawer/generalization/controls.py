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
