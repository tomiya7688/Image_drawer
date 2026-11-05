"""汎化評価の統計。外部依存なし。group_id単位のブートストラップ、勝率、Holm法、Fleissのκ、ITT集計。"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from statistics import mean, pstdev


# {
#   責務: [
#     inference_seed: 画像と通し番号から推論seedを決める。条件に依存せず、before/afterで同じ値になる
#   ]
#   処理: [
#     SHA-256("{eval_set_id}:{image_id}:{k}")の先頭32ビットを返す
#   ]
#   引数: [
#     eval_set_id: 評価集合のID, image_id: 画像のID, k: 0から始まる通し番号
#   ]
#   戻り値: [
#     seed: 0以上2^32未満の整数
#   ]
# }
def inference_seed(eval_set_id: str, image_id: str, k: int) -> int:
    digest = hashlib.sha256(f"{eval_set_id}:{image_id}:{k}".encode()).hexdigest()
    return int(digest[:8], 16)


# {
#   責務: [
#     cluster_bootstrap: group_id単位で復元抽出し、平均の信頼区間を求める
#   ]
#   処理: [
#     1: 値をgroup_idごとの(合計, 件数)にまとめる。近重複の画像は独立でないため、抽出の単位はgroup_id
#     2: group_idを復元抽出して同数の標本を作り、標本ごとに全itemの平均を求める
#     3: 標本平均の下側・上側のパーセンタイルを区間とする。乱数は固定seedで決める
#   ]
#   引数: [
#     values: itemごとの値(例: gain)
#     group_ids: itemごとのgroup_id。valuesと同じ長さ
#     resamples: 抽出回数
#     rng_seed: 評価seed
#     ci: 信頼水準(0.95など)
#   ]
#   戻り値: [
#     (mean, lower, upper): 観測平均と区間
#   ]
#   エラー: [
#     ValueError: 値が空、長さの不一致、信頼水準が範囲外
#   ]
# }
def cluster_bootstrap(
    values: Sequence[float],
    group_ids: Sequence[str],
    *,
    resamples: int = 10000,
    rng_seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    if not values or len(values) != len(group_ids):
        raise ValueError("values and group_ids must be non-empty and of equal length")
    if not 0.0 < ci < 1.0 or resamples < 1:
        raise ValueError("ci must be in (0, 1) and resamples must be positive")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for value, group in zip(values, group_ids):
        sums[group] = sums.get(group, 0.0) + value
        counts[group] = counts.get(group, 0) + 1
    keys = sorted(sums)
    group_sums = [sums[k] for k in keys]
    group_counts = [counts[k] for k in keys]
    rng = random.Random(rng_seed)
    size = len(keys)
    estimates: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        estimates.append(sum(group_sums[p] for p in picks) / sum(group_counts[p] for p in picks))
    estimates.sort()
    tail = (1.0 - ci) / 2.0
    lower = estimates[min(resamples - 1, int(math.floor(tail * resamples)))]
    upper = estimates[min(resamples - 1, int(math.ceil((1.0 - tail) * resamples)) - 1)]
    return mean(values), lower, upper


# {
#   責務: [
#     win_rate: 改善の勝率。引き分けは0.5勝として数える
#   ]
# }
def win_rate(gains: Sequence[float]) -> float:
    if not gains:
        raise ValueError("gains must not be empty")
    return sum(1.0 if g > 0 else 0.5 if g == 0 else 0.0 for g in gains) / len(gains)


# {
#   責務: [
#     effect_size: gainの平均を、before側の標準偏差で割った効果量
#   ]
#   戻り値: [
#     size: 効果量。beforeの標準偏差が0のときはNone(定義できない)
#   ]
# }
def effect_size(gains: Sequence[float], before_values: Sequence[float]) -> float | None:
    spread = pstdev(before_values) if len(before_values) > 1 else 0.0
    return None if spread == 0.0 else mean(gains) / spread


# {
#   責務: [
#     holm_adjust: 複数の検定のp値をHolm法で調整する
#   ]
#   処理: [
#     p値を昇順に並べ、(m-順位)の係数を掛けて単調に増やし、1で打ち切る。入力順に戻す
#   ]
# }
def holm_adjust(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


# {
#   責務: [
#     sign_test_p: 勝ち数と負け数に対する両側の正確二項検定(p=0.5)のp値
#   ]
#   処理: [
#     引き分けは除いた件数で、観測より極端でない確率を足し引きして両側のp値を求める
#   ]
# }
def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    low = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(low + 1)) / 2**n
    return min(1.0, 2.0 * tail)


# {
#   責務: [
#     fleiss_kappa: 評価者間の一致をFleissのκで測る
#   ]
#   引数: [
#     table: 行が対象、列が選択肢。各セルは、その選択肢を選んだ評価者の数。行ごとの合計は同じ
#   ]
#   戻り値: [
#     kappa: κ。偶然の一致だけのとき0、全員一致で1。偶然の一致が1(選択肢が1つ)のときはNone
#   ]
#   エラー: [
#     ValueError: 表が空、または行ごとの評価者数が異なる
#   ]
# }
def fleiss_kappa(table: Sequence[Sequence[int]]) -> float | None:
    if not table:
        raise ValueError("table must not be empty")
    raters = sum(table[0])
    if raters < 2 or any(sum(row) != raters for row in table):
        raise ValueError("every subject needs the same number (>=2) of ratings")
    n = len(table)
    categories = len(table[0])
    category_share = [sum(row[j] for row in table) / (n * raters) for j in range(categories)]
    agreement = [(sum(c * c for c in row) - raters) / (raters * (raters - 1)) for row in table]
    observed = sum(agreement) / n
    chance = sum(p * p for p in category_share)
    if chance == 1.0:
        return None
    return (observed - chance) / (1.0 - chance)


# {
#   責務: [
#     aggregate_scores: 失敗を分母に残す集計(ITT)と、完成だけの集計(条件付き)を返す
#   ]
#   処理: [
#     1: reporting="itt"は、失敗(None)の生成に、同じ生成のblank得点を割り当てて全件を平均する
#     2: reporting="conditional"は、失敗を除いた平均。分母は完成数
#   ]
#   引数: [
#     scores: 生成ごとの得点。失敗はNone
#     blank_scores: 生成ごとのblank(白紙)の得点。scoresと同じ長さ
#     reporting: "itt" または "conditional"
#   ]
#   戻り値: [
#     (mean, denominator): 平均と分母。条件付きで完成が0件ならmeanはNone
#   ]
# }
def aggregate_scores(
    scores: Sequence[float | None],
    blank_scores: Sequence[float],
    *,
    reporting: str = "itt",
) -> tuple[float | None, int]:
    if len(scores) != len(blank_scores):
        raise ValueError("scores and blank_scores must have the same length")
    if reporting == "itt":
        filled = [b if s is None else s for s, b in zip(scores, blank_scores)]
    elif reporting == "conditional":
        filled = [s for s in scores if s is not None]
    else:
        raise ValueError("reporting must be 'itt' or 'conditional'")
    return (mean(filled) if filled else None), len(filled)
