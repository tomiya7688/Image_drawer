"""独立した人間評価の資料(盲検パック)の作成と、回答の集計。

条件名は盲検化し、対応表(blind_key.json)は評価者に渡さない。集計は事前登録した計画のハッシュと
一致する場合だけ行う。LLMの判定は人間評価として数えない。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from image_drawer.core import SerializableModel
from image_drawer.dataset.png_encode import encode_png
from image_drawer.generalization.stats import (
    cluster_bootstrap,
    fleiss_kappa,
    holm_adjust,
    sign_test_p,
)

RESPONSE_COLUMNS = (
    "rater_id",
    "session_id",
    "item_id",
    "task",
    "choice",
    "response_time_ms",
    "display_order",
    "is_catch_trial",
    "catch_expected",
    "timestamp",
    "display_resolution",
)
CHOICES = ("left", "right", "same")
CATCH_RATE = 0.10
CATCH_PASS_MIN = 0.90
KAPPA_MIN = 0.2
MIN_RATERS_PER_ITEM = 3
DEFAULT_COMPARISONS = (("after", "before"), ("after", "random_valid"))


# {
#   責務: [
#     HumanEvalPlan: 結果を見る前に凍結する人間評価の計画
#   ]
#   フィールド: [
#     item_ids: 評価に使う画像のID。before/afterの結果を見ずに選ぶ
#     comparisons: 比較する条件の組(第1要素が評価の対象)
#     tasks: T1(prompt一致)、T2(完成度)
#     instruction_sha256: 評価者への指示文のハッシュ
#     rng_seed: 抽出と左右の配置に使う固定seed
#   ]
# }
@dataclass(slots=True)
class HumanEvalPlan(SerializableModel):
    item_ids: list[str]
    instruction_sha256: str
    rng_seed: int
    comparisons: list[list[str]] = field(
        default_factory=lambda: [list(c) for c in DEFAULT_COMPARISONS]
    )
    tasks: list[str] = field(default_factory=lambda: ["T1", "T2"])

    # {
    #   責務: [ digest: 計画全体のSHA-256(パックと集計で一致を確認する) ]
    # }
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# {
#   責務: [
#     select_human_items: 評価用の項目を固定乱数で選ぶ(結果を見て選ばない)
#   ]
#   引数: [
#     item_ids: 評価集合の画像ID, rng_seed: 固定seed, limit: 上限(暫定100)
#   ]
# }
def select_human_items(item_ids: Sequence[str], rng_seed: int, limit: int = 100) -> list[str]:
    ordered = sorted(item_ids)
    random.Random(rng_seed).shuffle(ordered)
    return sorted(ordered[: min(limit, len(ordered))])


# {
#   責務: [
#     strip_png_metadata: PNGを画素だけの固定設定のPNGに作り直し、テキストチャンク等を残さない
#   ]
# }
def strip_png_metadata(data: bytes) -> bytes:
    from PIL import Image

    return encode_png(Image.open(io.BytesIO(data)))


# {
#   責務: [
#     build_human_pack: 盲検化した画像対・指示・回答の雛形を作る
#   ]
#   処理: [
#     1: 項目×比較の組ごとに画像対を作り、左右の位置を固定seedで決める
#     2: 全体の約10%に確認項目(同一画像=期待は「差がない」、白紙と完成画像=期待は完成画像側)を混ぜる
#     3: 対を乱数で並べ替えて番号を付け、pair_<番号>_<L|R>.pngで保存する。メタデータは含めない
#     4: 条件との対応はpack_dirの外のblind_key.jsonにだけ保存する
#   ]
#   引数: [
#     plan: 凍結した計画, images: (image_id, 条件)→PNGのbytes。obviousの確認にはblankが必要
#     prompts: image_id→prompt(T1で文章として見せる), pack_dir: 評価者に渡す出力先
#     key_path: 対応表の保存先(pack_dirの外)
#   ]
#   戻り値: [
#     pair_count: 作った画像対の数
#   ]
# }
def build_human_pack(
    plan: HumanEvalPlan,
    images: Mapping[tuple[str, str], bytes],
    prompts: Mapping[str, str],
    pack_dir: Path,
    key_path: Path,
) -> int:
    rng = random.Random(plan.rng_seed)
    pairs: list[dict] = []
    for item_id in plan.item_ids:
        for first, second in plan.comparisons:
            pairs.append(_pair(item_id, first, second, images, rng, catch=None))
    catch_count = max(2, math.ceil(CATCH_RATE * len(pairs))) if pairs else 0
    for number in range(catch_count):
        item_id = plan.item_ids[number % len(plan.item_ids)]
        kind = "identical" if number % 2 == 0 else "obvious"
        pairs.append(
            _pair(item_id, "after", "blank" if kind == "obvious" else "after", images, rng, kind)
        )
    rng.shuffle(pairs)
    pack_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict] = {}
    for number, pair in enumerate(pairs, start=1):
        pair_id = f"pair_{number:04d}"
        (pack_dir / f"{pair_id}_L.png").write_bytes(strip_png_metadata(pair.pop("left_png")))
        (pack_dir / f"{pair_id}_R.png").write_bytes(strip_png_metadata(pair.pop("right_png")))
        key[pair_id] = pair
    prompts_by_pair = {pid: prompts[entry["image_id"]] for pid, entry in key.items()}
    (pack_dir / "prompts.json").write_text(
        json.dumps(prompts_by_pair, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {"plan_sha256": plan.digest(), "pairs": sorted(key), "tasks": plan.tasks}
    (pack_dir / "pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (pack_dir / "responses_template.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(RESPONSE_COLUMNS)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps({"plan_sha256": plan.digest(), "pairs": key}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(pairs)


# {
#   責務: [
#     _pair: 1つの画像対と、左右の割り当て・確認項目の情報を作る
#   ]
# }
def _pair(item_id, first, second, images, rng, catch) -> dict:
    flipped = rng.random() < 0.5
    left_condition, right_condition = (second, first) if flipped else (first, second)
    entry = {
        "image_id": item_id,
        "left_condition": left_condition,
        "right_condition": right_condition,
        "catch": catch,
        "left_png": images[(item_id, left_condition)],
        "right_png": images[(item_id, right_condition)],
    }
    if catch == "identical":
        entry["expected"] = "same"
    elif catch == "obvious":
        entry["expected"] = "left" if left_condition != "blank" else "right"
    return entry


# {
#   責務: [
#     hash_rater_id: 評価者の呼び名を、氏名・連絡先を残さない匿名IDにする
#   ]
# }
def hash_rater_id(label: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{label}".encode()).hexdigest()[:16]


# {
#   責務: [ _majority: 選択肢の多数決。同数は「差がない」 ]
# }
def _majority(choices: Sequence[str]) -> str:
    counts = Counter(choices).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "same"
    return counts[0][0]


# {
#   責務: [ _outcome: 多数決の選択を、評価対象(第1条件)の勝ち1・引き分け0.5・負け0に直す ]
# }
def _outcome(choice: str, entry: dict, target: str) -> float:
    if choice == "same":
        return 0.5
    chosen = entry["left_condition"] if choice == "left" else entry["right_condition"]
    return 1.0 if chosen == target else 0.0


# {
#   責務: [
#     collect_human_results: 回答を読み、事前登録の計画と一致する場合だけ集計する
#   ]
#   処理: [
#     1: 計画のハッシュが対応表と一致しなければ拒否する
#     2: 確認項目の正答率が90%未満の評価者を除外し、除外数を報告する
#     3: 項目ごとに多数決(同数は差がない)し、after対beforeの勝率をgroup_id単位で区間推定する
#     4: 符号検定のp値をT1とT2でHolm調整し、Fleissのκが0.2未満の課題は判定不能とする
#   ]
#   引数: [
#     plan: 計画, key_path: 対応表, responses_csv: 回答, group_of: image_id→group_id
#   ]
#   戻り値: [
#     result: 課題ごとの集計と、除外した評価者数
#   ]
#   エラー: [
#     ValueError: 計画のハッシュが一致しない、列が仕様と違う
#   ]
# }
def collect_human_results(
    plan: HumanEvalPlan,
    key_path: Path,
    responses_csv: Path,
    group_of: Mapping[str, str],
    *,
    resamples: int = 2000,
    rng_seed: int = 0,
) -> dict:
    key_doc = json.loads(key_path.read_text(encoding="utf-8"))
    if key_doc["plan_sha256"] != plan.digest():
        raise ValueError("human evaluation plan hash does not match the pack")
    key = key_doc["pairs"]
    with responses_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RESPONSE_COLUMNS:
            raise ValueError("responses.csv columns do not match the specification")
        rows = [row for row in reader if row["choice"] in CHOICES]
    excluded = _excluded_raters(rows, key)
    kept = [r for r in rows if r["rater_id"] not in excluded and not key[r["item_id"]]["catch"]]
    tasks = {
        task: _collect_task(task, kept, key, plan, group_of, resamples, rng_seed)
        for task in plan.tasks
    }
    adjusted = holm_adjust([tasks[t]["p_value"] for t in plan.tasks])
    for task, p_holm in zip(plan.tasks, adjusted):
        tasks[task]["p_holm"] = p_holm
    return {
        "plan_sha256": plan.digest(),
        "raters": len({r["rater_id"] for r in rows}),
        "excluded_raters": len(excluded),
        "tasks": tasks,
    }


# {
#   責務: [ _excluded_raters: 確認項目の正答率が90%未満の評価者を返す ]
# }
def _excluded_raters(rows: list[dict], key: dict) -> set[str]:
    totals: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        entry = key[row["item_id"]]
        if entry["catch"]:
            totals[row["rater_id"]].append(row["choice"] == entry["expected"])
    return {
        rater for rater, results in totals.items() if sum(results) / len(results) < CATCH_PASS_MIN
    }


# {
#   責務: [ _collect_task: 1つの課題の勝率・区間・符号検定・κを求める ]
# }
def _collect_task(task, rows, key, plan, group_of, resamples, rng_seed) -> dict:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["task"] == task:
            by_pair[row["item_id"]].append(row)
    target_pairs = [
        pid
        for pid, entry in key.items()
        if pid in by_pair
        and "before" in (entry["left_condition"], entry["right_condition"])
        and "after" in (entry["left_condition"], entry["right_condition"])
    ]
    outcomes, groups = [], []
    for pid in sorted(target_pairs):
        choice = _majority([r["choice"] for r in by_pair[pid]])
        outcomes.append(_outcome(choice, key[pid], "after"))
        groups.append(group_of.get(key[pid]["image_id"], key[pid]["image_id"]))
    wins, losses = outcomes.count(1.0), outcomes.count(0.0)
    interval = (
        cluster_bootstrap(outcomes, groups, resamples=resamples, rng_seed=rng_seed)
        if outcomes
        else (None, None, None)
    )
    kappa = _task_kappa(by_pair)
    return {
        "n_items": len(outcomes),
        "win_rate": interval[0],
        "ci95": [interval[1], interval[2]],
        "p_value": sign_test_p(wins, losses),
        "kappa": kappa,
        "inconclusive": kappa is None or kappa < KAPPA_MIN,
    }


# {
#   責務: [
#     _task_kappa: 項目ごとの評価者数を最小の共通数に揃え(rater_id順)、Fleissのκを求める
#   ]
#   戻り値: [
#     kappa: 評価者3人以上の項目が無い、または計算できなければNone
#   ]
# }
def _task_kappa(by_pair: dict[str, list[dict]]) -> float | None:
    eligible = {pid: rows for pid, rows in by_pair.items() if len(rows) >= MIN_RATERS_PER_ITEM}
    if not eligible:
        return None
    common = min(len(rows) for rows in eligible.values())
    table = []
    for pid in sorted(eligible):
        chosen = sorted(eligible[pid], key=lambda r: r["rater_id"])[:common]
        counts = Counter(r["choice"] for r in chosen)
        table.append([counts[c] for c in CHOICES])
    return fleiss_kappa(table)
