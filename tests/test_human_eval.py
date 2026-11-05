"""独立した人間評価の盲検パックと集計を確認する(設計 8節)。"""

import csv
import io
import json
import re
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from image_drawer.generalization.human import (
    RESPONSE_COLUMNS,
    HumanEvalPlan,
    build_human_pack,
    collect_human_results,
    hash_rater_id,
)

COLORS = {
    "after": (200, 30, 30),
    "before": (30, 200, 30),
    "random_valid": (30, 30, 200),
    "blank": (255, 255, 255),
}
ITEMS = ["i1", "i2", "i3", "i4", "i5", "i6"]


# {
#   責務: [ png_with_text: 条件名を含むテキストチャンク付きのPNGを作る(メタデータが残らないことの確認用) ]
# }
def png_with_text(color, label: str) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("condition", label)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


IMAGES = {(i, c): png_with_text(color, c) for i in ITEMS for c, color in COLORS.items()}
PROMPTS = {i: f"絵{i}" for i in ITEMS}


# {
#   責務: [ make_pack: 計画から盲検パックと対応表を作る ]
# }
def make_pack(tmp_path: Path, seed: int = 1):
    plan = HumanEvalPlan(item_ids=ITEMS, instruction_sha256="x", rng_seed=seed)
    pack, key = tmp_path / f"pack{seed}", tmp_path / f"key{seed}.json"
    count = build_human_pack(plan, IMAGES, PROMPTS, pack, key)
    return plan, pack, key, count


def test_human_pack_hides_condition_names_and_metadata(tmp_path):
    plan, pack, key, count = make_pack(tmp_path)
    names = sorted(p.name for p in pack.iterdir() if p.suffix == ".png")
    assert len(names) == 2 * count and all(re.fullmatch(r"pair_\d{4}_[LR]\.png", n) for n in names)
    for path in pack.iterdir():
        data = path.read_bytes()
        assert b"tEXt" not in data and b"condition" not in data
        if path.suffix != ".png":
            assert not any(c.encode() in data for c in COLORS)
    # 対応表は評価者に渡すパックの外に置く。
    assert key.parent != pack and not (pack / key.name).exists()


def test_human_pack_randomizes_left_right_by_seed(tmp_path):
    _, _, key1, _ = make_pack(tmp_path, seed=1)
    _, _, key1_again, _ = make_pack(tmp_path / "again", seed=1)
    _, _, key2, _ = make_pack(tmp_path, seed=2)
    pairs1 = json.loads(key1.read_text(encoding="utf-8"))["pairs"]
    assert pairs1 == json.loads(key1_again.read_text(encoding="utf-8"))["pairs"]
    assert pairs1 != json.loads(key2.read_text(encoding="utf-8"))["pairs"]
    sides = {p["left_condition"] for p in pairs1.values() if not p["catch"]}
    assert len(sides) > 1  # 評価対象が常に同じ側にならない


def test_human_pack_includes_catch_trials_identical_and_obvious(tmp_path):
    _, pack, key, count = make_pack(tmp_path)
    pairs = json.loads(key.read_text(encoding="utf-8"))["pairs"]
    kinds = [p["catch"] for p in pairs.values() if p["catch"]]
    assert "identical" in kinds and "obvious" in kinds
    assert len(kinds) >= count * 0.08
    for pair_id, entry in pairs.items():
        if entry["catch"] == "identical":
            assert (pack / f"{pair_id}_L.png").read_bytes() == (
                pack / f"{pair_id}_R.png"
            ).read_bytes()
            assert entry["expected"] == "same"
        if entry["catch"] == "obvious":
            assert "blank" in (entry["left_condition"], entry["right_condition"])


# {
#   責務: [
#     write_responses: 評価者ごとの方針で回答CSVを作る。after側を選ぶ評価者と、確認項目を誤る評価者を作れる
#   ]
# }
def write_responses(path: Path, key: Path, raters: dict[str, str]) -> None:
    pairs = json.loads(key.read_text(encoding="utf-8"))["pairs"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(RESPONSE_COLUMNS)
        for rater, behavior in raters.items():
            for pair_id, entry in pairs.items():
                for task in ("T1", "T2"):
                    writer.writerow(
                        [
                            rater,
                            "s1",
                            pair_id,
                            task,
                            choose(behavior, entry, rater),
                            1000,
                            0,
                            bool(entry["catch"]),
                            "",
                            "t",
                            "1x1",
                        ]
                    )


# {
#   責務: [ choose: 評価者の方針(good=正しく、bad_catch=確認項目を誤る)に従う選択 ]
# }
def choose(behavior: str, entry: dict, rater: str) -> str:
    if entry["catch"]:
        wrong = {"left": "right", "right": "left", "same": "left"}[entry["expected"]]
        return wrong if behavior == "bad_catch" else entry["expected"]
    after_left = entry["left_condition"] == "after"
    return "left" if after_left else "right"


def test_human_collect_excludes_raters_below_catch_threshold_and_reports_count(tmp_path):
    plan, _, key, _ = make_pack(tmp_path)
    responses = tmp_path / "r.csv"
    write_responses(responses, key, {"a": "good", "b": "good", "c": "good", "bad": "bad_catch"})
    groups = {i: f"g{i}" for i in ITEMS}
    result = collect_human_results(plan, key, responses, groups, resamples=200)
    assert result["excluded_raters"] == 1 and result["raters"] == 4
    t1 = result["tasks"]["T1"]
    assert t1["win_rate"] == pytest.approx(1.0) and t1["ci95"][0] > 0.5
    assert t1["kappa"] == pytest.approx(1.0) and not t1["inconclusive"]


def test_human_collect_marks_task_inconclusive_when_kappa_is_below_threshold(tmp_path):
    plan, _, key, _ = make_pack(tmp_path)
    responses = tmp_path / "r.csv"
    # 3人が別々の基準で答えると一致が低く、判定不能になる。
    pairs = json.loads(key.read_text(encoding="utf-8"))["pairs"]
    with responses.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(RESPONSE_COLUMNS)
        for pair_id, entry in pairs.items():
            for index, rater in enumerate(("a", "b", "c")):
                if entry["catch"]:
                    choice = entry["expected"]
                else:
                    choice = ("left", "right", "same")[index]
                writer.writerow(
                    [rater, "s1", pair_id, "T1", choice, 1, 0, bool(entry["catch"]), "", "t", "1x1"]
                )
    result = collect_human_results(plan, key, responses, {}, resamples=100)
    assert result["tasks"]["T1"]["inconclusive"] is True
    assert result["tasks"]["T2"]["kappa"] is None and result["tasks"]["T2"]["inconclusive"]


def test_human_plan_hash_must_match_before_collecting(tmp_path):
    plan, _, key, _ = make_pack(tmp_path)
    responses = tmp_path / "r.csv"
    write_responses(responses, key, {"a": "good"})
    changed = HumanEvalPlan(item_ids=ITEMS[:-1], instruction_sha256="x", rng_seed=1)
    with pytest.raises(ValueError, match="plan hash"):
        collect_human_results(changed, key, responses, {})


def test_responses_csv_has_no_personal_fields(tmp_path):
    _, pack, _, _ = make_pack(tmp_path)
    header = next(csv.reader((pack / "responses_template.csv").open(encoding="utf-8")))
    assert tuple(header) == RESPONSE_COLUMNS
    assert not any(word in col for col in header for word in ("name", "email", "contact", "mail"))
    anonymous = hash_rater_id("山田", "salt")
    assert "山田" not in anonymous and len(anonymous) == 16
