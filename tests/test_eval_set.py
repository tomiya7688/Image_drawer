"""汎化評価の評価集合・凍結宣言・test帳簿を確認する(設計: design_evaluation_protocol.md 3節)。"""

import dataclasses
import json
from pathlib import Path

import pytest

from image_drawer.dataset import DatasetManifest, ExtractorInfo
from image_drawer.dataset.models import TrainingImage
from image_drawer.generalization.eval_set import EvalSet, build_eval_set, build_probes
from image_drawer.generalization.freeze import (
    LedgerError,
    authorize_run,
    create_freeze_manifest,
    read_ledger,
)

EXTRACTOR = ExtractorInfo(id="x", version="1", capabilities=["tag"])


# {
#   責務: [
#     make_image: 評価集合の試験用に、分割・グループ・promptを指定したTrainingImageを作る
#   ]
# }
def make_image(image_id: str, split: str, group: str, prompt: str) -> TrainingImage:
    return TrainingImage(
        image_id=image_id,
        image_sha256=f"sha-{image_id}",
        group_id=group,
        split=split,
        image_path=f"images/{image_id}.png",
        width=8,
        height=8,
        normalization_version="n1",
        tags=[],
        caption=prompt,
        training_prompt=prompt,
        tagger=EXTRACTOR,
        captioner=EXTRACTOR,
        prompt_builder_version="p1",
        cache_key=f"k-{image_id}",
    )


# {
#   責務: [
#     write_dataset: 指定した画像でmanifestだけを持つDatasetを書き出す
#   ]
# }
def write_dataset(directory: Path, images: list[TrainingImage]) -> Path:
    manifest = DatasetManifest(
        normalization_version="n1",
        prompt_builder_version="p1",
        tagger=EXTRACTOR,
        captioner=EXTRACTOR,
        split_seed=7,
        images=images,
    )
    manifest.save(directory)
    return directory


BASE_IMAGES = [
    make_image("a1", "train", "g1", "赤い 帽子の 少女"),
    make_image("v1", "validation", "g2", "青い 空"),
    make_image("t1", "test", "g3", "緑の 帽子の 少女"),
    make_image("t2", "test", "g4", "赤い  帽子の 少女"),
    make_image("t3", "test", "g5", "  "),
]


def test_eval_set_snapshot_is_unchanged_when_dataset_grows(tmp_path):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = build_eval_set(dataset, "main_v1", created_at="2026-11-05T00:00:00+00:00")
    path = tmp_path / "eval_set.json"
    eval_set.save(path)
    write_dataset(dataset, [*BASE_IMAGES, make_image("t4", "test", "g9", "新しい 画像")])
    reloaded = EvalSet.load(path)
    assert [i.image_id for i in reloaded.items] == ["t1", "t2"]
    assert reloaded.eval_set_sha256 == eval_set.eval_set_sha256
    assert reloaded.excluded == ["t3"]
    # 新しい画像を加えた評価は、別のeval_set_idで作り直す。
    grown = build_eval_set(dataset, "main_v2", created_at="2026-11-05T00:00:00+00:00")
    assert grown.eval_set_sha256 != eval_set.eval_set_sha256


def test_eval_set_rejects_group_shared_with_train(tmp_path):
    images = [make_image("a1", "train", "g1", "p"), make_image("t1", "test", "g1", "q")]
    dataset = write_dataset(tmp_path / "ds", images)
    with pytest.raises(ValueError, match="shared"):
        build_eval_set(dataset, "bad")


def test_eval_set_hash_changes_when_any_item_changes(tmp_path):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = build_eval_set(dataset, "main_v1", created_at="t")
    path = tmp_path / "eval_set.json"
    eval_set.save(path)
    original = eval_set.eval_set_sha256
    eval_set.items[0].prompt = "別の文"
    assert eval_set.compute_hash() != original
    # 保存後に手で書き換えられた場合は、読込で検出する。
    data = json.loads(path.read_text(encoding="utf-8"))
    data["items"][0]["prompt"] = "改ざん"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        EvalSet.load(path)


def test_unseen_prompt_excludes_prompts_seen_in_train_and_validation(tmp_path):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = build_eval_set(dataset, "main_v1")
    # t2は空白の違いを除くとtrainのpromptと同じなので、未見promptから外れる。
    assert [i.image_id for i in eval_set.unseen_prompt_items()] == ["t1"]


def test_probe_templates_change_only_the_declared_attribute(tmp_path):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = build_eval_set(dataset, "main_v1")
    template = {
        "probe_id": "p1",
        "base_image_id": "t1",
        "attribute": "color",
        "original": "緑",
        "replacement": "青",
    }
    probe = build_probes(eval_set.items, [template])[0]
    assert probe.prompt.replace("青", "緑") == eval_set.items[0].prompt
    with pytest.raises(ValueError, match="exactly once"):
        build_probes(eval_set.items, [{**template, "original": "存在しない"}])  # 現れない語は0回
    with pytest.raises(ValueError, match="exactly once"):
        build_probes(eval_set.items, [{**template, "original": "の"}])  # 複数回現れる語


# {
#   責務: [
#     frozen_setup: 評価集合・checkpoint・分析計画から凍結宣言のファイルを作る
#   ]
# }
def frozen_setup(tmp_path: Path, after_bytes: bytes = b"after"):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = build_eval_set(dataset, "main_v1", created_at="t")
    before = tmp_path / "before.ckpt"
    after = tmp_path / "after.ckpt"
    plan = tmp_path / "plan.json"
    before.write_bytes(b"before")
    after.write_bytes(after_bytes)
    plan.write_text("{}", encoding="utf-8")
    manifest = create_freeze_manifest(
        eval_set,
        before_checkpoint=before,
        after_checkpoint=after,
        analysis_plan=plan,
        git_commit="abc",
        reward_version="r1",
        reward_weights_sha256="w",
        metrics_version="m1",
        metrics_config_sha256="c",
        created_at="t",
    )
    path = tmp_path / f"freeze_{len(after_bytes)}.json"
    manifest.save(path)
    return eval_set, path


def test_test_split_run_requires_freeze_manifest(tmp_path):
    eval_set, _ = frozen_setup(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(LedgerError, match="freeze-manifest"):
        authorize_run("test", eval_set, None, ledger)
    assert not ledger.exists()


def test_second_freeze_on_same_eval_set_marks_retest_in_ledger(tmp_path):
    eval_set, first = frozen_setup(tmp_path)
    _, second = frozen_setup(tmp_path, after_bytes=b"after-changed")
    ledger = tmp_path / "ledger.jsonl"
    assert authorize_run("test", eval_set, first, ledger)["retest"] is False
    # 同じ宣言の再実行は決定性の確認であり、retestにしない。
    again = authorize_run("test", eval_set, first, ledger)
    assert again["retest"] is False and again["rerun"] is True
    # 別の宣言でtestを実行すると、その評価集合の結果はretestになる。
    assert authorize_run("test", eval_set, second, ledger)["retest"] is True


def test_ledger_is_append_only_and_records_manifest_hash(tmp_path):
    eval_set, first = frozen_setup(tmp_path)
    _, second = frozen_setup(tmp_path, after_bytes=b"after-changed")
    ledger = tmp_path / "ledger.jsonl"
    authorize_run("test", eval_set, first, ledger, now="2026-11-05T00:00:00+00:00")
    before_text = ledger.read_text(encoding="utf-8")
    authorize_run("test", eval_set, second, ledger)
    assert ledger.read_text(encoding="utf-8").startswith(before_text)
    entries = read_ledger(ledger)
    assert (
        len(entries) == 2
        and entries[0]["freeze_manifest_sha256"] != entries[1]["freeze_manifest_sha256"]
    )
    assert entries[0]["eval_set_id"] == "main_v1" and entries[0]["timestamp"].startswith("2026")


def test_validation_run_does_not_touch_the_ledger(tmp_path):
    eval_set, _ = frozen_setup(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    assert authorize_run("validation", eval_set, None, ledger) is None
    assert not ledger.exists()
    with pytest.raises(LedgerError):
        authorize_run("train", eval_set, None, ledger)
    assert dataclasses.is_dataclass(eval_set)
