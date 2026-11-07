"""image-drawer-evaluateのコマンド(plan・実行不能の終了コード・報告書)を確認する。"""

import json

from PIL import Image

from test_eval_set import BASE_IMAGES, write_dataset

from image_drawer.generalization.cli import main


def test_plan_creates_a_verifiable_eval_set(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    out = tmp_path / "eval_set.json"
    assert main(["plan", "--dataset", str(dataset), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["eval_set_id"] == "eval_set"


def test_run_on_test_split_without_freeze_manifest_exits_with_2(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    out = tmp_path / "eval_set.json"
    main(["plan", "--dataset", str(dataset), "--out", str(out)])
    code = main(
        ["run", "--dataset", str(dataset), "--eval-set", str(out), "--split", "test",
         "--condition", "after=random", "--out", str(tmp_path / "run")]
    )  # fmt: skip
    assert code == 2 and "freeze-manifest" in capsys.readouterr().err
    assert not (tmp_path / "test_ledger.jsonl").exists()


# {
#   責務: [ run_args: runサブコマンドの引数列を作る ]
# }
def run_args(dataset, eval_set, tmp_path, *conditions):
    args = ["run", "--dataset", str(dataset), "--eval-set", str(eval_set), "--split", "test"]
    for condition in conditions:
        args += ["--condition", condition]
    return [*args, "--out", str(tmp_path / "run")]


def test_run_rejects_invalid_conditions_before_touching_the_ledger(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = tmp_path / "eval_set.json"
    main(["plan", "--dataset", str(dataset), "--out", str(eval_set)])
    for bad in ([], ["=random"], ["after="], ["a=random", "a=random"], ["blank=random"]):
        assert main(run_args(dataset, eval_set, tmp_path, *bad)) == 2
    assert not (tmp_path / "test_ledger.jsonl").exists()


def test_run_rejects_eval_set_items_missing_from_dataset(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = tmp_path / "eval_set.json"
    main(["plan", "--dataset", str(dataset), "--out", str(eval_set)])
    # t2が消えたDatasetでは、一部だけの結果を正常終了として保存しない。
    write_dataset(dataset, [i for i in BASE_IMAGES if i.image_id != "t2"])
    assert main(run_args(dataset, eval_set, tmp_path, "after=random")) == 2
    assert "missing" in capsys.readouterr().err
    assert not (tmp_path / "run" / "run.json").exists()


def test_compare_accepts_run_directory_or_run_json(tmp_path):
    from test_eval_runner import PaintPolicy, make_runner, run_toy

    output = run_toy(
        tmp_path, {"before": make_runner(PaintPolicy()), "after": make_runner(PaintPolicy())}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(output.to_json(), encoding="utf-8")
    for source in (run_dir, run_dir / "run.json"):
        out = tmp_path / f"c{source.name}.json"
        assert main(["compare", "--run", str(source), "--out", str(out), "--resamples", "50"]) == 0
        assert json.loads(out.read_text(encoding="utf-8"))["comparisons"]


def test_human_pack_without_images_exits_with_2(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    eval_set = tmp_path / "eval_set.json"
    main(["plan", "--dataset", str(dataset), "--out", str(eval_set)])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"item_ids": ["t1"], "instruction_sha256": "x", "rng_seed": 1}), encoding="utf-8"
    )
    code = main(["human-pack", "--run", str(tmp_path / "run"), "--plan", str(plan),
                 "--eval-set", str(eval_set), "--out", str(tmp_path / "pack")])  # fmt: skip
    assert code == 2 and "images" in capsys.readouterr().err


def test_memorization_tau_is_computed_from_validation_and_saved(tmp_path, capsys):
    dataset = write_dataset(tmp_path / "ds", BASE_IMAGES)
    (dataset / "images").mkdir()
    for index, image in enumerate(BASE_IMAGES):  # 目標画像の実体(濃さを変えて距離を非ゼロにする)
        Image.new("RGB", (8, 8), (index * 40,) * 3).save(dataset / image.image_path)
    out = tmp_path / "tau.json"
    assert main(["memorization-tau", "--dataset", str(dataset), "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["source"] == "validation" and saved["n_train"] == 1 and saved["n_validation"] == 1
    assert 0.0 <= saved["tau"] <= 2.0
    empty = write_dataset(tmp_path / "ds2", [i for i in BASE_IMAGES if i.split != "validation"])
    assert (
        main(["memorization-tau", "--dataset", str(empty), "--out", str(tmp_path / "x.json")]) == 2
    )
    assert not (tmp_path / "x.json").exists()
