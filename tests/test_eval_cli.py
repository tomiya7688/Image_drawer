"""image-drawer-evaluateのコマンド(plan・実行不能の終了コード・報告書)を確認する。"""

import json

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
         "--out", str(tmp_path / "run")]
    )  # fmt: skip
    assert code == 2 and "freeze-manifest" in capsys.readouterr().err
    assert not (tmp_path / "test_ledger.jsonl").exists()
