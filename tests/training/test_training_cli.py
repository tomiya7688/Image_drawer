"""学習CLI(python -m image_drawer.training)の配線テスト。改善は主張しない(設計書8.4節)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from training_helpers import make_dataset  # noqa: E402

from image_drawer.training.__main__ import main  # noqa: E402

CONFIG = {
    "canvas_side": 16, "summary_side": 4, "max_actions": 24, "batch_images": 2,
    "seeds_per_image": 2, "iterations": 2, "checkpoint_every": 1, "master_seed": 5,
    "text_encoder_id": "hashed-ngram-v1",
}  # fmt: skip


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("cli_data"), count=100, diverse=True)


def write_config(tmp_path: Path) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CONFIG), encoding="utf-8")
    return str(path)


def test_cli_train_smoke_writes_records_without_claim(
    dataset_dir: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "out"
    code = main(["train", str(dataset_dir), "--config", write_config(tmp_path),
                 "--output-dir", str(out)])  # fmt: skip
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["claim"] is None and summary["abort_reason"] is None
    record = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert record["claim"] is None and record["counts"]["updates_applied"] >= 0
    assert (out / "metrics.jsonl").exists() and (out / "checkpoints" / "iter_2.json").exists()


def test_cli_train_overrides_and_resume(dataset_dir: Path, tmp_path: Path) -> None:
    config = write_config(tmp_path)
    out = tmp_path / "resume"
    base = ["train", str(dataset_dir), "--config", config, "--output-dir", str(out)]
    assert main([*base, "--stop-after", "1"]) == 0
    sidecar = out / "checkpoints" / "iter_1.json"
    assert sidecar.exists()
    assert main([*base, "--resume", str(sidecar)]) == 0
    record = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert record["policy_after"]["path"].endswith("iter_2.npz")


def test_cli_evaluate_saves_verdict_as_is(dataset_dir: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "eval"
    assert main(["train", str(dataset_dir), "--config", write_config(tmp_path),
                 "--output-dir", str(out)]) == 0  # fmt: skip
    capsys.readouterr()
    code = main(["evaluate", str(out), "--dataset", str(dataset_dir), "--images", "2",
                 "--seeds", "2"])  # fmt: skip
    assert code == 0
    claim = json.loads(capsys.readouterr().out)
    assert claim["verdict"] in {"learned", "not_learned", "inconclusive"}
    # 画像2枚・seed2はMIN未満のため、改善を主張せずinconclusiveになる
    assert claim["verdict"] == "inconclusive"
    stored = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    assert stored["claim"]["verdict"] == "inconclusive"
    assert (out / "claim.json").exists() and (out / "evaluation.jsonl").exists()


def test_cli_returns_2_for_unusable_inputs(dataset_dir: Path, tmp_path: Path, capsys) -> None:
    assert main(["train", str(tmp_path / "missing"), "--config", write_config(tmp_path),
                 "--output-dir", str(tmp_path / "o")]) == 2  # fmt: skip
    assert json.loads(capsys.readouterr().err)["error"]
    bad = tmp_path / "bad.json"
    bad.write_text('{"unknown_key": 1}', encoding="utf-8")
    assert main(["train", str(dataset_dir), "--config", str(bad)]) == 2
    assert main(["evaluate", str(tmp_path / "none"), "--dataset", str(dataset_dir)]) == 2


def test_cli_evaluate_rejects_too_few_images_or_seeds(dataset_dir: Path, tmp_path: Path) -> None:
    for flag in ("--images", "--seeds"):
        for value in ("0", "-1"):
            argv = ["evaluate", str(tmp_path), "--dataset", str(dataset_dir), flag, value]
            assert main(argv) == 2
