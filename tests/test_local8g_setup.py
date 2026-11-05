"""8GB向けGGUFダウンロード手順のファイル分離をネットワークなしで試す。"""

from __future__ import annotations

from pathlib import Path

from scripts import setup_local_8g_models as setup


def test_model_download_plan_includes_vision_projector_for_each_model():
    models = setup.MODEL_FILES
    assert [entry.model_id for entry in models] == [
        "qwen3vl4b-q4",
        "joycaption-beta-one-q3",
        "qwen3vl4b-uncensored-q4",
    ]
    for model in models:
        assert len(model.files) == 2
        assert model.files[0].filename.endswith(".gguf")
        assert model.files[1].dest_name.startswith("mmproj")
        assert model.files[1].filename.endswith(".gguf")
    assert models[1].files[1].repo_id.startswith("concedo/")


def test_downloads_only_required_quant_and_mmproj_without_overwrite(tmp_path):
    requested: list[tuple[str, str]] = []

    def fake_fetch(*, repo_id: str, filename: str, local_dir: Path) -> str:
        requested.append((repo_id, filename))
        src = local_dir / filename
        src.write_bytes(b"stand-in-gguf")
        return str(src)

    assert len(setup.missing_model_files(tmp_path)) == 6
    downloaded = setup.download_model_files(tmp_path, fake_fetch)
    assert len(downloaded) == 6
    assert not setup.missing_model_files(tmp_path)
    assert all(path.name.endswith(".gguf") for path in downloaded)
    assert all(path.stat().st_size > 0 for path in downloaded)
    assert len(requested) == 6
    assert setup.download_model_files(tmp_path, fake_fetch) == []
    assert len(requested) == 6


def test_check_fails_without_files_and_dry_run_needs_no_hub(tmp_path, capsys):
    assert setup.main(["--models-dir", str(tmp_path), "--dry-run"]) == 0
    assert "joycaption-beta-one-q3" in capsys.readouterr().out
    assert setup.main(["--models-dir", str(tmp_path), "--check"]) == 1
    assert "不足" in capsys.readouterr().out


def test_check_succeeds_with_projector_for_all_three_models(tmp_path, capsys):
    for model in setup.MODEL_FILES:
        folder = tmp_path / model.model_id
        folder.mkdir()
        for part in model.files:
            (folder / part.dest_name).write_bytes(b"mock gguf")
    assert setup.main(["--models-dir", str(tmp_path), "--check"]) == 0
    output = capsys.readouterr().out
    assert "--models-max 1" in output
    assert "--models-memory-margin 2048" in output
    assert "--fit on --fit-target 2048" in output
    assert "--gpu-layers auto" in output
    assert "--no-mmproj-offload" in output
    assert "--port 8080" in output
