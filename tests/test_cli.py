import json

from image_drawer.cli import main


def test_cli_stdout_contract(capsys):
    assert main(["--input", "hello"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"input": "hello", "status": "ok"}
    assert captured.err == ""


def test_cli_file_contract(tmp_path, capsys):
    output_path = tmp_path / "nested" / "result.json"

    assert main(["--input", "file-value", "--output", str(output_path)]) == 0

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "input": "file-value",
        "status": "ok",
    }
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
