import json

from image_drawer.gui import main


def test_gui_headless_check(capsys):
    assert main(["--check"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"component": "gui", "status": "ok"}
    assert captured.err == ""
