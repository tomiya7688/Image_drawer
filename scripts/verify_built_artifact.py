"""Verify installed entrypoints from a freshly built wheel."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(
            "usage: verify_built_artifact.py <image-drawer> <image-drawer-gui>"
        )

    cli, gui = args

    stdout_result = run(cli, "--input", "artifact-smoke")
    assert json.loads(stdout_result.stdout) == {
        "input": "artifact-smoke",
        "status": "ok",
    }
    assert stdout_result.stderr == ""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "result.json"
        file_result = run(
            cli,
            "--input",
            "artifact-file-smoke",
            "--output",
            str(output_path),
        )
        assert file_result.stdout == ""
        assert file_result.stderr == ""
        assert json.loads(output_path.read_text(encoding="utf-8")) == {
            "input": "artifact-file-smoke",
            "status": "ok",
        }

    gui_result = run(gui, "--check")
    assert json.loads(gui_result.stdout) == {
        "component": "gui",
        "status": "ok",
    }
    assert gui_result.stderr == ""

    print("built artifact smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
