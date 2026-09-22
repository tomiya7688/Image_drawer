"""Exercise the installed M3 wheel, outside the source checkout, end to end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = b"P3\n2 1\n255\n255 0 0   0 255 0\n"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_part_bank_artifact.py <venv-python> <part-bank-cli>")
    # Do not resolve the Python symlink: doing so would bypass the venv.
    python, cli = (os.path.abspath(value) for value in sys.argv[1:])
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        source, bank = cwd / "input", cwd / "bank"
        source.mkdir()
        (source / "fixture.ppm").write_bytes(FIXTURE)

        def run(*args: str, expected_code: int = 0):
            result = subprocess.run(
                args, cwd=cwd, env=env, text=True, capture_output=True, timeout=30,
            )
            if result.returncode != expected_code:
                raise AssertionError(f"exit={result.returncode}: {result.stdout}\n{result.stderr}")
            return result

        args = ["ingest", str(source), "--bank", str(bank),
                "--dataset", "artifact-fixture", "--split", "train"]
        first = run(cli, *args)
        assert first.stderr == ""
        assert json.loads(first.stdout) == dict(
            processed=1, skipped=0, sources_added=1, parts_added=1, duplicates=0, errors=[],
        )
        second = run(python, "-I", "-m", "image_drawer.part_bank", *args)
        assert second.stderr == ""
        assert json.loads(second.stdout) == dict(
            processed=1, skipped=0, sources_added=0, parts_added=0, duplicates=1, errors=[],
        )
        # Import from site-packages and reopen the actual SQLite/crop outputs.
        inspection = run(python, "-I", "-c", """
import hashlib
import sys
from pathlib import Path
from PIL import Image
import image_drawer
from image_drawer.part_bank import SQLitePartRepository
assert Path(image_drawer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
bank, original_path = Path(sys.argv[1]), Path(sys.argv[2])
with SQLitePartRepository(bank / 'metadata.sqlite3') as repo:
    assert len(repo.list_sources()) == len(repo.list_parts()) == 1
    part = repo.list_parts()[0]
    source = repo.get_source(part.source_image_id)
    assert (source.width, source.height) == (2, 1)
    assert source.dataset == 'artifact-fixture'
    assert source.metadata['split'] == 'train'
    assert source.checksum == 'sha256:' + hashlib.sha256(original_path.read_bytes()).hexdigest()
    assert (bank / source.uri).read_bytes() == original_path.read_bytes()
    assert part.bbox == (0, 0, 2, 1)
    assert (part.extraction_method, part.extraction_version) == ('whole_image', '1')
    assert repo.origins(source.id)[0]['uri'] == original_path.as_uri()
    with Image.open(bank / part.crop_uri) as crop:
        assert crop.mode == 'RGBA' and crop.size == (2, 1)
        assert crop.tobytes() == bytes([255, 0, 0, 255, 0, 255, 0, 255])
print('installed Part Bank outputs verified')
""", str(bank), str(source / "fixture.ppm"))
        assert "outputs verified" in inspection.stdout
        (source / "broken.png").write_bytes(b"not an image")
        partial = run(cli, *args, expected_code=1)
        payload = json.loads(partial.stdout)
        assert payload["processed"] == 2 and payload["duplicates"] == 1
        assert len(payload["errors"]) == 1 and payload["errors"][0]["path"] == "broken.png"
        assert partial.stderr == ""
    print("built Part Bank artifact smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
