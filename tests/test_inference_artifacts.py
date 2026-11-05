"""S4-2: 生成結果の保存（原子的な書込み・ハッシュ・Draftの名前・PNG）。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from inference_helpers import SIDE, complete_actions, scripted_registry
from stage_helpers import model

from image_drawer.drawing.episode_io import loads_episode
from image_drawer.inference import DrawRequest
from image_drawer.inference.artifacts import (
    KNOWN_FILES,
    OutputConflictError,
    episode_metadata,
    write_atomic,
    write_outcome,
)
from image_drawer.inference.png import PngError, decode_png_rgba, encode_png_rgba
from image_drawer.inference.session import generate_episode


def draw(registry=None, **kwargs):
    base = {"prompt": "猫", "policy": "scripted", "width": SIDE, "height": SIDE}
    return generate_episode(DrawRequest(**{**base, **kwargs}), registry or scripted_registry())


def test_episode_file_roundtrip_matches_sha256_in_metadata(tmp_path: Path) -> None:
    outcome = draw()
    written = write_outcome(outcome, tmp_path / "out")
    data = written.episode_path.read_bytes()
    metadata = episode_metadata(outcome, written)
    assert metadata["episode_sha256"] == hashlib.sha256(data).hexdigest()
    assert loads_episode(data).episode_id == outcome.episode.episode_id
    assert metadata["final_image_rgba_sha256"] == hashlib.sha256(outcome.image.rgba).hexdigest()
    assert metadata["status"] == "complete" and "events" not in metadata
    assert metadata["event_count"] == len(outcome.episode.events)


def test_episode_is_written_atomically_and_draft_is_never_named_episode_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = write_outcome(draw(max_actions=5), tmp_path / "draft")
    assert written.draft_path.name == "draft.json" and written.episode_path is None
    assert not (tmp_path / "draft" / "episode.json").exists()
    document = json.loads(written.draft_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "idraw-draft/v1"
    assert document["reason"] == "action_budget_exhausted"
    assert document["episode_prefix"]["status"] == "draft"

    target = tmp_path / "atomic.bin"
    target.write_bytes(b"old")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("image_drawer.inference.artifacts.os.replace", broken_replace)
    with pytest.raises(OSError):
        write_atomic(target, b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_png_encoder_is_deterministic_and_decodes_to_same_rgba() -> None:
    rgba = bytes(
        (x * 7 + y * 13 + c * 31) % 256 for y in range(5) for x in range(7) for c in range(4)
    )
    first, second = encode_png_rgba(7, 5, rgba), encode_png_rgba(7, 5, rgba)
    assert first == second
    assert decode_png_rgba(first) == (7, 5, rgba)
    with pytest.raises(PngError):
        decode_png_rgba(first[:-20])
    with pytest.raises(PngError):
        encode_png_rgba(7, 5, rgba[:-1])


def test_png_decoder_reads_rgb_png_with_filtered_rows() -> None:
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (9, 6))
    image.putdata(
        [(x * 20 % 256, y * 40 % 256, (x + y) * 9 % 256) for y in range(6) for x in range(9)]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    width, height, rgba = decode_png_rgba(buffer.getvalue())
    assert (width, height) == (9, 6)
    assert rgba == image.convert("RGBA").tobytes()


def test_attempts_file_records_reason_and_action_for_each_rejection(tmp_path: Path) -> None:
    script = [model("skip_stage"), *complete_actions()]
    outcome = draw(scripted_registry(actions=script))
    written = write_outcome(outcome, tmp_path / "o")
    rows = json.loads(written.attempts_path.read_text(encoding="utf-8"))["attempts"]
    assert len(rows) == len(outcome.attempts) and rows[0]["accepted"] is False
    assert rows[0]["action"]["type"] == "skip_stage" and rows[0]["reason"]
    assert rows[0]["stage_id"] == "rough" and rows[0]["seqs"] == []
    assert all(row["accepted"] and row["seqs"] for row in rows[1:])
    assert [r["attempt"] for r in rows] == list(range(1, len(rows) + 1))


def test_overwrite_touches_only_known_output_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_outcome(draw(), out)
    (out / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OutputConflictError):
        write_outcome(draw(), out)
    write_outcome(draw(max_actions=5), out, overwrite=True)
    names = {p.name for p in out.iterdir()}
    assert names == {"notes.txt", "draft.json", "image.png", "attempts.json"}
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert set(KNOWN_FILES) >= names - {"notes.txt"}
    file_path = tmp_path / "plain.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(OutputConflictError):
        write_outcome(draw(), file_path)
