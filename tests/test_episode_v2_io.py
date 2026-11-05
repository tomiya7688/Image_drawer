"""S1c-2: Episode v2のcanonical保存・strict読込・1ファイルの原子的置換。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from episode_helpers import full_episode, full_raw, groups_of

from image_drawer.drawing import episode_io
from image_drawer.drawing.episode_io import (
    EPISODE_ARTIFACT_TYPE,
    MAX_EPISODE_BYTES,
    dumps_episode,
    episode_artifact_metadata,
    episode_sha256,
    load_episode,
    loads_episode,
    save_episode,
)
from image_drawer.drawing.validation import EpisodeFormatError, EpisodeStorageError

V1_FIXTURE = Path(__file__).parent / "fixtures" / "drawing_process_v1" / "golden_episode.json"


class SimulatedCrash(BaseException):
    """プロセス終了の代わりに、保存処理を途中で打ち切る。"""


def text_of() -> str:
    return dumps_episode(full_episode())


def reject_text(text: str | bytes) -> None:
    with pytest.raises(EpisodeFormatError):
        loads_episode(text)


def test_dumps_is_canonical_and_ends_with_single_lf():
    text = text_of()
    assert text.endswith("}\n") and not text.endswith("\n\n") and "\r" not in text
    top_keys = [k for k, _ in json.loads(text, object_pairs_hook=lambda pairs: pairs)]
    assert top_keys == sorted(top_keys)
    head, _, rest = text.partition('"events":[\n')
    body, _, tail = rest.partition("\n]")
    for line in body.split(",\n"):
        assert line == json.dumps(
            json.loads(line), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    assert head.startswith('{"brush_manifest_sha256":') and tail.endswith("}\n")


def test_dumps_loads_roundtrip_is_byte_identical():
    text = text_of()
    assert dumps_episode(loads_episode(text)) == text
    assert dumps_episode(loads_episode(text.encode("utf-8"))) == text


def test_loads_rejects_float_exponent_nan_and_duplicate_keys():
    text = text_of()
    for replacement in ('"seed":0.0', '"seed":0e0', '"seed":NaN', '"seed":0,"seed":0'):
        reject_text(text.replace('"seed":0', replacement, 1))


def test_loads_rejects_unknown_schema_version():
    reject_text(text_of().replace("drawing-episode/v2", "drawing-episode/v3", 1))
    reject_text("[]")
    reject_text("not json")


def test_loads_rejects_v1_header_with_v2_renderer():
    reject_text(text_of().replace("drawing-episode/v2", "drawing-episode/v1", 1))


def test_loads_rejects_wrong_brush_manifest_or_control_hash():
    raw = full_raw()
    for key in ("brush_manifest_sha256", "control_sha256"):
        bad = {**raw, key: "0" * 64}
        reject_text(json.dumps(bad))


def test_loads_rejects_wrong_workflow_hash():
    reject_text(json.dumps({**full_raw(), "workflow_sha256": "0" * 64}))
    reject_text(json.dumps({**full_raw(), "workflow_id": "other"}))


def test_loads_rejects_bom_and_invalid_utf8():
    text = text_of()
    reject_text("﻿" + text)
    reject_text(b"\xef\xbb\xbf" + text.encode("utf-8"))
    reject_text(text.encode("utf-8")[:-3] + b"\xff\xfe\n")


def test_loads_rejects_missing_unknown_and_duplicate_header_keys():
    raw = full_raw()
    missing = {k: v for k, v in raw.items() if k != "prompt"}
    reject_text(json.dumps(missing))
    reject_text(json.dumps({**raw, "extra": 1}))
    reject_text(text_of().replace('"prompt":', '"prompt":"a","prompt":', 1))


def test_loads_rejects_oversized_input():
    text = text_of().rstrip("\n").encode("utf-8")
    exact = text + b" " * (MAX_EPISODE_BYTES - len(text))
    assert len(exact) == MAX_EPISODE_BYTES
    assert loads_episode(exact).episode_id
    reject_text(exact + b" ")


def test_loads_rejects_event_count_20001():
    raw = full_raw()
    raw["events"] = [raw["events"][0]] * 20001
    reject_text(json.dumps(raw))


def test_loads_rejects_event_with_missing_stage_id_or_group():
    for key in ("stage_id", "group"):
        raw = full_raw()
        del raw["events"][3][key]
        reject_text(json.dumps(raw))


def test_loads_rejects_canvas_out_of_range_and_bad_seed():
    raw = full_raw()
    for canvas in (
        {"width": 0, "height": 8},
        {"width": 1025, "height": 8},
        {"width": "a", "height": 8},
    ):
        reject_text(json.dumps({**raw, "canvas": canvas}))
    for seed in (-1, 2**32, True, "0"):
        reject_text(json.dumps({**raw, "seed": seed}))


def test_loads_rejects_malformed_event_payload():
    raw = full_raw()
    index = next(i for i, e in enumerate(raw["events"]) if e["type"] == "select_color")
    raw["events"][index]["payload"] = {"color_id": 256}
    reject_text(json.dumps(raw))


def test_loads_dispatches_v1_episode_to_v1_path():
    episode = loads_episode(V1_FIXTURE.read_text(encoding="utf-8"))
    assert episode.renderer_version == "raster-v1"
    assert episode.events[0].stage_id is None and episode.events[0].group is None


def test_save_then_load_returns_equal_episode(tmp_path):
    episode = full_episode()
    save_episode(episode, tmp_path / "e.episode.json")
    assert load_episode(tmp_path / "e.episode.json") == episode


def test_save_returns_sha256_of_written_bytes(tmp_path):
    episode = full_episode()
    digest = save_episode(episode, tmp_path / "e.json")
    written = (tmp_path / "e.json").read_bytes()
    assert digest == hashlib.sha256(written).hexdigest() == episode_sha256(episode)


def test_save_replaces_existing_file_atomically(tmp_path):
    path = tmp_path / "e.json"
    path.write_text("old", encoding="utf-8")
    save_episode(full_episode(), path)
    assert path.read_text(encoding="utf-8") == text_of()
    assert [p.name for p in tmp_path.iterdir()] == ["e.json"]


def test_save_is_deterministic_for_same_episode(tmp_path):
    episode = full_episode()
    save_episode(episode, tmp_path / "a.json")
    save_episode(episode, tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_save_rejects_invalid_episode_without_touching_destination(tmp_path):
    path = tmp_path / "e.json"
    path.write_text("keep", encoding="utf-8")
    broken = full_episode()
    del broken.events[-1]
    with pytest.raises(EpisodeFormatError):
        save_episode(broken, path)
    assert path.read_text(encoding="utf-8") == "keep"
    assert [p.name for p in tmp_path.iterdir()] == ["e.json"]


def test_save_fails_cleanly_when_directory_is_missing(tmp_path):
    with pytest.raises(EpisodeStorageError) as info:
        save_episode(full_episode(), tmp_path / "missing" / "e.json")
    assert isinstance(info.value.__cause__, OSError)
    assert not (tmp_path / "missing").exists()


def old_file(tmp_path):
    path = tmp_path / "e.json"
    save_episode(full_episode(), path)
    return path, path.read_bytes()


def test_save_failure_before_replace_keeps_original_and_removes_temp(tmp_path, monkeypatch):
    path, original = old_file(tmp_path)

    def fail(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    other = full_episode()
    other.episode_id = "other"
    with pytest.raises(EpisodeStorageError):
        save_episode(other, path)
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["e.json"]


def crash(*_args, **_kwargs):
    raise SimulatedCrash


def changed_episode():
    episode = full_episode()
    episode.episode_id = "changed"
    return episode


def test_save_crash_after_temp_write_leaves_original_intact(tmp_path, monkeypatch):
    path, original = old_file(tmp_path)
    monkeypatch.setattr(os, "fsync", crash)
    with pytest.raises(SimulatedCrash):
        save_episode(changed_episode(), path)
    assert path.read_bytes() == original


def test_save_crash_after_fsync_leaves_original_intact(tmp_path, monkeypatch):
    path, original = old_file(tmp_path)
    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(SimulatedCrash):
        save_episode(changed_episode(), path)
    assert path.read_bytes() == original
    assert load_episode(path).episode_id != "changed"


def test_save_crash_after_replace_leaves_new_file_complete(tmp_path, monkeypatch):
    path, _ = old_file(tmp_path)
    monkeypatch.setattr(episode_io, "_sync_directory", crash)
    with pytest.raises(SimulatedCrash):
        save_episode(changed_episode(), path)
    assert load_episode(path).episode_id == "changed"


def test_orphan_temp_file_is_not_loadable_as_episode(tmp_path):
    orphan = tmp_path / ".e.json.abc123.tmp"
    orphan.write_bytes(text_of().encode("utf-8")[:500])
    with pytest.raises(EpisodeFormatError):
        load_episode(orphan)
    assert not (tmp_path / "e.json").exists()


def test_load_missing_file_raises_storage_error(tmp_path):
    with pytest.raises(EpisodeStorageError):
        load_episode(tmp_path / "none.json")


def test_episode_artifact_metadata_has_required_keys():
    episode = full_episode()
    meta = episode_artifact_metadata(episode, episode_sha256(episode))
    assert EPISODE_ARTIFACT_TYPE == "Episode"
    assert set(meta) == {
        "schema_version", "episode_sha256", "status", "termination_reason", "event_count",
        "group_count", "renderer_version", "workflow_id", "canvas",
    }  # fmt: skip
    assert meta["event_count"] == len(episode.events)
    assert meta["group_count"] == len(groups_of(episode.to_dict()))
    assert meta["status"] == "complete" and meta["termination_reason"] == "finished"
