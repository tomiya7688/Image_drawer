"""Tiny, deterministic M3 fixtures; no private datasets or models required."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from image_drawer.core import Part, SourceImage
from image_drawer.part_bank import (
    IngestReport, ProvenanceConflict, Region, SQLitePartRepository,
    WholeImageExtractor, ingest_directory,
)
from image_drawer.part_bank.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "part_bank"


@pytest.fixture
def directories(tmp_path):
    source, bank = tmp_path / "input", tmp_path / "bank"
    source.mkdir()
    shutil.copyfile(FIXTURES / "palette.ppm", source / "palette.ppm")
    return source, bank


def test_ingest_persists_images_models_and_provenance(directories):
    source, bank = directories
    (source / "nested").mkdir()
    shutil.copyfile(FIXTURES / "gray.ppm", source / "nested" / "gray.PPM")
    report = ingest_directory(source, bank, dataset="fixture", split="train",
                              provenance={"license": "test-only", "group": "a"})
    assert report.to_dict() == dict(processed=2, skipped=0, sources_added=2,
                                    parts_added=2, duplicates=0, errors=[])
    assert IngestReport.from_json(report.to_json()) == report
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        assert len(repo.list_sources()) == len(repo.list_parts()) == 2
        for part in repo.list_parts():
            original = repo.get_source(part.source_image_id)
            assert repo.get(part.id) == part
            assert original.dataset == "fixture"
            assert original.metadata["split"] == part.metadata["split"] == "train"
            assert part.extraction_method == "whole_image"
            assert part.extraction_version == "1"
            assert part.mask_uri is None
            assert part.bbox == (0, 0, original.width, original.height)
            assert part.embedding_refs == {}
            assert repo.origins(original.id)[0]["provenance"]["license"] == "test-only"
            raw = (bank / original.uri).read_bytes()
            assert original.checksum == "sha256:" + hashlib.sha256(raw).hexdigest()
            with Image.open(bank / original.uri) as before, Image.open(bank / part.crop_uri) as after:
                assert after.mode == "RGBA"
                assert after.size == before.size
                assert after.tobytes() == before.convert("RGBA").tobytes()


def test_rerun_and_renamed_exact_copies_do_not_duplicate(directories):
    source, bank = directories
    first = ingest_directory(source, bank, dataset="fixture")
    shutil.copyfile(source / "palette.ppm", source / "renamed.ppm")
    second = ingest_directory(source, bank, dataset="fixture")
    assert first.sources_added == first.parts_added == 1
    assert second.sources_added == second.parts_added == 0
    assert second.duplicates == 2
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        original = repo.list_sources()[0]
        assert original.metadata["split"] is None
        assert len(repo.origins(original.id)) == 2
        assert len(repo.list_sources()) == len(repo.list_parts()) == 1


def test_ids_are_stable_across_bank_locations(directories, tmp_path):
    source, bank = directories
    other = tmp_path / "other-bank"
    for target in (bank, other):
        ingest_directory(source, target, dataset="fixture")
    with SQLitePartRepository(bank / "metadata.sqlite3") as a, \
            SQLitePartRepository(other / "metadata.sqlite3") as b:
        assert a.list_sources() == b.list_sources()
        assert a.list_parts() == b.list_parts()


def test_missing_crop_is_repaired_on_rerun(directories):
    source, bank = directories
    ingest_directory(source, bank, dataset="fixture")
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        path = bank / repo.list_parts()[0].crop_uri
    expected = path.read_bytes()
    path.unlink()
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.parts_added == 0 and not report.errors
    assert path.read_bytes() == expected


@pytest.mark.parametrize("dataset,split", [("fixture", "test"), ("other", "train"), ("fixture", None)])
def test_dedup_rejects_dataset_and_split_conflicts(directories, dataset, split):
    source, bank = directories
    ingest_directory(source, bank, dataset="fixture", split="train")
    report = ingest_directory(source, bank, dataset=dataset, split=split)
    assert report.sources_added == report.parts_added == report.duplicates == 0
    assert report.errors[0].error_type == "ProvenanceConflict"
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        assert len(repo.list_sources()) == len(repo.list_parts()) == 1
        assert repo.list_sources()[0].metadata["split"] == "train"


def test_corrupt_image_is_reported_and_other_files_continue(directories):
    source, bank = directories
    (source / "bad.png").write_bytes(b"not a png")
    (source / "notes.txt").write_text("not an image")
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.processed == 2 and report.skipped == 1
    assert report.sources_added == report.parts_added == 1
    assert report.errors[0].path == "bad.png"
    assert IngestReport.from_json(report.to_json()) == report


def test_png_verification_rejects_truncated_image(directories):
    source, bank = directories
    path = source / "truncated.png"
    with Image.new("RGB", (3, 3)) as image:
        image.save(path)
    path.write_bytes(path.read_bytes()[:35])
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.sources_added == 1
    assert len(report.errors) == 1


@pytest.mark.parametrize("options", [{"max_pixels": 5}, {"max_file_bytes": 10}])
def test_resource_limits_fail_before_registration(directories, options):
    source, bank = directories
    report = ingest_directory(source, bank, dataset="fixture", **options)
    assert report.sources_added == report.parts_added == 0
    assert len(report.errors) == 1
    assert "exceeds" in report.errors[0].message


def test_pillow_decompression_warning_is_an_error(directories, monkeypatch):
    source, bank = directories
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5)
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.errors[0].error_type == "DecompressionBombWarning"
    assert report.sources_added == 0


def test_rgba_alpha_is_preserved(directories):
    source, bank = directories
    (source / "palette.ppm").unlink()
    with Image.new("RGBA", (2, 1), (24, 48, 72, 80)) as image:
        image.save(source / "alpha.png")
    ingest_directory(source, bank, dataset="fixture")
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        with Image.open(bank / repo.list_parts()[0].crop_uri) as crop:
            assert crop.getpixel((0, 0)) == (24, 48, 72, 80)


def test_exif_policy_is_stored_pixel_coordinates(directories):
    source, bank = directories
    (source / "palette.ppm").unlink()
    with Image.new("RGB", (3, 2), "red") as image:
        exif = Image.Exif()
        exif[274] = 6
        image.save(source / "oriented.jpg", exif=exif)
    ingest_directory(source, bank, dataset="fixture")
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        part = repo.list_parts()[0]
        assert part.bbox == (0, 0, 3, 2)
        with Image.open(bank / part.crop_uri) as crop:
            assert crop.size == (3, 2)
            assert crop.getexif().get(274) is None


def test_multipage_is_rejected_instead_of_silently_using_first_frame(directories):
    source, bank = directories
    with Image.new("RGB", (2, 2), "red") as a, Image.new("RGB", (2, 2), "blue") as b:
        a.save(source / "multi.tiff", save_all=True, append_images=[b])
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.sources_added == 1
    assert "multipage" in report.errors[0].message


def test_symlinks_are_not_followed(directories):
    source, bank = directories
    (source / "alias.ppm").symlink_to(source / "palette.ppm")
    (source / "loop").symlink_to(source, target_is_directory=True)
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.processed == 1 and report.skipped == 1


@pytest.mark.parametrize("options", [
    {"dataset": ""}, {"split": "dev"}, {"max_pixels": 0},
    {"max_file_bytes": True}, {"provenance": []},
])
def test_invalid_settings_do_not_create_bank(directories, options):
    source, bank = directories
    with pytest.raises(ValueError):
        ingest_directory(source, bank, **({"dataset": "fixture"} | options))
    assert not bank.exists()


def test_missing_input_and_overlapping_paths_are_rejected(directories):
    source, bank = directories
    with pytest.raises(ValueError, match="does not exist"):
        ingest_directory(source / "missing", bank, dataset="fixture")
    with pytest.raises(ValueError, match="overlap"):
        ingest_directory(source, source / "bank", dataset="fixture")
    assert not bank.exists()


def test_extractor_is_replaceable_and_versioned(directories):
    class LeftPixel:
        method, version = "left_pixel", "1"
        def extract(self, source, image):
            return [Region((0, 0, 1, 1), "pixel")]
    source, bank = directories
    ingest_directory(source, bank, dataset="fixture")
    report = ingest_directory(source, bank, dataset="fixture", extractor=LeftPixel())
    assert report.sources_added == 0 and report.parts_added == 1
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        parts = repo.list_parts()
        pixel = next(part for part in parts if part.category == "pixel")
        assert len(parts) == 2 and pixel.bbox == (0, 0, 1, 1)
        with Image.open(bank / pixel.crop_uri) as crop:
            assert crop.getpixel((0, 0)) == (255, 0, 0, 255)
    report = ingest_directory(source, bank, dataset="fixture",
                              extractor=replace(WholeImageExtractor(), version="2"))
    assert report.parts_added == 1


@pytest.mark.parametrize("bbox", [(-1, 0, 1, 1), (0, 0, 4, 4), (0, 0, 0, 1), (0, 0, True, 1)])
def test_bad_extraction_does_not_register_partial_records(directories, bbox):
    class Bad:
        method, version = "bad", "1"
        def extract(self, source, image):
            return [Region((0, 0, 1, 1)), Region(bbox)]
    source, bank = directories
    report = ingest_directory(source, bank, dataset="fixture", extractor=Bad())
    assert len(report.errors) == 1
    with SQLitePartRepository(bank / "metadata.sqlite3") as repo:
        assert repo.list_sources() == repo.list_parts() == []


def test_extractor_failure_is_retryable(directories):
    class Broken:
        method, version = "broken", "1"
        def extract(self, source, image):
            yield Region((0, 0, 1, 1))
            raise RuntimeError("extractor failed")
    source, bank = directories
    report = ingest_directory(source, bank, dataset="fixture", extractor=Broken())
    assert report.errors[0].error_type == "RuntimeError"
    report = ingest_directory(source, bank, dataset="fixture")
    assert report.sources_added == report.parts_added == 1


def test_repository_transaction_rolls_back_on_conflicting_part(tmp_path):
    source = SourceImage("s", "source/image.png", 3, 2, "abc", "fixture")
    a = Part("a", "s", "generic", "parts/a.png", (0, 0, 1, 1))
    b = replace(a, id="b", crop_uri="parts/b.png")
    with SQLitePartRepository(tmp_path / "db.sqlite3") as repo:
        assert repo.store(source, [a], origin_uri="file:///first") == (1, 1)
        with pytest.raises(ProvenanceConflict):
            repo.store(source, [b, replace(a, category="changed")], origin_uri="file:///second")
        assert repo.list_parts() == [a]
        assert len(repo.origins("s")) == 1
        with pytest.raises(ValueError, match="different source"):
            repo.store(source, [replace(a, source_image_id="missing")], origin_uri="file:///first")
        with pytest.raises(KeyError):
            repo.get("missing")
        with pytest.raises(KeyError):
            repo.get_source("missing")


def test_repository_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "db.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="schema version"):
        SQLitePartRepository(path)


def test_cli_json_and_exit_codes(directories, capsys):
    source, bank = directories
    args = ["ingest", str(source), "--bank", str(bank), "--dataset", "fixture"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["sources_added"] == 1
    assert captured.err == ""
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["duplicates"] == 1
    (source / "bad.png").write_bytes(b"broken")
    assert main(args) == 1
    assert len(json.loads(capsys.readouterr().out)["errors"]) == 1
    assert main(["ingest", str(source / "missing"), "--bank", str(bank),
                 "--dataset", "fixture"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "does not exist" in captured.err
