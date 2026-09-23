"""checksum deduplicationとatomic file操作を行うstreaming filesystem ingest。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from image_drawer.core import Part, SerializableModel, SourceImage
from image_drawer.part_bank.extractor import PartExtractor, Region, WholeImageExtractor
from image_drawer.part_bank.repository import SQLitePartRepository

EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".ppm"})
FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif", "PPM": ".ppm"}
SPLITS = ("train", "validation", "test")


@dataclass(slots=True)
class IngestFailure(SerializableModel):
    path: str
    error_type: str
    message: str


@dataclass(slots=True)
class IngestReport(SerializableModel):
    processed: int = 0
    skipped: int = 0
    sources_added: int = 0
    parts_added: int = 0
    duplicates: int = 0
    errors: list[IngestFailure] = field(default_factory=list)


def _files(root: Path) -> Iterator[Path]:
    def on_error(error: OSError) -> None:
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            yield Path(directory) / name


def _snapshot(path: Path, destination: Path, max_bytes: int) -> str:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as reader, destination.open("wb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            count += len(chunk)
            if count > max_bytes:
                raise ValueError(f"image exceeds max_file_bytes={max_bytes}")
            digest.update(chunk)
            writer.write(chunk)
    return digest.hexdigest()


def _validate_regions(regions: list[Region], source: SourceImage) -> None:
    seen = set()
    for region in regions:
        if not isinstance(region, Region):
            raise ValueError("extractor must return Region records")
        if not isinstance(region.category, str) or not region.category.strip():
            raise ValueError("region category must be non-empty")
        if len(region.bbox) != 4 or any(type(value) is not int for value in region.bbox):
            raise ValueError("bbox must contain four integers")
        x, y, width, height = region.bbox
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bbox origin must be non-negative and size positive")
        if x + width > source.width or y + height > source.height:
            raise ValueError("bbox is outside the source image")
        key = (tuple(region.bbox), region.category)
        if key in seen:
            raise ValueError("extractor returned a duplicate region")
        seen.add(key)


def _save_crop(image: Any, bbox: tuple[int, int, int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    x, y, width, height = bbox
    try:
        with image.crop((x, y, x + width, y + height)) as crop:
            crop.info.clear()
            crop.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest_directory(
    source_dir: str | Path,
    bank_dir: str | Path,
    *,
    dataset: str,
    split: str | None = None,
    extractor: PartExtractor | None = None,
    provenance: dict[str, Any] | None = None,
    max_pixels: int = 40_000_000,
    max_file_bytes: int = 256 * 1024 * 1024,
) -> IngestReport:
    """1 fileずつingestし、別fileが失敗しても正常fileは保持する。

    record内URIはbank-relative POSIX pathとする。
    座標はstored pixelを使い、EXIF orientationは適用しない。
    animated/multipage imageはrejectする。
    bank treeとinput treeは分離し、v0ではingest writerを1つだけ想定する。
    """
    root, bank = Path(source_dir).resolve(), Path(bank_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"source directory does not exist: {root}")
    if root == bank or root in bank.parents or bank in root.parents:
        raise ValueError("source directory and bank directory must not overlap")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, or None")
    for name, value in (("max_pixels", max_pixels), ("max_file_bytes", max_file_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    provenance = json.loads(json.dumps(provenance or {}, allow_nan=False))
    extractor = extractor if extractor is not None else WholeImageExtractor()
    if not all(isinstance(value, str) and value.strip()
               for value in (extractor.method, extractor.version)):
        raise ValueError("extractor method and version must be non-empty strings")
    try:
        from PIL import Image, __version__ as pillow_version
    except ImportError as exc:
        raise RuntimeError('install the image-drawer[part-bank] extra to ingest images') from exc

    report = IngestReport()
    bank.mkdir(parents=True, exist_ok=True)
    rasterizer = f"pillow-{pillow_version}-rgba-stored-pixels-v1"
    with SQLitePartRepository(bank / "metadata.sqlite3") as repository:
        for path in _files(root):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                report.skipped += 1
                continue
            report.processed += 1
            try:
                with tempfile.TemporaryDirectory(prefix=".ingest-", dir=bank) as temporary:
                    snapshot = Path(temporary) / "source"
                    digest = _snapshot(path, snapshot, max_file_bytes)
                    # hashしたものと同一byte列をdecodeし、途中で変化し得るinput fileを直接使わない。
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(snapshot, formats=list(FORMATS)) as probe:
                            if probe.width * probe.height > max_pixels:
                                raise ValueError(f"image exceeds max_pixels={max_pixels}")
                            if getattr(probe, "n_frames", 1) != 1:
                                raise ValueError("animated/multipage images are not supported")
                            image_format, original_mode = probe.format, probe.mode
                            probe.verify()
                        with Image.open(snapshot, formats=list(FORMATS)) as raw:
                            raw.load()
                            image = raw.convert("RGBA")
                    with image:
                        image.info.clear()
                        source = SourceImage(
                            id=f"source_{digest}",
                            uri=f"source/{digest}{FORMATS[image_format]}",
                            width=image.width, height=image.height,
                            checksum=f"sha256:{digest}", dataset=dataset,
                            metadata={"split": split, "format": image_format,
                                      "original_mode": original_mode,
                                      "coordinate_system": "stored_pixels",
                                      "decoder": "Pillow", "decoder_version": pillow_version},
                        )
                        repository.check_source(source)
                        regions = list(extractor.extract(source, image))
                        _validate_regions(regions, source)
                        parts = []
                        for region in regions:
                            identity = [source.id, extractor.method, extractor.version,
                                        region.category, region.bbox, rasterizer]
                            part_digest = hashlib.sha256(json.dumps(
                                identity, ensure_ascii=False, separators=(",", ":")
                            ).encode("utf-8")).hexdigest()
                            part_id = f"part_{part_digest}"
                            parts.append(Part(
                                id=part_id, source_image_id=source.id,
                                category=region.category,
                                crop_uri=f"parts/crops/{part_id}.png", bbox=tuple(region.bbox),
                                extraction_method=extractor.method,
                                extraction_version=extractor.version,
                                metadata={"split": split, "rasterizer": rasterizer,
                                          "coordinate_system": "stored_pixels"},
                            ))
                        # referenceをcommitする前に、完成したfileを先にpublishする。
                        stored_source = bank / source.uri
                        stored_source.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(snapshot, stored_source)
                        for part in parts:
                            _save_crop(image, part.bbox, bank / part.crop_uri)
                        added_sources, added_parts = repository.store(
                            source, parts, origin_uri=path.as_uri(), provenance=provenance,
                        )
                        report.sources_added += added_sources
                        report.parts_added += added_parts
                        report.duplicates += int(added_sources == 0)
            except sqlite3.Error:
                # storage-level failureを暗黙のpartial successとして扱わない。
                raise
            except Exception as exc:
                report.errors.append(IngestFailure(
                    path=path.relative_to(root).as_posix(),
                    error_type=type(exc).__name__, message=str(exc),
                ))
    return report
