# Part Bank v0 (Issue #5)

This milestone implements image ingest and provenance, not semantic segmentation
or vector retrieval. It uses the existing M1 SourceImage and Part records unchanged.

## Run

```bash
python -m pip install -e ".[dev]"
image-drawer-part-bank ingest /path/to/images \
  --bank ./data/part-bank --dataset my-dataset --split train
```

For a non-editable install from this checkout, use `python -m pip install ".[part-bank]"`.
This does not assume the project has been published to PyPI.
The equivalent module entrypoint is `python -m image_drawer.part_bank`.
The original `image-drawer --input ...` M0 command is unchanged.
Pillow is optional; the core package and mock runtime still have no runtime dependencies.

Input and bank directories must be disjoint. The command recursively visits files
in a stable directory/name order without following symlinks. Supported suffixes
are PNG, JPEG, WebP, BMP, TIFF and PPM (case-insensitive). Other files are counted
as skipped. Images must identify as one of these formats regardless of extension.
Animated/multipage images are explicitly rejected rather than silently truncated.

Stdout is a JSON report with `processed`, `skipped`, `sources_added`, `parts_added`,
`duplicates`, and `errors`. Exit codes: 0 = all attempted files succeeded;
1 = some files failed (see per-file path/type/message); 2 = invalid configuration,
missing optional dependency, or storage-level failure. Fatal errors go to stderr.
An empty directory produces a zero-count report. Nothing is uploaded.

## Python API

```python
from image_drawer.part_bank import SQLitePartRepository, ingest_directory

report = ingest_directory(
    "/path/to/images", "./data/part-bank",
    dataset="my-dataset", split="train",
    provenance={"license": "record the actual license", "source_group": "group-a"},
)
with SQLitePartRepository("./data/part-bank/metadata.sqlite3") as repository:
    for part in repository.list_parts():
        source = repository.get_source(part.source_image_id)
        origins = repository.origins(source.id)
```

All image URIs in records are **bank-relative POSIX paths**, resolved against the
bank root, not the process working directory. Origin URIs record original local
file locations separately. A moved bank retains the same record IDs and paths.

```text
part-bank/
  metadata.sqlite3
  source/<sha256>.<detected-format>
  parts/crops/part_<identity-hash>.png
```

The original bytes are copied into the bank. Source IDs and exact checksums derive
from these bytes, not filenames. Part IDs include source ID, extractor method and
version, category, bbox, and the versioned Pillow rasterizer identity. Distinct
extractor versions coexist. No binary image or embedding payload is put in JSON.

## Extraction and coordinates

WholeImageExtractor v1 returns one region for the entire decoded image. This is a
fixture/development baseline, not a face/object/semantic detector. Custom extractors
implement `extract(source, image) -> Iterable[Region]` with explicit `method` and
`version`; zero regions are allowed. Invalid or duplicate boxes reject that file.
Extractors must treat the supplied image as read-only and bump their version when
behavior changes. `Region` currently carries bbox and category; masks are not
produced in v0 (`Part.mask_uri` remains null).

Bboxes are `(x, y, width, height)` in **stored-pixel coordinates**. Source dimensions
and RGBA PNG crops use that same coordinate frame. EXIF display orientation is NOT
applied; crop EXIF is cleared so rotation metadata cannot change interpretation.
RGBA alpha is retained. There is no resizing, ICC color management or semantic
quality filtering in this milestone. Decoder/rasterizer versions are recorded.

## Deduplication, split policy, and recovery

A bank has one SourceImage per exact SHA-256 checksum. Renamed copies add origin
records, not new source/part rows. Re-ingesting an identical file is idempotent;
missing generated crops are recreated. Changed original bytes get new IDs.

`split` is `train`, `validation`, `test`, or unset. The first accepted dataset/split
assignment is retained. A duplicate imported with a different dataset or split
(including unset vs assigned) is rejected, never silently reassigned. This prevents
exact duplicates crossing splits within one bank. Multi-dataset membership and
near-duplicate grouping are future work. Group-level splitting still belongs
before extraction; the ingester does not randomly split individual parts.

A private temporary snapshot is hashed and decoded, so a changing input path cannot
make the recorded checksum refer to different bytes than the extracted image.
Validation checks file structure and full pixel decoding, with defaults of 40 million
pixels and 256 MiB per source file. Pillow decompression warnings are treated as
errors. Limits are configurable through the Python API.

Files are atomically published before one SQLite transaction commits a source,
its parts and an origin record. A failed transaction cannot partially insert rows.
A crash between file publication and SQL commit can leave unreferenced files;
a retry is safe. Automatic orphan deletion is deliberately not implemented.
Existing records are not silently overwritten. v0 supports one ingest writer per
bank, not concurrent ingest orchestration. Listing APIs materialize their results;
the ingest path processes one image at a time rather than loading the full dataset.

## Tests and CI

Tiny hand-authored PPM fixtures and generated test images cover exact pixel output,
alpha, provenance, persistence/reopening, deterministic IDs, duplicate imports,
split conflicts, corrupt/truncated images, resource limits, EXIF policy, multipage
rejection, extractor replacement/failure, rollback and command exit codes.

CI retains the existing tests/build/core-entrypoint checks. It then installs the
built wheel's part-bank extra into the fresh venv and executes its real CLI and
module entrypoint from a temporary directory outside the checkout, without
PYTHONPATH. It asserts imports come from the venv, opens the SQLite records and
PNG crop, verifies pixels and checksum, repeats ingest to check deduplication, and
checks that a corrupt file produces a nonzero exit and explicit error report.

API references used for file validation:
- [Pillow Image.open/load/verify](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [Pillow release notes](https://pillow.readthedocs.io/en/stable/releasenotes/index.html)

Next: Issue #6 adds embeddings, index abstraction, and real RETRIEVE_PARTS.
