"""Local Part Bank ingest, embeddings, indexing, and retrieval."""

from image_drawer.part_bank.embedding import (
    EmbeddingIdentity,
    EmbeddingRecord,
    MetadataHashEmbedder,
    PartEmbedder,
)
from image_drawer.part_bank.extractor import PartExtractor, Region, WholeImageExtractor
from image_drawer.part_bank.index import (
    BruteForceCosineIndex,
    IndexIdentity,
    PartIndex,
    SearchHit,
)
from image_drawer.part_bank.ingest import IngestFailure, IngestReport, ingest_directory
from image_drawer.part_bank.rendering import (
    build_composition,
    fixed_grid_layout,
    placements_from_layout,
    render_composition,
    save_png_atomic,
)
from image_drawer.part_bank.repository import ProvenanceConflict, SQLitePartRepository

__all__ = [
    "BruteForceCosineIndex",
    "EmbeddingIdentity",
    "EmbeddingRecord",
    "IndexIdentity",
    "IngestFailure",
    "IngestReport",
    "MetadataHashEmbedder",
    "PartEmbedder",
    "PartExtractor",
    "PartIndex",
    "ProvenanceConflict",
    "Region",
    "SQLitePartRepository",
    "SearchHit",
    "WholeImageExtractor",
    "build_composition",
    "fixed_grid_layout",
    "ingest_directory",
    "placements_from_layout",
    "render_composition",
    "save_png_atomic",
]
