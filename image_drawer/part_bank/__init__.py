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
    "ingest_directory",
]
