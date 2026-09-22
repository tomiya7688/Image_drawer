"""SQLite metadata storage; binary payloads stay on the filesystem."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from image_drawer.core import Part, SourceImage


class ProvenanceConflict(ValueError):
    """A checksum cannot silently acquire another dataset or train/test split."""


class SQLitePartRepository:
    """Single-writer v0 repository with transactional source/part insertion."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=30)
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported Part Bank schema version: {version}")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parts (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES sources(id),
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS parts_source ON parts(source_id);
                CREATE TABLE IF NOT EXISTS origins (
                    source_id TEXT NOT NULL REFERENCES sources(id),
                    uri TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    PRIMARY KEY (source_id, uri)
                );
                PRAGMA user_version = 1;
            """)
        except Exception:
            self._db.close()
            raise

    def __enter__(self) -> SQLitePartRepository:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def get_source(self, source_id: str) -> SourceImage:
        row = self._db.execute("SELECT payload FROM sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown source: {source_id}")
        return SourceImage.from_json(row[0])

    def find_source(self, checksum: str) -> SourceImage | None:
        row = self._db.execute(
            "SELECT payload FROM sources WHERE checksum=?", (checksum,)
        ).fetchone()
        return SourceImage.from_json(row[0]) if row else None

    def get(self, part_id: str) -> Part:
        row = self._db.execute("SELECT payload FROM parts WHERE id=?", (part_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown part: {part_id}")
        return Part.from_json(row[0])

    def list_sources(self) -> list[SourceImage]:
        return [SourceImage.from_json(row[0]) for row in self._db.execute(
            "SELECT payload FROM sources ORDER BY id"
        )]

    def list_parts(self, source_id: str | None = None) -> list[Part]:
        sql, args = "SELECT payload FROM parts", ()
        if source_id is not None:
            sql, args = sql + " WHERE source_id=?", (source_id,)
        return [Part.from_json(row[0]) for row in self._db.execute(sql + " ORDER BY id", args)]

    def origins(self, source_id: str) -> list[dict]:
        return [dict(uri=uri, provenance=json.loads(provenance)) for uri, provenance
                in self._db.execute(
                    "SELECT uri, provenance FROM origins WHERE source_id=? ORDER BY uri",
                    (source_id,),
                )]

    def check_source(self, source: SourceImage) -> SourceImage | None:
        """Check deduplication compatibility before publishing any image files."""
        existing = self.find_source(source.checksum)
        if existing is not None:
            if (existing.id, existing.dataset, existing.metadata.get("split")) != (
                source.id, source.dataset, source.metadata.get("split")
            ):
                raise ProvenanceConflict("checksum already belongs to a different dataset or split")
            if (existing.uri, existing.width, existing.height) != (
                source.uri, source.width, source.height
            ):
                raise ProvenanceConflict("checksum has conflicting source location or dimensions")
        return existing

    def store(
        self, source: SourceImage, parts: Iterable[Part], *, origin_uri: str,
        provenance: dict | None = None,
    ) -> tuple[int, int]:
        """Commit one source, all its parts, and its origin, or none of them.

        Returns (new sources, new parts). Repeated identical records are no-ops.
        Existing extraction records are never silently replaced.
        """
        parts = list(parts)
        if len({part.id for part in parts}) != len(parts):
            raise ValueError("duplicate part IDs in one extraction")
        if any(part.source_image_id != source.id for part in parts):
            raise ValueError("part references a different source")
        # Serialize before opening the transaction; reject non-JSON metadata.
        source_json = json.dumps(source.to_dict(), allow_nan=False, sort_keys=True)
        part_json = [(part, json.dumps(part.to_dict(), allow_nan=False, sort_keys=True))
                     for part in parts]
        provenance_json = json.dumps(provenance or {}, allow_nan=False, sort_keys=True)
        added_parts = 0
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self.check_source(source)
            if existing is None:
                self._db.execute("INSERT INTO sources VALUES (?, ?, ?)",
                                 (source.id, source.checksum, source_json))
            for part, payload in part_json:
                row = self._db.execute("SELECT payload FROM parts WHERE id=?", (part.id,)).fetchone()
                if row is not None:
                    if Part.from_json(row[0]) != part:
                        raise ProvenanceConflict(f"conflicting part: {part.id}; bump extractor version")
                    continue
                self._db.execute("INSERT INTO parts VALUES (?, ?, ?)",
                                 (part.id, source.id, payload))
                added_parts += 1
            row = self._db.execute(
                "SELECT provenance FROM origins WHERE source_id=? AND uri=?",
                (source.id, origin_uri),
            ).fetchone()
            if row is not None and json.loads(row[0]) != json.loads(provenance_json):
                raise ProvenanceConflict("origin provenance changed; refusing to overwrite it")
            self._db.execute("INSERT OR IGNORE INTO origins VALUES (?, ?, ?)",
                             (source.id, origin_uri, provenance_json))
        return int(existing is None), added_parts
