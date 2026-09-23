"""SQLiteでmetadataとembeddingを保存し、binary payloadはfilesystemへ置く。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

from image_drawer.core import Part, PartSet, SourceImage
from image_drawer.part_bank.embedding import (
    EmbeddingIdentity,
    EmbeddingRecord,
    PartEmbedder,
)
from image_drawer.part_bank.index import BruteForceCosineIndex, PartIndex


class ProvenanceConflict(ValueError):
    """同一checksumへ別datasetやtrain/test splitを暗黙に割り当てることを禁止する。"""


class SQLitePartRepository:
    """Partとversioned embeddingを扱うsingle-writer local repository。"""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=30)
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
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
                CREATE TABLE IF NOT EXISTS embeddings (
                    part_id TEXT NOT NULL REFERENCES parts(id) ON DELETE CASCADE,
                    embedding_key TEXT NOT NULL,
                    identity_payload TEXT NOT NULL,
                    vector_payload TEXT NOT NULL,
                    metadata_payload TEXT NOT NULL,
                    PRIMARY KEY (part_id, embedding_key)
                );
                CREATE INDEX IF NOT EXISTS embeddings_key
                    ON embeddings(embedding_key, part_id);
                PRAGMA user_version = 2;
            """)
        except Exception:
            self._db.close()
            raise

    def __enter__(self) -> "SQLitePartRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def get_source(self, source_id: str) -> SourceImage:
        row = self._db.execute(
            "SELECT payload FROM sources WHERE id=?", (source_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown source: {source_id}")
        return SourceImage.from_json(row[0])

    def find_source(self, checksum: str) -> SourceImage | None:
        row = self._db.execute(
            "SELECT payload FROM sources WHERE checksum=?", (checksum,)
        ).fetchone()
        return SourceImage.from_json(row[0]) if row else None

    def get(self, part_id: str) -> Part:
        row = self._db.execute(
            "SELECT payload FROM parts WHERE id=?", (part_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown part: {part_id}")
        return Part.from_json(row[0])

    def list_sources(self) -> list[SourceImage]:
        return [
            SourceImage.from_json(row[0])
            for row in self._db.execute(
                "SELECT payload FROM sources ORDER BY id"
            )
        ]

    def list_parts(self, source_id: str | None = None) -> list[Part]:
        sql, args = "SELECT payload FROM parts", ()
        if source_id is not None:
            sql, args = sql + " WHERE source_id=?", (source_id,)
        return [
            Part.from_json(row[0])
            for row in self._db.execute(sql + " ORDER BY id", args)
        ]

    def origins(self, source_id: str) -> list[dict]:
        return [
            dict(uri=uri, provenance=json.loads(provenance))
            for uri, provenance in self._db.execute(
                "SELECT uri, provenance FROM origins "
                "WHERE source_id=? ORDER BY uri",
                (source_id,),
            )
        ]

    def check_source(self, source: SourceImage) -> SourceImage | None:
        """image fileをpublishする前にdeduplication互換性を確認する。"""
        existing = self.find_source(source.checksum)
        if existing is not None:
            if (
                existing.id,
                existing.dataset,
                existing.metadata.get("split"),
            ) != (
                source.id,
                source.dataset,
                source.metadata.get("split"),
            ):
                raise ProvenanceConflict(
                    "checksum already belongs to a different dataset or split"
                )
            if (existing.uri, existing.width, existing.height) != (
                source.uri,
                source.width,
                source.height,
            ):
                raise ProvenanceConflict(
                    "checksum has conflicting source location or dimensions"
                )
        return existing

    def store(
        self,
        source: SourceImage,
        parts: Iterable[Part],
        *,
        origin_uri: str,
        provenance: dict | None = None,
    ) -> tuple[int, int]:
        """1 SourceとそのPart・Originをまとめてcommitし、失敗時は一切commitしない。"""
        parts = list(parts)
        if len({part.id for part in parts}) != len(parts):
            raise ValueError("duplicate part IDs in one extraction")
        if any(part.source_image_id != source.id for part in parts):
            raise ValueError("part references a different source")
        source_json = json.dumps(
            source.to_dict(), allow_nan=False, sort_keys=True
        )
        part_json = [
            (
                part,
                json.dumps(part.to_dict(), allow_nan=False, sort_keys=True),
            )
            for part in parts
        ]
        provenance_json = json.dumps(
            provenance or {}, allow_nan=False, sort_keys=True
        )
        added_parts = 0
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self.check_source(source)
            if existing is None:
                self._db.execute(
                    "INSERT INTO sources VALUES (?, ?, ?)",
                    (source.id, source.checksum, source_json),
                )
            for part, payload in part_json:
                row = self._db.execute(
                    "SELECT payload FROM parts WHERE id=?", (part.id,)
                ).fetchone()
                if row is not None:
                    if Part.from_json(row[0]) != part:
                        raise ProvenanceConflict(
                            f"conflicting part: {part.id}; "
                            "bump extractor version"
                        )
                    continue
                self._db.execute(
                    "INSERT INTO parts VALUES (?, ?, ?)",
                    (part.id, source.id, payload),
                )
                added_parts += 1
            row = self._db.execute(
                "SELECT provenance FROM origins "
                "WHERE source_id=? AND uri=?",
                (source.id, origin_uri),
            ).fetchone()
            if row is not None and json.loads(row[0]) != json.loads(
                provenance_json
            ):
                raise ProvenanceConflict(
                    "origin provenance changed; refusing to overwrite it"
                )
            self._db.execute(
                "INSERT OR IGNORE INTO origins VALUES (?, ?, ?)",
                (source.id, origin_uri, provenance_json),
            )
        return int(existing is None), added_parts

    def get_embedding(
        self,
        part_id: str,
        identity: EmbeddingIdentity,
    ) -> EmbeddingRecord | None:
        row = self._db.execute(
            "SELECT identity_payload, vector_payload, metadata_payload "
            "FROM embeddings WHERE part_id=? AND embedding_key=?",
            (part_id, identity.key),
        ).fetchone()
        if row is None:
            return None
        identity_payload = json.loads(row[0])
        stored_identity = EmbeddingIdentity(
            family=identity_payload["family"],
            model=identity_payload["model"],
            version=identity_payload["version"],
            dimensions=identity_payload["dimensions"],
        )
        if stored_identity != identity:
            raise ProvenanceConflict(
                f"embedding identity mismatch for key: {identity.key}"
            )
        return EmbeddingRecord(
            part_id=part_id,
            identity=stored_identity,
            vector=tuple(json.loads(row[1])),
            metadata=json.loads(row[2]),
        )

    def store_embedding(self, record: EmbeddingRecord) -> bool:
        """1 embeddingを永続化し、新規recordの場合だけTrueを返す。"""
        exists = self._db.execute(
            "SELECT 1 FROM parts WHERE id=?", (record.part_id,)
        ).fetchone()
        if exists is None:
            raise KeyError(f"unknown part: {record.part_id}")
        if len(record.vector) != record.identity.dimensions:
            raise ValueError("embedding vector dimensions do not match identity")
        identity_json = json.dumps(
            record.identity.to_dict(), allow_nan=False, sort_keys=True
        )
        vector_json = json.dumps(
            list(record.vector), allow_nan=False, separators=(",", ":")
        )
        metadata_json = json.dumps(
            record.metadata, allow_nan=False, sort_keys=True
        )
        with self._db:
            row = self._db.execute(
                "SELECT identity_payload, vector_payload, metadata_payload "
                "FROM embeddings WHERE part_id=? AND embedding_key=?",
                (record.part_id, record.identity.key),
            ).fetchone()
            if row is not None:
                existing = (
                    json.loads(row[0]),
                    json.loads(row[1]),
                    json.loads(row[2]),
                )
                incoming = (
                    json.loads(identity_json),
                    json.loads(vector_json),
                    json.loads(metadata_json),
                )
                if existing != incoming:
                    raise ProvenanceConflict(
                        f"conflicting embedding: "
                        f"{record.part_id}/{record.identity.key}"
                    )
                return False
            self._db.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)",
                (
                    record.part_id,
                    record.identity.key,
                    identity_json,
                    vector_json,
                    metadata_json,
                ),
            )
        return True

    def list_embeddings(
        self,
        identity: EmbeddingIdentity,
    ) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        for part_id, identity_json, vector_json, metadata_json in self._db.execute(
            "SELECT part_id, identity_payload, vector_payload, metadata_payload "
            "FROM embeddings WHERE embedding_key=? ORDER BY part_id",
            (identity.key,),
        ):
            payload = json.loads(identity_json)
            stored_identity = EmbeddingIdentity(
                family=payload["family"],
                model=payload["model"],
                version=payload["version"],
                dimensions=payload["dimensions"],
            )
            if stored_identity != identity:
                raise ProvenanceConflict(
                    f"embedding identity mismatch for key: {identity.key}"
                )
            records.append(
                EmbeddingRecord(
                    part_id=part_id,
                    identity=stored_identity,
                    vector=tuple(json.loads(vector_json)),
                    metadata=json.loads(metadata_json),
                )
            )
        return records

    def ensure_embeddings(self, embedder: PartEmbedder) -> int:
        """選択したembedding identityについて不足しているembeddingだけを生成する。"""
        added = 0
        for part in self.list_parts():
            if self.get_embedding(part.id, embedder.identity) is not None:
                continue
            added += int(
                self.store_embedding(
                    EmbeddingRecord(
                        part_id=part.id,
                        identity=embedder.identity,
                        vector=tuple(embedder.embed_part(part)),
                        metadata={
                            "category": part.category,
                            "source_image_id": part.source_image_id,
                            "split": part.metadata.get("split"),
                            "extraction_method": part.extraction_method,
                            "extraction_version": part.extraction_version,
                        },
                    )
                )
            )
        return added

    def build_index(
        self,
        embedder: PartEmbedder,
    ) -> BruteForceCosineIndex:
        self.ensure_embeddings(embedder)
        index = BruteForceCosineIndex(embedder.identity)
        for record in self.list_embeddings(embedder.identity):
            index.add(
                record.part_id,
                record.vector,
                metadata=record.metadata,
            )
        return index

    def query(
        self,
        query: str,
        *,
        embedder: PartEmbedder,
        category: str | None = None,
        top_k: int = 20,
        filters: Mapping[str, object] | None = None,
        index: PartIndex | None = None,
    ) -> PartSet:
        """raw retrieval scoreとversionを保持したままPartを検索する。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        active_index = index or self.build_index(embedder)
        if active_index.embedding_identity != embedder.identity:
            raise ValueError(
                "index embedding identity does not match selected embedder"
            )

        active_filters = dict(filters or {})
        if category is not None:
            if not isinstance(category, str) or not category:
                raise ValueError("category must be a non-empty string or None")
            existing_category = active_filters.get("category")
            if existing_category not in (None, category):
                raise ValueError("category conflicts with filters['category']")
            active_filters["category"] = category

        hits = active_index.search(
            embedder.embed_text(query),
            top_k,
            filters=active_filters,
        )
        query_payload = {
            "query": query,
            "category": category,
            "filters": active_filters,
            "embedding": embedder.identity.to_dict(),
            "index": active_index.identity.to_dict(),
        }
        query_id = "query_" + hashlib.sha256(
            json.dumps(
                query_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        return PartSet(
            id=f"partset_{query_id.removeprefix('query_')}",
            query_id=query_id,
            category=category,
            part_ids=[hit.part_id for hit in hits],
            retrieval_scores=[hit.score for hit in hits],
            metadata={
                "query": query,
                "filters": active_filters,
                "embedding": embedder.identity.to_dict(),
                "index": active_index.identity.to_dict(),
            },
        )
