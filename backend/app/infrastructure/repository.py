from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.security import Principal
from app.domain.models import Chunk, Document, Entity, Relation


class SQLiteKnowledgeRepository:
    """SQLite persistence boundary for documents, chunks, and the knowledge graph."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        schema = """
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            content_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            chunk_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            UNIQUE(document_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunk_entities (
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY(chunk_id, entity_id)
        );
        CREATE TABLE IF NOT EXISTS relations (
            source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            weight REAL NOT NULL,
            evidence_chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            PRIMARY KEY(source_id, target_id, relation_type, evidence_chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);
        CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, workspace TEXT NOT NULL,
            filename TEXT NOT NULL, staged_path TEXT NOT NULL, payload TEXT NOT NULL,
            status TEXT NOT NULL, stage TEXT NOT NULL, created_at TEXT NOT NULL,
            document_id TEXT, error TEXT
        );
        """
        with self._connection() as connection:
            connection.executescript(schema)
            migrations = {
                "documents": {
                    "content_hash": "TEXT NOT NULL DEFAULT ''",
                    "workspace": "TEXT NOT NULL DEFAULT 'default'",
                    "owner_id": "TEXT NOT NULL DEFAULT 'local'",
                    "allowed_groups": "TEXT NOT NULL DEFAULT '[]'",
                    "version": "INTEGER NOT NULL DEFAULT 1",
                    "is_current": "INTEGER NOT NULL DEFAULT 1",
                    "supersedes_id": "TEXT",
                },
                "chunks": {
                    "page": "INTEGER",
                    "section": "TEXT NOT NULL DEFAULT ''",
                },
                "relations": {"evidence_text": "TEXT NOT NULL DEFAULT ''"},
            }
            for table, columns in migrations.items():
                existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                for column, definition in columns.items():
                    if column not in existing:
                        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('revision', '0')")

    @staticmethod
    def _scope(principal: Principal | None, alias: str = "documents") -> tuple[str, tuple]:
        if principal is None:
            return "1=1", ()
        clause = f"{alias}.workspace = ? AND {alias}.is_current = 1"
        params: tuple = (principal.workspace,)
        if not principal.admin:
            marks = ",".join("?" for _ in principal.groups) or "NULL"
            clause += (
                f" AND ({alias}.owner_id = ? OR EXISTS (SELECT 1 FROM "
                f"json_each({alias}.allowed_groups) WHERE value IN ({marks})))"
            )
            params += (principal.user_id, *principal.groups)
        return clause, params

    @property
    def revision(self) -> int:
        with self._connection() as connection:
            return int(
                connection.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0]
            )

    def ensure_embedding_signature(self, signature: str) -> None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='embedding_signature'"
            ).fetchone()
            if row and row[0] != signature:
                raise ValueError(
                    "Embedding configuration changed. Use a new data directory "
                    "or reingest into a new database."
                )
            if not row and connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]:
                raise ValueError(
                    "Legacy embeddings have no model signature. Preserve the old database "
                    "and reingest into a new data directory."
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('embedding_signature', ?)", (signature,)
            )

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            filename=row["filename"],
            media_type=row["media_type"],
            byte_size=row["byte_size"],
            content_path=row["content_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=row["status"],
            chunk_count=row["chunk_count"],
            content_hash=row["content_hash"],
            workspace=row["workspace"],
            owner_id=row["owner_id"],
            allowed_groups=json.loads(row["allowed_groups"]),
            version=row["version"],
            is_current=bool(row["is_current"]),
            supersedes_id=row["supersedes_id"],
        )

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        keys = set(row.keys())
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            ordinal=row["ordinal"],
            text=row["text"],
            token_count=row["token_count"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            embedding=json.loads(row["embedding"]),
            filename=row["filename"] if "filename" in keys else "",
            page=row["page"],
            section=row["section"],
            version=row["version"] if "version" in keys else 1,
        )

    @staticmethod
    def _entity(row: sqlite3.Row) -> Entity:
        return Entity(row["id"], row["name"], row["normalized_name"], row["kind"])

    def get_document(self, document_id: str) -> Document | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document(row) if row else None

    def list_documents(self, principal: Principal | None = None) -> list[Document]:
        clause, params = self._scope(principal)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM documents WHERE {clause} ORDER BY created_at DESC", params
            ).fetchall()
        return [self._document(row) for row in rows]

    def previous_version(self, filename: str, principal: Principal) -> Document | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE workspace=? AND owner_id=? "
                "AND lower(filename)=lower(?) ORDER BY version DESC LIMIT 1",
                (principal.workspace, principal.user_id, filename),
            ).fetchone()
        return self._document(row) if row else None

    def save_document_graph(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        entities: Sequence[Entity],
        chunk_entities: Sequence[tuple[str, str]],
        relations: Sequence[Relation],
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE documents SET is_current=0 WHERE workspace=? AND owner_id=? "
                "AND lower(filename)=lower(?)",
                (document.workspace, document.owner_id, document.filename),
            )
            connection.execute(
                "INSERT INTO documents (id,filename,media_type,byte_size,content_path,created_at,"
                "status,chunk_count,content_hash,workspace,owner_id,allowed_groups,version,"
                "is_current,supersedes_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.filename,
                    document.media_type,
                    document.byte_size,
                    document.content_path,
                    document.created_at.isoformat(),
                    document.status,
                    document.chunk_count,
                    document.content_hash,
                    document.workspace,
                    document.owner_id,
                    json.dumps(document.allowed_groups),
                    document.version,
                    int(document.is_current),
                    document.supersedes_id,
                ),
            )
            connection.executemany(
                "INSERT INTO chunks (id,document_id,ordinal,text,token_count,start_char,"
                "end_char,embedding,page,section) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.token_count,
                        chunk.start_char,
                        chunk.end_char,
                        json.dumps(chunk.embedding),
                        chunk.page,
                        chunk.section,
                    )
                    for chunk in chunks
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)",
                [
                    (entity.id, entity.name, entity.normalized_name, entity.kind)
                    for entity in entities
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO chunk_entities VALUES (?, ?)", chunk_entities
            )
            connection.executemany(
                "INSERT OR IGNORE INTO relations VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        relation.source_id,
                        relation.target_id,
                        relation.relation_type,
                        relation.weight,
                        relation.evidence_chunk_id,
                        relation.evidence_text,
                    )
                    for relation in relations
                ],
            )
            connection.execute(
                "UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'"
            )

    def list_chunks(self, principal: Principal | None = None) -> list[Chunk]:
        clause, params = self._scope(principal)
        query = f"""
            SELECT chunks.*, documents.filename, documents.version
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE {clause}
            ORDER BY chunks.document_id, chunks.ordinal
        """
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._chunk(row) for row in rows]

    def get_chunks(
        self, chunk_ids: Sequence[str], principal: Principal | None = None
    ) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        clause, scope_params = self._scope(principal)
        query = f"""
            SELECT chunks.*, documents.filename, documents.version
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.id IN ({placeholders}) AND {clause}
        """
        with self._connection() as connection:
            rows = connection.execute(query, tuple(chunk_ids) + scope_params).fetchall()
        by_id = {row["id"]: self._chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def chunks_for_entities(
        self, entity_ids: Sequence[str], principal: Principal | None = None
    ) -> list[Chunk]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        clause, scope_params = self._scope(principal)
        query = f"""
            SELECT DISTINCT chunks.*, documents.filename, documents.version
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            JOIN chunk_entities ON chunk_entities.chunk_id = chunks.id
            WHERE chunk_entities.entity_id IN ({placeholders}) AND {clause}
        """
        with self._connection() as connection:
            rows = connection.execute(query, tuple(entity_ids) + scope_params).fetchall()
        return [self._chunk(row) for row in rows]

    def entities_for_chunks(
        self, chunk_ids: Sequence[str], principal: Principal | None = None
    ) -> dict[str, list[Entity]]:
        result: dict[str, list[Entity]] = {chunk_id: [] for chunk_id in chunk_ids}
        if not chunk_ids:
            return result
        placeholders = ",".join("?" for _ in chunk_ids)
        clause, scope_params = self._scope(principal)
        query = f"""
            SELECT chunk_entities.chunk_id, entities.*
            FROM chunk_entities JOIN entities ON entities.id = chunk_entities.entity_id
            JOIN chunks ON chunks.id=chunk_entities.chunk_id
            JOIN documents ON documents.id=chunks.document_id
            WHERE chunk_entities.chunk_id IN ({placeholders}) AND {clause}
        """
        with self._connection() as connection:
            rows = connection.execute(query, tuple(chunk_ids) + scope_params).fetchall()
        for row in rows:
            result[row["chunk_id"]].append(self._entity(row))
        return result

    def neighboring_entities(
        self, entity_ids: Sequence[str], hops: int, principal: Principal | None = None
    ) -> list[Entity]:
        if not entity_ids or hops <= 0:
            return []
        visited = set(entity_ids)
        frontier = set(entity_ids)
        clause, scope_params = self._scope(principal)
        with self._connection() as connection:
            for _ in range(min(hops, 3)):
                placeholders = ",".join("?" for _ in frontier)
                query = f"""
                    SELECT source_id, target_id FROM relations
                    JOIN chunks ON chunks.id=relations.evidence_chunk_id
                    JOIN documents ON documents.id=chunks.document_id
                    WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                    AND {clause}
                    LIMIT 1000
                """
                params = tuple(frontier) + tuple(frontier) + scope_params
                rows = connection.execute(query, params).fetchall()
                next_frontier = {
                    entity_id
                    for row in rows
                    for entity_id in (row["source_id"], row["target_id"])
                    if entity_id not in visited
                }
                if not next_frontier:
                    break
                visited.update(next_frontier)
                frontier = set(sorted(next_frontier)[:100])
            discovered = visited - set(entity_ids)
            if not discovered:
                return []
            placeholders = ",".join("?" for _ in discovered)
            rows = connection.execute(
                f"SELECT * FROM entities WHERE id IN ({placeholders})", tuple(discovered)
            ).fetchall()
        return [self._entity(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            connection.execute(
                "DELETE FROM entities WHERE id NOT IN "
                "(SELECT DISTINCT entity_id FROM chunk_entities)"
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'"
                )
        return cursor.rowcount > 0

    def stats(self, principal: Principal | None = None) -> dict[str, int]:
        if principal is not None:
            chunks = self.list_chunks(principal)
            entities = self.entities_for_chunks([chunk.id for chunk in chunks], principal)
            return {
                "documents": len(self.list_documents(principal)),
                "chunks": len(chunks),
                "entities": len({entity.id for values in entities.values() for entity in values}),
            }
        with self._connection() as connection:
            return {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("documents", "chunks", "entities", "relations")
            }

    def graph_view(self, principal: Principal, limit: int = 100) -> dict:
        clause, params = self._scope(principal)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT relations.* FROM relations "
                "JOIN chunks ON chunks.id=relations.evidence_chunk_id "
                f"JOIN documents ON documents.id=chunks.document_id WHERE {clause} LIMIT ?",
                (*params, limit),
            ).fetchall()
            ids = sorted({row[key] for row in rows for key in ("source_id", "target_id")})
            if not ids:
                return {"nodes": [], "edges": []}
            marks = ",".join("?" for _ in ids)
            nodes = connection.execute(
                f"SELECT * FROM entities WHERE id IN ({marks})", ids
            ).fetchall()
        return {"nodes": [dict(row) for row in nodes], "edges": [dict(row) for row in rows]}
