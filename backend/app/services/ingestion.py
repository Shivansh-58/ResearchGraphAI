from __future__ import annotations

import hashlib
import mimetypes
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.core.security import Principal
from app.domain.models import Document, Entity, Relation
from app.domain.ports import Embedder, KnowledgeRepository
from app.infrastructure.document_processing import DocumentParser, SlidingWindowChunker
from app.infrastructure.graph_extraction import RuleBasedGraphExtractor


class IngestionService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        parser: DocumentParser,
        chunker: SlidingWindowChunker,
        embedder: Embedder,
        graph_extractor: RuleBasedGraphExtractor,
        document_directory: Path,
        refresh_indexes: Callable[[], None],
    ) -> None:
        self.repository = repository
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.graph_extractor = graph_extractor
        self.document_directory = document_directory
        self.refresh_indexes = refresh_indexes
        self._lock = threading.RLock()

    def ingest(
        self,
        filename: str,
        content: bytes,
        media_type: str | None = None,
        principal: Principal | None = None,
        allowed_groups: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[Document, bool]:
        with self._lock:
            return self._ingest(
                filename,
                content,
                media_type,
                principal or Principal(),
                allowed_groups or [],
                progress or (lambda _: None),
            )

    def _ingest(self, filename, content, media_type, principal, allowed_groups, progress):
        if not content:
            raise ValueError("Cannot ingest an empty document.")
        filename = Path(filename.replace("\\", "/")).name[:200]
        if not principal.admin and not set(allowed_groups).issubset(principal.groups):
            raise ValueError("You can share only with your own groups.")
        digest = hashlib.sha256(content).hexdigest()
        document_id = hashlib.sha256(
            f"{principal.workspace}:{principal.user_id}:{digest}".encode()
        ).hexdigest()
        existing = self.repository.get_document(document_id)
        if existing and existing.is_current:
            return existing, False
        previous = self.repository.previous_version(filename, principal)
        version = previous.version + 1 if previous else 1
        if existing:
            document_id = hashlib.sha256(f"{document_id}:v{version}".encode()).hexdigest()
        progress("parsing")
        pages = self.parser.parse_pages(filename, content)
        chunks = []
        for page, text in pages:
            for chunk in self.chunker.split(f"{document_id}:page{page}", text):
                chunk.document_id = document_id
                chunk.ordinal = len(chunks)
                chunk.page = page
                headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+)$", text[: chunk.end_char]))
                chunk.section = headings[-1].group(1)[:120] if headings else ""
                chunks.append(chunk)
        if not chunks:
            raise ValueError("The document did not produce any searchable chunks.")
        progress("embedding")
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding
        progress("graph_extraction")
        entities, chunk_entities, relations = self.graph_extractor.extract(chunks)
        # A hidden document must not decide the label/type shown for another user's node.
        # Canonical labels are deterministic; IDs are isolated between companies.
        namespace = hashlib.sha256(principal.workspace.encode()).hexdigest()[:16]
        scoped_ids = {
            entity.id: hashlib.sha256(f"{namespace}:{entity.id}".encode()).hexdigest()[:24]
            for entity in entities
        }
        entities = [
            Entity(
                scoped_ids[entity.id],
                entity.normalized_name,
                f"{namespace}:{entity.normalized_name}",
                "entity",
            )
            for entity in entities
        ]
        chunk_entities = [
            (chunk_id, scoped_ids[entity_id]) for chunk_id, entity_id in chunk_entities
        ]
        relations = [
            Relation(
                scoped_ids[edge.source_id],
                scoped_ids[edge.target_id],
                edge.relation_type,
                edge.weight,
                edge.evidence_chunk_id,
                edge.evidence_text,
            )
            for edge in relations
        ]

        suffix = Path(filename).suffix.lower()
        content_path = self.document_directory / f"{document_id}{suffix}"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_bytes(content)
        document = Document(
            id=document_id,
            filename=Path(filename).name,
            media_type=(
                media_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            ),
            byte_size=len(content),
            content_path=str(content_path),
            created_at=datetime.now(UTC),
            chunk_count=len(chunks),
            content_hash=digest,
            workspace=principal.workspace,
            owner_id=principal.user_id,
            allowed_groups=sorted(set(allowed_groups)),
            version=version,
            supersedes_id=previous.id if previous else None,
        )
        progress("persisting")
        try:
            self.repository.save_document_graph(
                document, chunks, entities, chunk_entities, relations
            )
        except Exception:
            content_path.unlink(missing_ok=True)
            raise
        self.refresh_indexes()
        return document, True

    def delete(self, document_id: str) -> bool:
        with self._lock:
            document = self.repository.get_document(document_id)
            if not document:
                return False
            deleted = self.repository.delete_document(document_id)
            if deleted:
                Path(document.content_path).unlink(missing_ok=True)
                self.refresh_indexes()
            return deleted
