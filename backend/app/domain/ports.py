from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.core.security import Principal

from app.domain.models import Answer, Chunk, Document, Entity, Relation, SearchHit


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class KnowledgeRepository(Protocol):
    @property
    def revision(self) -> int: ...

    def get_document(self, document_id: str) -> Document | None: ...
    def list_documents(self, principal: Principal | None = None) -> list[Document]: ...
    def previous_version(self, filename: str, principal: Principal) -> Document | None: ...
    def save_document_graph(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        entities: Sequence[Entity],
        chunk_entities: Sequence[tuple[str, str]],
        relations: Sequence[Relation],
    ) -> None: ...
    def list_chunks(self, principal: Principal | None = None) -> list[Chunk]: ...
    def get_chunks(
        self, chunk_ids: Sequence[str], principal: Principal | None = None
    ) -> list[Chunk]: ...
    def chunks_for_entities(
        self, entity_ids: Sequence[str], principal: Principal | None = None
    ) -> list[Chunk]: ...
    def entities_for_chunks(
        self, chunk_ids: Sequence[str], principal: Principal | None = None
    ) -> dict[str, list[Entity]]: ...
    def neighboring_entities(
        self, entity_ids: Sequence[str], hops: int, principal: Principal | None = None
    ) -> list[Entity]: ...
    def delete_document(self, document_id: str) -> bool: ...


class AnswerGenerator(Protocol):
    async def generate(self, question: str, hits: Sequence[SearchHit]) -> Answer: ...
