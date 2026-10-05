from __future__ import annotations

import threading

from app.core.security import Principal
from app.domain.models import SearchHit
from app.domain.ports import Embedder, KnowledgeRepository
from app.infrastructure.search_indexes import BM25Index, VectorIndex


class HybridRetriever:
    def __init__(
        self,
        repository: KnowledgeRepository,
        embedder: Embedder,
        vector_index: VectorIndex,
        keyword_index: BM25Index,
        vector_weight: float = 0.55,
        min_similarity: float = 0.55,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.vector_weight = vector_weight
        self.min_similarity = min_similarity
        self._cache: dict = {}
        self._lock = threading.RLock()

    def refresh(self) -> None:
        with self._lock:
            self._cache.clear()

    def search(
        self, question: str, limit: int = 8, principal: Principal | None = None
    ) -> list[SearchHit]:
        principal = principal or Principal(role="admin")
        revision = self.repository.revision
        key = (principal, revision)
        with self._lock:
            if key not in self._cache:
                chunks = self.repository.list_chunks(principal)
                vectors_index, keyword_index = VectorIndex(), BM25Index()
                vectors_index.rebuild(chunks)
                keyword_index.rebuild(chunks)
                if len(self._cache) >= 32:
                    self._cache.clear()
                self._cache[key] = (vectors_index, keyword_index)
            vectors_index, keyword_index = self._cache[key]
        if not self.repository.list_documents(principal):
            return []
        candidate_limit = max(limit * 3, 12)
        query_vector = self.embedder.embed([question])[0]
        vectors = vectors_index.search(query_vector, candidate_limit)
        keywords = keyword_index.search(question, candidate_limit)
        raw_vectors = dict(vectors)
        raw_keywords = dict(keywords)
        vector_scores = {chunk_id: 1 / (60 + rank) for rank, (chunk_id, _) in enumerate(vectors, 1)}
        keyword_scores = {
            chunk_id: 1 / (60 + rank) for rank, (chunk_id, _) in enumerate(keywords, 1)
        }
        candidate_ids = list(dict.fromkeys([item[0] for item in vectors + keywords]))
        chunks = {chunk.id: chunk for chunk in self.repository.get_chunks(candidate_ids, principal)}
        hits = []
        for chunk_id in candidate_ids:
            if chunk_id not in chunks:
                continue
            if (
                raw_keywords.get(chunk_id, 0) <= 0
                and raw_vectors.get(chunk_id, 0) < self.min_similarity
            ):
                continue
            vector_score = vector_scores.get(chunk_id, 0.0)
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            score = 61 * (
                self.vector_weight * vector_score + (1 - self.vector_weight) * keyword_score
            )
            hits.append(
                SearchHit(
                    chunk=chunks[chunk_id],
                    score=score,
                    vector_score=raw_vectors.get(chunk_id, 0),
                    keyword_score=raw_keywords.get(chunk_id, 0),
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
