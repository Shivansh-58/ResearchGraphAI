from __future__ import annotations

import asyncio

from app.core.security import Principal
from app.domain.models import SearchHit
from app.domain.ports import KnowledgeRepository


class GraphExpansionAgent:
    name = "graph-expansion-agent"

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    async def run(
        self,
        initial_hits: list[SearchHit],
        hops: int = 1,
        limit: int = 12,
        principal: Principal | None = None,
    ) -> list[SearchHit]:
        return await asyncio.to_thread(
            self._expand, initial_hits, hops, limit, principal or Principal(role="admin")
        )

    def _expand(self, initial_hits, hops, limit, principal):
        if not initial_hits or hops <= 0:
            return initial_hits[:limit]
        by_id = {hit.chunk.id: hit for hit in initial_hits}
        entities_by_chunk = self.repository.entities_for_chunks(list(by_id), principal)
        seed_entities = {
            entity.id for entities in entities_by_chunk.values() for entity in entities
        }
        neighbors = self.repository.neighboring_entities(
            sorted(seed_entities)[:100], hops, principal
        )
        expanded_entity_ids = list(seed_entities | {entity.id for entity in neighbors})
        graph_chunks = self.repository.chunks_for_entities(expanded_entity_ids, principal)
        graph_entities = self.repository.entities_for_chunks(
            [chunk.id for chunk in graph_chunks], principal
        )

        for chunk in graph_chunks:
            chunk_entities = graph_entities.get(chunk.id, [])
            matched_names = [entity.name for entity in chunk_entities if entity.id in seed_entities]
            graph_score = min(1.0, 0.15 + 0.12 * len(matched_names))
            if chunk.id in by_id:
                hit = by_id[chunk.id]
                hit.graph_score = graph_score
                hit.matched_entities = matched_names
                hit.score = min(1.0, hit.score + graph_score * 0.2)
            else:
                by_id[chunk.id] = SearchHit(
                    chunk=chunk,
                    score=graph_score * 0.65,
                    graph_score=graph_score,
                    matched_entities=matched_names,
                )
        return sorted(by_id.values(), key=lambda hit: hit.score, reverse=True)[:limit]
