from __future__ import annotations

from time import perf_counter

from app.core.security import Principal
from app.domain.models import Answer
from app.services.agents.graph_agent import GraphExpansionAgent
from app.services.agents.retrieval_agent import RetrievalAgent
from app.services.agents.synthesis_agent import SynthesisAgent


class ResearchGraphOrchestrator:
    """Coordinates specialized agents while retaining a debuggable execution trace."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        graph_agent: GraphExpansionAgent,
        synthesis_agent: SynthesisAgent,
    ) -> None:
        self.retrieval_agent = retrieval_agent
        self.graph_agent = graph_agent
        self.synthesis_agent = synthesis_agent

    async def answer(
        self,
        question: str,
        top_k: int = 6,
        graph_hops: int = 1,
        include_debug: bool = False,
        principal: Principal | None = None,
    ) -> Answer:
        started = perf_counter()
        retrieved = await self.retrieval_agent.run(question, max(top_k, 8), principal)
        retrieval_ms = (perf_counter() - started) * 1000
        expanded = await self.graph_agent.run(retrieved, graph_hops, max(top_k * 2, 10), principal)
        graph_ms = (perf_counter() - started) * 1000 - retrieval_ms
        answer = await self.synthesis_agent.run(question, expanded[:top_k])
        synthesis_ms = (perf_counter() - started) * 1000 - retrieval_ms - graph_ms
        if include_debug:
            answer.trace = {
                "agents": [
                    self.retrieval_agent.name,
                    self.graph_agent.name,
                    self.synthesis_agent.name,
                ],
                "retrieved_chunk_count": len(retrieved),
                "expanded_chunk_count": len(expanded),
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "graph_expansion": round(graph_ms, 2),
                    "synthesis": round(synthesis_ms, 2),
                    "total": round((perf_counter() - started) * 1000, 2),
                },
                "scores": [
                    {
                        "chunk_id": hit.chunk.id,
                        "final": round(hit.score, 4),
                        "vector": round(hit.vector_score, 4),
                        "keyword": round(hit.keyword_score, 4),
                        "graph": round(hit.graph_score, 4),
                        "entities": hit.matched_entities,
                    }
                    for hit in expanded[:top_k]
                ],
            }
        return answer
