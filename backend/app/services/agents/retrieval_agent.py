from __future__ import annotations

import asyncio

from app.core.security import Principal
from app.domain.models import SearchHit
from app.services.retrieval import HybridRetriever


class RetrievalAgent:
    name = "hybrid-retrieval-agent"

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    async def run(
        self, question: str, limit: int, principal: Principal | None = None
    ) -> list[SearchHit]:
        return await asyncio.to_thread(self.retriever.search, question, limit, principal)
