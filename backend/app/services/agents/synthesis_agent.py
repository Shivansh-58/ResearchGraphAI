from __future__ import annotations

from app.domain.models import Answer, SearchHit
from app.domain.ports import AnswerGenerator


class SynthesisAgent:
    name = "grounded-synthesis-agent"

    def __init__(self, generator: AnswerGenerator) -> None:
        self.generator = generator

    async def run(self, question: str, hits: list[SearchHit]) -> Answer:
        return await self.generator.generate(question, hits)
