from __future__ import annotations

import re
from collections.abc import Sequence

from app.domain.models import Answer, Citation, SearchHit


def _citations(hits: Sequence[SearchHit]) -> list[Citation]:
    return [
        Citation(
            number=index,
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            filename=hit.chunk.filename,
            excerpt=hit.chunk.text[:260].strip(),
            score=round(hit.score, 4),
            page=hit.chunk.page,
            section=hit.chunk.section,
            version=hit.chunk.version,
        )
        for index, hit in enumerate(hits, start=1)
    ]


class ExtractiveAnswerGenerator:
    """Offline grounded baseline used when no external LLM is configured."""

    async def generate(self, question: str, hits: Sequence[SearchHit]) -> Answer:
        if not hits:
            return Answer(
                text="I could not find relevant evidence in the indexed documents.",
                citations=[],
                confidence=0.0,
            )
        terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        candidates: list[tuple[float, str, int]] = []
        for citation_number, hit in enumerate(hits, start=1):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit.chunk.text):
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(terms & sentence_terms) / max(len(terms), 1)
                candidates.append((overlap + hit.score * 0.4, sentence, citation_number))
        selected: list[str] = []
        used: set[int] = set()
        seen: set[str] = set()
        for _, sentence, citation_number in sorted(candidates, reverse=True):
            normalized = sentence.lower()
            if normalized in seen:
                continue
            selected.append(f"{sentence} [{citation_number}]")
            used.add(citation_number)
            seen.add(normalized)
            if len(selected) == 4:
                break
        confidence = min(0.95, sum(hit.score for hit in hits[:3]) / min(3, len(hits)))
        if not selected:
            return Answer("The available passages do not contain enough evidence to answer.", [], 0)
        citations = [item for item in _citations(hits) if item.number in used]
        return Answer(" ".join(selected), citations, round(confidence, 3))
