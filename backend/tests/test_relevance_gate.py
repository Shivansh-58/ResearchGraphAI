import asyncio

import numpy as np
from test_local_models import hit

from app.infrastructure.search_indexes import BM25Index


def test_common_question_words_are_not_keyword_evidence():
    index = BM25Index()
    index.rebuild([hit().chunk])
    assert index.search("Who is the inventor on that world?", 10) == []


def test_moderate_similarity_without_keyword_evidence_abstains(container, monkeypatch):
    document, _ = container.ingestion.ingest("sample.txt", b"Orion retention is 47 days.")
    vector = np.asarray(container.repository.list_chunks()[0].embedding)
    orthogonal = np.roll(vector, 1)
    orthogonal -= np.dot(orthogonal, vector) * vector
    orthogonal /= np.linalg.norm(orthogonal)
    query = 0.4 * vector + np.sqrt(1 - 0.4**2) * orthogonal
    monkeypatch.setattr(container.ingestion.embedder, "embed", lambda _: [query.tolist()])
    answer = asyncio.run(container.orchestrator.answer("Who invented the nebular ice engine?"))
    assert not answer.citations and document.id
