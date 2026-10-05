import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.domain.models import Chunk, SearchHit
from app.infrastructure.embeddings import OllamaEmbedder
from app.infrastructure.local_answers import OllamaAnswerGenerator, verified_answer
from app.infrastructure.local_models import ModelUnavailableError, OllamaClient
from app.infrastructure.structured_graph import StructuredGraphExtractor


def hit():
    return SearchHit(
        Chunk(
            "chunk",
            "doc",
            0,
            "Orion uses Atlas. Retention is 47 days.",
            8,
            0,
            40,
            filename="sample.txt",
            page=2,
            version=3,
        ),
        0.8,
    )


def test_local_answer_uses_structured_citations(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    captured = {}

    def chat(model, messages, schema=None):
        captured["messages"] = messages
        return json.dumps({"answer": "Retention is 47 days. [1]", "citations": [1]})

    monkeypatch.setattr(client, "chat", chat)
    answer = asyncio.run(
        OllamaAnswerGenerator(client, "qwen3:1.7b", 1000).generate(
            "How long is retention?", [hit()]
        )
    )
    assert answer.citations[0].page == 2 and answer.citations[0].version == 3
    assert "untrusted" in captured["messages"][0]["content"]


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"47 days [2]","citations":[2]}',
        '{"answer":"47 days [1]","citations":[]}',
    ],
)
def test_invalid_citation_numbers_abstain(text):
    assert verified_answer(text, [hit()]).citations == []


def test_single_declared_source_is_formatted_without_inventing_a_source():
    answer = verified_answer('{"answer":"47 days","citations":[1]}', [hit()])
    assert answer.text == "47 days [1]" and answer.citations[0].chunk_id == "chunk"


def test_invalid_json_is_a_model_error():
    with pytest.raises(ModelUnavailableError):
        verified_answer("not JSON", [hit()])


def test_model_can_explicitly_abstain():
    result = verified_answer(
        '{"answer":"Unknown","citations":[],"insufficient_evidence":true}', [hit()]
    )
    assert result.confidence == 0


def test_embedding_batches_are_validated_and_normalized(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    monkeypatch.setattr(
        client, "request", lambda path, payload: {"embeddings": [[3, 4] for _ in payload["input"]]}
    )
    embedder = OllamaEmbedder(client, "nomic")
    vectors = embedder.embed(["a"] * 17)
    assert len(vectors) == 17 and embedder.dimension == 2
    assert np.isclose(np.linalg.norm(vectors[0]), 1)


@pytest.mark.parametrize("vectors", [[], [[0, 0]], [[float("nan"), 1]]])
def test_embedding_invalid_vectors_fail(monkeypatch, vectors):
    client = OllamaClient("http://localhost:11434")
    monkeypatch.setattr(client, "request", lambda path, payload: {"embeddings": vectors})
    with pytest.raises(ModelUnavailableError):
        OllamaEmbedder(client, "nomic").embed(["a"])


def test_graph_requires_source_entities_and_verbatim_evidence(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    monkeypatch.setattr(
        client,
        "chat",
        lambda *args: json.dumps(
            {
                "entities": [{"name": "Orion"}, {"name": "Atlas"}, {"name": "Invented"}],
                "relations": [
                    {
                        "source": "Orion",
                        "target": "Atlas",
                        "type": "USES",
                        "evidence": "Orion uses Atlas",
                    },
                    {
                        "source": "Orion",
                        "target": "Invented",
                        "type": "USES",
                        "evidence": "invented quote",
                    },
                ],
            }
        ),
    )
    entities, links, relations = StructuredGraphExtractor(client, "qwen").extract([hit().chunk])
    assert {entity.name for entity in entities} == {"Orion", "Atlas"}
    assert len(relations) == 1 and relations[0].evidence_text == "Orion uses Atlas"


def test_explicit_relations_survive_small_model_omission(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    monkeypatch.setattr(client, "chat", lambda *args: '{"entities":[],"relations":[]}')
    entities, _, relations = StructuredGraphExtractor(client, "qwen").extract([hit().chunk])
    assert {entity.name for entity in entities} == {"Orion", "Atlas"}
    assert len(relations) == 1 and relations[0].relation_type == "USES"


@pytest.mark.parametrize("sentence", ["Orion never uses Atlas.", "Orion uses Atlas if authorized."])
def test_explicit_relation_fallback_does_not_infer_negations_or_conditions(monkeypatch, sentence):
    client = OllamaClient("http://localhost:11434")
    monkeypatch.setattr(client, "chat", lambda *args: '{"entities":[],"relations":[]}')
    chunk = hit().chunk
    chunk.text = sentence
    assert not StructuredGraphExtractor(client, "qwen").extract([chunk])[2]


def test_concurrent_duplicates_preserve_source(container):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: container.ingestion.ingest("sample.txt", b"Orion uses Atlas."), range(4)
            )
        )
    assert sum(created for _, created in results) == 1
    assert len(container.repository.list_documents()) == 1


def test_embedding_signature_change_is_rejected(container):
    with pytest.raises(ValueError, match="Embedding configuration changed"):
        container.repository.ensure_embedding_signature("different:space")


def test_lexically_unrelated_question_abstains(container):
    container.ingestion.ingest("notes.txt", b"Orion retention policy lasts 47 days.")
    answer = asyncio.run(container.orchestrator.answer("zqxv blorp flibble"))
    assert not answer.citations
