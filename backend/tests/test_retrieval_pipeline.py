import asyncio

from app.api.dependencies import ApplicationContainer


def test_hybrid_and_graph_pipeline_returns_grounded_citations(
    container: ApplicationContainer,
) -> None:
    container.ingestion.ingest(
        "architecture.txt",
        (
            b"ResearchGraph AI uses FastAPI for its backend. "
            b"FAISS provides dense semantic retrieval. "
            b"BM25 provides exact keyword retrieval. "
            b"The Knowledge Graph connects entities across chunks. "
            b"A Graph Expansion Agent follows those entity links before grounded synthesis."
        ),
        "text/plain",
    )

    result = asyncio.run(
        container.orchestrator.answer(
            "How does ResearchGraph combine FAISS and BM25?",
            top_k=4,
            graph_hops=1,
            include_debug=True,
        )
    )

    assert result.citations
    assert "FAISS" in result.text
    assert result.confidence > 0
    assert result.trace["agents"] == [
        "hybrid-retrieval-agent",
        "graph-expansion-agent",
        "grounded-synthesis-agent",
    ]


def test_empty_knowledge_base_is_explicit(container: ApplicationContainer) -> None:
    result = asyncio.run(container.orchestrator.answer("What is indexed?"))
    assert result.confidence == 0.0
    assert result.citations == []
    assert "could not find" in result.text
