from app.api.dependencies import ApplicationContainer


def test_content_addressing_deduplicates_identical_bytes(container: ApplicationContainer) -> None:
    content = b"FastAPI serves ResearchGraph. FAISS supports semantic vector retrieval."
    first, first_created = container.ingestion.ingest("notes.txt", content, "text/plain")
    second, second_created = container.ingestion.ingest("copy.txt", content, "text/plain")

    assert first_created is True
    assert second_created is False
    assert first.id == second.id
    assert len(container.repository.list_documents()) == 1
    assert first.chunk_count == 1


def test_delete_removes_document_and_derived_graph(container: ApplicationContainer) -> None:
    document, _ = container.ingestion.ingest(
        "graph.md",
        b"# ResearchGraph AI\nResearchGraph AI uses FastAPI and Knowledge Graph traversal.",
        "text/markdown",
    )

    assert container.ingestion.delete(document.id) is True
    assert container.ingestion.delete(document.id) is False
    assert container.repository.stats()["documents"] == 0
    assert container.repository.stats()["chunks"] == 0


def test_rejects_unsupported_documents(container: ApplicationContainer) -> None:
    try:
        container.ingestion.ingest("malware.exe", b"not really", "application/octet-stream")
    except ValueError as error:
        assert "Unsupported file type" in str(error)
    else:
        raise AssertionError("Unsupported input should fail")
