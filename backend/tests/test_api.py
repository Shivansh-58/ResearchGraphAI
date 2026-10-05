from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_upload_query_list_delete_flow(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "knowledge.txt",
                    b"Graph RAG retrieves connected evidence.",
                    "text/plain",
                )
            },
        )
        assert upload.status_code == 201
        document_id = upload.json()["document"]["id"]

        listing = client.get("/api/v1/documents")
        assert listing.status_code == 200
        assert listing.json()[0]["id"] == document_id

        query = client.post(
            "/api/v1/query",
            json={"question": "What does Graph RAG retrieve?", "include_debug": True},
        )
        assert query.status_code == 200
        assert query.json()["citations"][0]["filename"] == "knowledge.txt"
        assert query.json()["trace"]["retrieved_chunk_count"] >= 1

        deleted = client.delete(f"/api/v1/documents/{document_id}")
        assert deleted.status_code == 204


def test_health_reports_pipeline_backends(test_settings: Settings) -> None:
    with TestClient(create_app(test_settings)) as client:
        payload = client.get("/health").json()
        system = client.get("/api/v1/system").json()
    assert payload["status"] == "ok"
    assert system["llm_provider"] == "extractive"
    assert system["counts"]["documents"] == 0
