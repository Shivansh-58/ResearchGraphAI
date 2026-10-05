import asyncio
import hashlib
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.core.security import Principal
from app.main import create_app


def secure_settings(settings):
    path = settings.data_dir / "users.json"
    path.write_text(
        json.dumps(
            {
                "users": [
                    {
                        "id": "alice",
                        "workspace": "company-a",
                        "groups": ["engineering"],
                        "role": "member",
                        "token_sha256": hashlib.sha256(b"alice-test-token").hexdigest(),
                    },
                    {
                        "id": "bob",
                        "workspace": "company-a",
                        "groups": ["engineering"],
                        "role": "member",
                        "token_sha256": hashlib.sha256(b"bob-test-token").hexdigest(),
                    },
                    {
                        "id": "eve",
                        "workspace": "company-b",
                        "groups": ["engineering"],
                        "role": "admin",
                        "token_sha256": hashlib.sha256(b"eve-test-token").hexdigest(),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return replace(settings, auth_required=True, environment="private", users_file=path)


def headers(user):
    return {"Authorization": f"Bearer {user}-test-token"}


def test_authentication_and_workspace_boundaries(test_settings):
    with TestClient(create_app(secure_settings(test_settings))) as client:
        assert client.get("/api/v1/documents").status_code == 401
        assert client.get("/api/v1/documents", headers=headers("wrong")).status_code == 401
        upload = client.post(
            "/api/v1/documents",
            headers=headers("alice"),
            files={
                "file": (
                    "finance.txt",
                    b"Orion has a secret launch budget of 913 dollars.",
                    "text/plain",
                )
            },
        )
        assert upload.status_code == 201
        doc_id = upload.json()["document"]["id"]
        for user in ("bob", "eve"):
            assert client.get("/api/v1/documents", headers=headers(user)).json() == []
            assert (
                client.get(
                    f"/api/v1/documents/{doc_id}/download", headers=headers(user)
                ).status_code
                == 404
            )
            assert (
                client.delete(f"/api/v1/documents/{doc_id}", headers=headers(user)).status_code
                == 404
            )
            response = client.post(
                "/api/v1/query",
                headers=headers(user),
                json={"question": "What is the Orion launch budget?"},
            )
            assert response.json()["citations"] == []
            assert "913" not in response.json()["answer"]


def test_group_sharing_does_not_grant_delete_or_cross_workspace_read(test_settings):
    with TestClient(create_app(secure_settings(test_settings))) as client:
        response = client.post(
            "/api/v1/documents",
            headers=headers("alice"),
            data={"allowed_groups": "engineering"},
            files={
                "file": ("shared.txt", b"Orion has a retention period of 47 days.", "text/plain")
            },
        )
        doc_id = response.json()["document"]["id"]
        assert len(client.get("/api/v1/documents", headers=headers("bob")).json()) == 1
        assert (
            client.delete(f"/api/v1/documents/{doc_id}", headers=headers("bob")).status_code == 403
        )
        assert client.get("/api/v1/documents", headers=headers("eve")).json() == []
        forbidden_share = client.post(
            "/api/v1/documents/jobs",
            headers=headers("alice"),
            data={"allowed_groups": "finance"},
            files={"file": ("bad.txt", b"Some content.", "text/plain")},
        )
        assert forbidden_share.status_code == 422


def test_graph_never_walks_edges_from_hidden_evidence(container):
    alice = Principal("alice", "a", ("engineering",))
    bob = Principal("bob", "a")
    container.ingestion.ingest(
        "public.txt", b"Orion connects to Atlas.", principal=alice, allowed_groups=["engineering"]
    )
    container.ingestion.ingest("secret.txt", b"Orion connects to ClassifiedBudget.", principal=bob)
    graph = container.repository.graph_view(alice)
    assert "ClassifiedBudget" not in json.dumps(graph)
    result = asyncio.run(
        container.orchestrator.answer(
            "What is Orion connected to?", include_debug=True, principal=alice
        )
    )
    assert "ClassifiedBudget" not in result.text
    assert {citation.filename for citation in result.citations} <= {"public.txt"}


def test_latest_version_only_is_retrieved(container):
    first, _ = container.ingestion.ingest("policy.txt", b"Orion retention lasts 30 days.")
    second, _ = container.ingestion.ingest("policy.txt", b"Orion retention lasts 60 days.")
    assert second.version == 2 and second.supersedes_id == first.id
    assert not container.repository.get_document(first.id).is_current
    result = asyncio.run(container.orchestrator.answer("How long does Orion retention last?"))
    assert "60" in result.text
    assert "30" not in result.text
    assert all(item.version == 2 for item in result.citations)


def test_jobs_are_durable_and_private(container):
    alice = Principal("alice", "a")
    bob = Principal("bob", "a")
    job = container.jobs.submit("sample.txt", b"Orion retention lasts 47 days.", alice, [])
    assert container.jobs.get(job["id"], bob) is None
    assert container.jobs.process_one()
    completed = container.jobs.get(job["id"], alice)
    assert completed["status"] == "completed"
    assert completed["document_id"]
    assert not list((container.settings.data_dir / "staging").iterdir())


def test_failed_job_does_not_leave_partial_document(container):
    principal = Principal()
    job = container.jobs.submit("blank.txt", b"\n\n ", principal, [])
    assert container.jobs.process_one()
    assert container.jobs.get(job["id"], principal)["status"] == "failed"
    assert container.repository.list_documents() == []


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://169.254.169.254",
        "http://localhost:11434/remote",
        "http://user:pass@localhost:11434",
        "http://localhost:11434?redirect=1",
    ],
)
def test_remote_or_unsafe_model_endpoints_fail_closed(test_settings, url):
    with pytest.raises(ValueError):
        replace(test_settings, ollama_base_url=url).validate()
