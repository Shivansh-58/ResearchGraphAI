import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_private_access import headers, secure_settings

from app.infrastructure.local_models import ModelUnavailableError, OllamaClient
from app.main import create_app


def test_local_transport_never_follows_redirects_or_environment_proxy(monkeypatch):
    original = httpx.Client
    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    302, headers={"location": "https://external.invalid"}
                )
            ),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(ModelUnavailableError):
        OllamaClient("http://localhost:11434").installed_models()
    assert captured["trust_env"] is False and captured["follow_redirects"] is False


@pytest.mark.parametrize("payload", [[], "unexpected", 1])
def test_transport_rejects_non_object_json(monkeypatch, payload):
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            **kwargs,
        ),
    )
    with pytest.raises(ModelUnavailableError):
        OllamaClient("http://localhost:11434").request("/api/tags")


def test_chat_uses_local_structured_non_thinking_request(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    captured = {}

    def request(path, payload):
        captured.update(payload)
        return {"message": {"content": "synthetic reply"}}

    monkeypatch.setattr(client, "request", request)
    assert client.chat("qwen", [], {"type": "object"}) == "synthetic reply"
    assert not captured["think"] and not captured["stream"]
    assert captured["format"] == {"type": "object"}


def test_chat_rejects_malformed_message(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    monkeypatch.setattr(client, "request", lambda *args: {"message": "wrong type"})
    with pytest.raises(ModelUnavailableError):
        client.chat("qwen", [])


def test_readiness_checks_installed_models_and_missing_models(test_settings, monkeypatch):
    settings = replace(
        test_settings, llm_provider="ollama", embedding_provider="ollama", graph_provider="ollama"
    )
    with TestClient(create_app(settings)) as client:
        model_client = client.app.state.container.model_client
        monkeypatch.setattr(
            model_client, "installed_models", lambda: {"qwen3:1.7b", "nomic-embed-text:latest"}
        )
        assert client.get("/api/v1/ready").status_code == 200
        monkeypatch.setattr(model_client, "installed_models", lambda: set())
        assert client.get("/api/v1/ready").status_code == 503


def test_readiness_hides_transport_errors(test_settings, monkeypatch):
    with TestClient(create_app(replace(test_settings, llm_provider="ollama"))) as client:

        def unavailable():
            raise ModelUnavailableError("synthetic transport error")

        monkeypatch.setattr(
            client.app.state.container.model_client, "installed_models", unavailable
        )
        assert client.get("/api/v1/ready").status_code == 503


def test_system_counts_and_training_runs_are_workspace_scoped(test_settings):
    settings = secure_settings(test_settings)
    users = json.loads(settings.users_file.read_text())
    users["users"].append(
        {
            "id": "admin",
            "workspace": "company-a",
            "groups": [],
            "role": "admin",
            "token_sha256": hashlib.sha256(b"admin-test-token").hexdigest(),
        }
    )
    settings.users_file.write_text(json.dumps(users))
    for company in ("company-a", "company-b"):
        directory = settings.data_dir / "training-runs" / company
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {"workspace": company, "run_id": company, "status": "trained", "metrics": {}}
            )
        )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/training-runs", headers=headers("alice")).status_code == 403
        report = client.get("/api/v1/training-runs", headers=headers("admin")).json()
        assert [item["run_id"] for item in report["runs"]] == ["company-a"]
        counts = client.get("/api/v1/system", headers=headers("bob")).json()["counts"]
        assert counts == {"documents": 0, "chunks": 0, "entities": 0}
        health = client.get("/health")
        assert "counts" not in health.json() and health.headers["cache-control"] == "no-store"
