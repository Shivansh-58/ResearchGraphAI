import asyncio
import hashlib
import importlib.util
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from test_local_models import hit
from test_private_access import headers, secure_settings

from app.core.security import Principal
from app.domain.models import Entity
from app.infrastructure.local_answers import OllamaAnswerGenerator
from app.infrastructure.local_models import OllamaClient
from app.main import create_app

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("offline_bundle", ROOT / "scripts/offline_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def test_whitespace_question_is_rejected(test_settings):
    with TestClient(create_app(test_settings)) as client:
        assert client.post("/api/v1/query", json={"question": "   "}).status_code == 422


def test_disabled_user_is_immediately_revoked(test_settings):
    settings = secure_settings(test_settings)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/session", headers=headers("alice")).status_code == 200
        users = json.loads(settings.users_file.read_text())
        users["users"][0]["disabled"] = True
        settings.users_file.write_text(json.dumps(users))
        assert client.get("/api/v1/session", headers=headers("alice")).status_code == 401


def test_citation_repair_is_bounded(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    responses = iter(
        [
            '{"answer":"47 days [2]","citations":[2]}',
            '{"answer":"47 days [1]","citations":[1]}',
        ]
    )
    calls = []

    def chat(*args):
        calls.append(args)
        return next(responses)

    monkeypatch.setattr(client, "chat", chat)
    result = asyncio.run(OllamaAnswerGenerator(client, "qwen", 1000).generate("How long?", [hit()]))
    assert len(calls) == 2 and result.citations


def test_citation_repair_still_abstains_if_invalid(monkeypatch):
    client = OllamaClient("http://localhost:11434")
    calls = []
    monkeypatch.setattr(
        client, "chat", lambda *args: calls.append(1) or '{"answer":"47 days [8]","citations":[8]}'
    )
    result = asyncio.run(OllamaAnswerGenerator(client, "qwen", 1000).generate("How long?", [hit()]))
    assert len(calls) == 2 and not result.citations


def test_pdf_page_provenance_is_preserved(container, monkeypatch):
    monkeypatch.setattr(
        container.ingestion.parser,
        "parse_pages",
        lambda *args: [(1, "Orion uses Atlas."), (3, "Atlas retention is 47 days.")],
    )
    document, _ = container.ingestion.ingest("research.pdf", b"synthetic parser fixture")
    assert {chunk.page for chunk in container.repository.list_chunks()} == {1, 3}
    assert all(chunk.document_id == document.id for chunk in container.repository.list_chunks())


def test_scanned_and_encrypted_pdf_rejected(container):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    content = io.BytesIO()
    writer.write(content)
    with pytest.raises(ValueError, match="OCR"):
        container.ingestion.ingest("scan.pdf", content.getvalue())
    writer.encrypt("test-fixture-password")
    content = io.BytesIO()
    writer.write(content)
    with pytest.raises(ValueError, match="Unlock"):
        container.ingestion.ingest("encrypted.pdf", content.getvalue())


def test_failed_update_keeps_current_version(container, monkeypatch):
    previous, _ = container.ingestion.ingest("policy.txt", b"Orion retention is 30 days.")
    monkeypatch.setattr(
        container.ingestion.embedder,
        "embed",
        lambda _: (_ for _ in ()).throw(ValueError("synthetic failure")),
    )
    with pytest.raises(ValueError):
        container.ingestion.ingest("policy.txt", b"Orion retention is 60 days.")
    assert container.repository.get_document(previous.id).is_current
    assert len(container.repository.list_documents()) == 1


def test_entity_metadata_cannot_cross_company_boundaries(container):
    one, two = Principal("alice", "one"), Principal("alice", "two")
    container.ingestion.ingest("one.txt", b"Orion uses Atlas.", principal=one)
    container.ingestion.ingest("two.txt", b"Orion uses Atlas.", principal=two)
    graph_one, graph_two = (
        container.repository.graph_view(one),
        container.repository.graph_view(two),
    )
    assert {node["id"] for node in graph_one["nodes"]}.isdisjoint(
        {node["id"] for node in graph_two["nodes"]}
    )


def test_canonical_node_labels_do_not_leak_hidden_types(container, monkeypatch):
    def extract(chunks):
        return (
            [Entity("node", "SECRET_DISPLAY", "orion", "SECRET_TYPE")],
            [(chunks[0].id, "node")],
            [],
        )

    monkeypatch.setattr(container.ingestion.graph_extractor, "extract", extract)
    container.ingestion.ingest("a.txt", b"Orion", principal=Principal("alice", "one"))
    chunks = container.repository.list_chunks(Principal("alice", "one"))
    values = container.repository.entities_for_chunks([chunks[0].id], Principal("alice", "one"))
    assert values[chunks[0].id][0].name == "orion"
    assert values[chunks[0].id][0].kind == "entity"


def test_company_adapter_refuses_mixed_workspace_installation(test_settings):
    settings = secure_settings(test_settings)
    run = settings.data_dir / "run"
    run.mkdir()
    base = settings.data_dir / "base"
    base.mkdir()
    config = base / "config.json"
    config.write_text("{}")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "evaluated",
                "approved_for_serving": True,
                "workspace": "company-a",
                "base_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError, match="own workspace"):
        replace(settings, model_dir=base, adapter_dir=str(run / "adapter")).validate()


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "C:/outside", "back\\slash"])
def test_bundle_rejects_unsafe_paths(unsafe):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        info = tarfile.TarInfo(unsafe)
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    output.seek(0)
    with tarfile.open(fileobj=output) as archive, pytest.raises(ValueError, match="Unsafe"):
        bundle.safe_members(archive)


def test_bundle_verifies_checksums_before_creating_directory(test_settings):
    archive_path = test_settings.data_dir / "bundle.tar"
    payload = b"safe"
    manifest = json.dumps(
        {
            "format": 1,
            "files": {"README.md": {"size": 4, "sha256": hashlib.sha256(payload).hexdigest()}},
        }
    ).encode()
    with tarfile.open(archive_path, "w") as archive:
        for name, data in [("README.md", payload), ("bundle-manifest.json", manifest)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    target = test_settings.data_dir / "imported"
    bundle.import_bundle(archive_path, target, load_images=False)
    assert (target / "README.md").read_bytes() == payload
    with pytest.raises(ValueError, match="new directory"):
        bundle.import_bundle(archive_path, target, load_images=False)
