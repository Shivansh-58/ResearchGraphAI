"""Authenticated synthetic smoke test; never prints tokens or document text."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from uuid import uuid4

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--token-file", type=Path, default=Path("secrets/admin.token"))
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    with httpx.Client(
        base_url=args.url,
        headers={"Authorization": f"Bearer {args.token_file.read_text().strip()}"},
        timeout=240,
        trust_env=False,
    ) as client:
        ready = client.get("/api/v1/ready")
        ready.raise_for_status()
        marker = f"Sentinel{uuid4().hex[:8]}"
        upload = client.post(
            "/api/v1/documents/jobs",
            files={
                "file": (
                    f"{marker}.txt",
                    f"{marker} has a document retention period of 47 days. "
                    f"{marker} belongs to the Compliance team.",
                    "text/plain",
                )
            },
        )
        upload.raise_for_status()
        job_id = upload.json()["id"]
        deadline = time.monotonic() + args.timeout
        document_id = None
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/documents/jobs/{job_id}").json()
            if job["status"] == "failed":
                raise RuntimeError(job["error"])
            if job["status"] == "completed":
                document_id = job["document_id"]
                break
            time.sleep(2)
        if document_id is None:
            raise TimeoutError("Ingestion did not complete before the deadline.")
        try:
            answer = client.post(
                "/api/v1/query",
                json={
                    "question": f"What is the document retention period for {marker}?",
                    "include_debug": True,
                },
            )
            answer.raise_for_status()
            payload = answer.json()
            if "47" not in payload["answer"] or not payload["citations"]:
                category = (
                    "invalid citations"
                    if "valid source citations" in payload["answer"]
                    else "insufficient evidence"
                    if "enough evidence" in payload["answer"]
                    else "no retrieval"
                    if "relevant evidence" in payload["answer"]
                    else "missing fact"
                )
                print(
                    f"Synthetic check failed: {category}; "
                    f"retrieved={payload.get('trace', {}).get('retrieved_chunk_count', 0)}; "
                    f"citations={len(payload['citations'])}."
                )
            assert "47" in payload["answer"] and payload["citations"], (
                "Answer was not grounded in the synthetic evidence."
            )
            assert any(item["document_id"] == document_id for item in payload["citations"])
            graph = client.get("/api/v1/graph")
            graph.raise_for_status()
            assert any(
                marker.casefold() in edge["evidence_text"].casefold()
                for edge in graph.json()["edges"]
            ), "Synthetic graph edge was not extracted."
            unauthenticated = httpx.get(args.url + "/api/v1/documents", trust_env=False)
            assert unauthenticated.status_code == 401
            print(
                "PASS: local models ready; authenticated ingestion, retrieval, answer, "
                "citations, and access checks."
            )
            print(
                "Run this again with internet physically disconnected to verify "
                "your installation's offline behavior."
            )
        finally:
            client.delete(f"/api/v1/documents/{document_id}").raise_for_status()


if __name__ == "__main__":
    main()
