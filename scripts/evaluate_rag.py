"""Run a private retrieval/GraphRAG ablation, storing metrics rather than document text."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from uuid import uuid4

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, default=Path("secrets/admin.token"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/synthetic.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    prefix = f"eval-{uuid4().hex[:10]}-"
    documents, results = {}, []
    with httpx.Client(
        base_url=args.url,
        timeout=240,
        trust_env=False,
        headers={"Authorization": "Bearer " + args.token_file.read_text().strip()},
    ) as client:
        client.get("/api/v1/ready").raise_for_status()
        try:
            for name, text in dataset["documents"].items():
                response = client.post(
                    "/api/v1/documents/jobs",
                    files={"file": (prefix + name, text.encode(), "text/plain")},
                )
                response.raise_for_status()
                job_id = response.json()["id"]
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    status = client.get(f"/api/v1/documents/jobs/{job_id}")
                    status.raise_for_status()
                    job = status.json()
                    if job["status"] == "failed":
                        raise RuntimeError(job["error"])
                    if job["status"] == "completed":
                        documents[name] = job["document_id"]
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError("Synthetic ingestion timed out.")
            for hops in (0, 1):
                for case in dataset["cases"]:
                    started = time.perf_counter()
                    response = client.post(
                        "/api/v1/query",
                        json={
                            "question": case["question"],
                            "top_k": 4,
                            "graph_hops": hops,
                            "include_debug": True,
                        },
                    )
                    response.raise_for_status()
                    answer = response.json()
                    cited = {item["document_id"] for item in answer["citations"]}
                    expected = {documents[name] for name in case.get("expected_documents", [])}
                    facts = case.get("required_facts", [])
                    facts_recalled = (
                        (
                            sum(fact.casefold() in answer["answer"].casefold() for fact in facts)
                            / len(facts)
                        )
                        if facts
                        else None
                    )
                    results.append(
                        {
                            "case_id": case["id"],
                            "graph_hops": hops,
                            "seconds": round(time.perf_counter() - started, 3),
                            "required_fact_recall": facts_recalled,
                            "expected_source_recall": len(cited & expected) / len(expected)
                            if expected
                            else None,
                            "abstained": not bool(cited),
                            "correct_abstention": not bool(cited)
                            if case.get("should_abstain")
                            else None,
                            "citation_count": len(answer["citations"]),
                            "timings_ms": answer.get("trace", {}).get("timings_ms", {}),
                        }
                    )
                    print(f"Evaluated case {case['id']}, graph_hops={hops}.")
        finally:
            for document_id in documents.values():
                client.delete(f"/api/v1/documents/{document_id}").raise_for_status()
    summary = {}
    for hops in (0, 1):
        group = [item for item in results if item["graph_hops"] == hops]
        summary[str(hops)] = {
            "median_seconds": statistics.median(item["seconds"] for item in group)
        }
        for metric in ("required_fact_recall", "expected_source_recall", "correct_abstention"):
            values = [item[metric] for item in group if item[metric] is not None]
            summary[str(hops)][metric] = statistics.mean(values) if values else None
    report = {
        "dataset": args.dataset.name,
        "cases": results,
        "summary": summary,
        "limitations": (
            "Synthetic string/source recall only, not a faithfulness "
            "or production-quality benchmark."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
