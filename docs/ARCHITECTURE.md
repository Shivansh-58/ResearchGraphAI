# Architecture and complete pipeline

## Data path

```text
Browser (memory-only bearer token)
    → local web/API → authenticate → workspace + owner/group authorization

Upload → durable SQLite job → parse pages → overlapping chunks
       → local normalized embeddings → structured local entity/relation extraction
       → validate names and verbatim edge evidence → atomic SQLite persistence
       → invalidate authorized retrieval caches

Question → authorized chunks ONLY → local query embedding + BM25
         → weighted reciprocal rank fusion → authorized entity/edge traversal
         → evidence passages → local small model → check source numbers
         → one citation-format repair if necessary → cited answer or abstention

Optional labeled examples → local frozen base + trainable LoRA/QLoRA adapter
                         → validation loss → separate held-out comparison
                         → administrator approval → same-company adapter serving
```

The application orchestrates three pipeline components called retrieval, graph-expansion and
synthesis agents. They are deterministic workflow stages, not autonomous agents making remote
calls. There is one SQLite database, one upload worker, and one API process per installation.

## Ingestion details

1. Authentication identifies a user, workspace, groups and role. Blank group sharing means owner
   and workspace administrators only. Members may share only with their own groups.
2. An accepted upload is staged on disk and recorded in a durable SQLite job. Restarted running
   jobs return to the queue. Files are limited to 25 MB. Uploaded names are sanitized.
3. TXT/Markdown must be UTF-8. PDFs preserve one-based page numbers. Scans require prior OCR;
   encrypted PDFs must be unlocked. Nothing is silently sent to an external OCR service.
4. Whitespace-word chunks default to 220 words with 40 overlap, independently per PDF page.
   These are not model-token counts. Chunk identifiers are hashes; page/section/version metadata
   remain attached to the evidence.
5. The local embedding service processes batches of up to 16 passages. The backend checks shape,
   finite numbers, dimensions and nonzero norms, then normalizes vectors.
6. Unambiguous full-sentence patterns such as "Orion uses Atlas" first preserve explicit typed
   relations cheaply, excluding negations/trailing conditions. The local generator then extracts
   constrained JSON: entities, explicit aliases, typed relations,
   and verbatim evidence. Names and edge quotes absent from the passage are discarded. This is a
   provenance check, not proof that the extracted relationship is semantically correct.
7. Entities use company-scoped identifiers and deterministic canonical labels; hidden documents
   cannot supply a secret display name/type for a visible node. Edges retain their evidence chunk.
8. Documents, chunks and graph associations are saved in one transaction. The previous version
   stays current until replacement succeeds. Failed persistence removes the newly written source.
9. Identical bytes from the same owner/workspace are deduplicated. New content with the same
   filename creates a version; only current versions participate in normal retrieval. Historical
   sources remain available through permission-checked version/download endpoints.

## Retrieval and answer details

Indexes are built from authorized, current chunks, **before** similarity/keyword ranking. Each
cache is keyed by the full principal and repository revision, with at most 32 cached index pairs.
This avoids cross-user score effects and unauthorized candidates entering the graph stage.

Vector and BM25 rankings are fused as:

```text
score = 61 × (0.55 / (60 + vector_rank) + 0.45 / (60 + keyword_rank))
```

An absent ranking contributes zero. A candidate with neither a positive keyword score nor a
vector similarity of at least 0.55 is discarded. Common English question words are excluded from
BM25 so a match on "the" alone is not evidence of relevance. The conservative semantic-only
threshold is a heuristic needing tuning; it can miss valid paraphrases.
FAISS uses an exact flat index; NumPy is the fallback. BM25 is an in-memory local implementation.

Graph expansion finds entities in retrieved chunks, traverses at most three hops, and accepts
only edges supported by authorized current documents. The frontier/edge queries are bounded.
Connected evidence gets a heuristic graph boost; final passages are limited by `top_k` and a
character budget. This is local entity-graph RAG, **not** Microsoft's community-summary/global
GraphRAG implementation. No graph-wide summaries or vector database cluster are claimed.

Ollama receives a system instruction plus evidence marked as untrusted data. The output schema
requires an insufficient-evidence decision, answer text and citation numbers. Inline citation markers
must match the returned citation list and selected source numbers. If the model explicitly declares
one valid source but omits an inline marker, that marker is appended deterministically to the answer.
Other formatting mismatches get one bounded model repair. Invalid markers cause abstention.
Neither formatting normalization nor valid markers guarantee claim entailment.
The API's legacy `confidence` field is a retrieval heuristic; the UI labels it a retrieval score,
not a probability that the answer is correct.

## File system and responsibilities

```text
backend/
  app/
    core/                 configuration, authentication, sanitized logging
    domain/               data objects and protocol interfaces
    api/                  request schemas, dependency wiring, HTTP routes
    infrastructure/       SQLite, parsing, embeddings, search, local model adapters
    services/             ingestion, durable jobs, retrieval, pipeline orchestration
      agents/             retrieval, graph expansion, synthesis stages
  tests/                  pipeline, privacy, model-output and deployment regression tests
  pyproject.toml          Python dependencies and tooling
  uv.lock                 resolved application dependencies
frontend/
  src/components/         document panel, answers/evidence, graph connections
  src/lib/api.ts          same-origin authenticated requests
  src/App.tsx             sign-in, job polling, workspace/session lifecycle
  package-lock.json       pinned frontend dependency resolution
deploy/                   Dockerfiles, Compose variants, local reverse proxy
scripts/                  bootstrap, native/Docker setup, model preparation, offline bundles
training/                 dataset validation, local adapter training/evaluation
evaluation/               synthetic RAG ablation fixture
docs/                     architecture, deployment, security, training, evaluation, interview
secrets/                  ignored: plaintext user tokens + hashed user configuration
models/                   ignored: public Ollama weights and optional HF base weights
data/private/             ignored: native SQLite, source documents, staging, training runs
.runtime/                 ignored: portable model runtime and validation tools
```

Docker persists knowledge in the `researchgraph_knowledge` volume instead of `data/private`.
Base weights and source documents are separate from code, credentials and adapters.

## Scaling limits

This design targets one laptop/server and modest document collections. Per-principal exact indexes
consume memory; large collections need permission-filtered persistent search, queue backpressure,
pagination and a worker service. SQLite is not configured for multiple ingestion processes.
Graph alias resolution is local to an ingestion batch and can miss cross-document synonyms.
More graph hops are not automatically more accurate; evaluate the ablation before enabling them.
