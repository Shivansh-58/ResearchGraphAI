# Evaluation and verified results

Verified locally on 2026-10-04. These results describe this code and the supplied synthetic fixtures,
not general company-document accuracy, an enterprise security certification or a cost guarantee.

## Checks actually executed

| Check | Result |
| --- | --- |
| Backend regression suite | 67 tests passed |
| Backend statement coverage | 91% (1,446 statements, 136 missed) |
| Python lint/format | Passed |
| Frontend TypeScript + production build | Passed |
| Native upload → embeddings → typed graph → retrieval → cited local answer | Passed |
| Unauthorized document access | Rejected; owner/group/workspace and revocation regressions passed |
| Compose base/GPU/setup/adapter configurations | All validated with official Compose 5.6.0 |
| LoRA train/save/reload/held-out comparison | Passed on a tiny random local transformer, CPU |
| Docker image builds/container runtime | Not executed locally: no Docker engine installed |
| Full Qwen company adapter or GPU QLoRA training | Not executed; no company dataset supplied |
| Physical internet-disconnection test | Not executed; native model traffic used loopback |

Coverage measures tested code statements, **not model accuracy**. Most regression tests use
deterministic providers or mocked model output; the separate real-model smoke check verifies the
native integration. One upstream Starlette/httpx TestClient deprecation warning remains. CUDA is
not available in the separately installed default Windows training wheel, so its smoke test uses CPU.

## Real local RAG comparison

The Qwen3 1.7B/nomic runtime was evaluated with three synthetic documents and three questions:
one explicit fact, one cross-document question and one unrelated unsupported question. Each
question ran with graph expansion off and then with one hop. Test documents were deleted afterward.

The report is saved locally as `data/evaluation/local-ablation-final.json`.

| Metric | Hybrid retrieval only | Hybrid + graph |
| --- | ---: | ---: |
| Required-fact recall (two supported cases) | 1.0 | 1.0 |
| Strict expected-source recall | 0.75 | 0.75 |
| Correct abstention (one unsupported case) | 1/1 | 1/1 |
| Median observed request time | 4.662 seconds | 4.300 seconds |

This sample is too small to establish production quality or a GraphRAG advantage. The graph run
came second, so caching/warm-up/order effects confound the latency comparison. Strict source recall
expects both premise documents in the cross-document case; a correctly cited answer naming only
the team can cite just the team document and therefore score 0.5 on that case. This metric is not
the same as answer correctness or semantic citation faithfulness.

An earlier run failed the unrelated-question case: common words gave BM25 hits and moderate vector
similarity admitted irrelevant evidence. Stop-word removal and a conservative 0.55 semantic-only
threshold now reject that example before generation. The stronger threshold can reduce paraphrase
recall and requires representative validation. Early inference checks also exposed omitted inline
citations and missed obvious relations. Single-source formatting is now deterministic, and simple
explicit relationship patterns supplement constrained model extraction. These fixes are regression
tested; they do not eliminate all hallucinations or graph extraction errors.

## Reproduce the checks

From `backend`, run `.venv/Scripts/python.exe -m pytest` on Windows, or `uv run pytest` in a prepared
environment. From the project root, with the native app running:

```powershell
.\backend\.venv\Scripts\python.exe scripts/verify_local.py --url http://127.0.0.1:8000
.\backend\.venv\Scripts\python.exe scripts/evaluate_rag.py --output data/evaluation/new-run.json
```

The first check uses a fresh synthetic name, verifies graph evidence, a cited answer and 401 for an
unauthenticated request, then deletes the test document. The ablation uses random filenames and
stores case IDs/aggregate metrics, not private document text or generated answers. Run it in a
clean test workspace; existing documents with the same entity names can confound the results.
Output files must be new, so previous evidence is not overwritten.

Repeat the smoke test after disconnecting internet on a prepared installation to establish actual
offline behavior. For Docker, start the private base stack only, without the setup-network override,
and use `--url http://localhost:8080`. CI defines an optional manually dispatched container smoke
job, but a workflow file is not evidence that CI has run.

## Training smoke is a mechanics test, not a quality result

`scripts/verify_training.py` creates a tiny random transformer and trains a real LoRA adapter for
two steps, saves safetensors, reloads it and compares on held-out examples. Its base and adapter
fact/citation scores were zero, as expected for a random test fixture. It remains **unapproved for
serving**. This verifies the workflow and exposed/fixed gradient-checkpointing and tokenizer-input
compatibility issues. It does not establish GPU QLoRA compatibility or useful company adaptation.

## Before a company pilot

Build a representative labeled set with direct questions, multi-hop relationships, absent facts,
conflicting/updated documents, paraphrases and prompt injections. Measure retrieval recall@k,
human-reviewed claim faithfulness, citation completeness, abstention precision/recall, access
leakage, latency distribution, ingestion throughput, memory and cost per acceptable answer.
Separate training/validation/test by document/topic/time. Compare plain local model, vector RAG,
hybrid RAG and hybrid+graph under the same model/hardware, randomized ordering and repeated runs.
Keep graph expansion or an adapter only when its measured benefit justifies its cost.
