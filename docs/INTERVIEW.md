# Explain ResearchGraph AI honestly in an AI internship interview

## A short introduction

“ResearchGraph AI is a private company-document assistant. It runs a pretrained small language
model, embeddings and graph extraction on company-controlled infrastructure, so documents don't
need to be sent to a third-party LLM API. I combine semantic search and BM25, then expand through
an evidence-backed entity graph. The answer includes citations that the user can inspect.

“The key design choice is separating knowledge from model weights: changing documents updates
retrieval instead of repeatedly fine-tuning. If a company needs different answer behavior, I have
an optional local LoRA/QLoRA workflow with held-out evaluation and explicit approval. I also enforce
document permissions before retrieval or graph traversal, and package the system for local Docker
deployment with an offline runtime.”

## Explain it from start to finish

1. **Problem:** private documents change often; uploading them to an outside API can violate a
   company's requirements, while training a new foundation model is unrealistic for this scope.
2. **Model choice:** reuse Qwen3 1.7B locally. I did not create its pretrained language abilities.
   Smaller models reduce resource requirements but can have weaker reasoning/extraction.
3. **Ingestion:** authenticate, queue the file, extract text with page provenance, chunk with overlap,
   generate local normalized embeddings, extract graph JSON, check quotes/names, persist atomically.
4. **Retrieval:** build only authorized current-document indexes, combine cosine/vector and BM25
   ranks with reciprocal rank fusion, then traverse authorized evidence edges for related passages.
5. **Generation:** pass the question and selected evidence to the local model. Validate inline source
   markers; make one bounded citation-format repair; return citations or abstain.
6. **Updates:** new versions supersede old ones in retrieval. A failed update leaves the last good
   version current. No model-weight update is required to reflect a changed policy.
7. **Customization:** train a low-rank adapter on reviewed examples, not raw documents by default.
   Compare against the base on held-out examples before deciding whether serving it is justified.
8. **Deployment:** use a connected public-model preparation phase, then a separate internal Docker
   runtime; company credentials are generated locally and knowledge persists outside containers.
9. **Evaluation:** distinguish integration correctness, factual quality, privacy and cost. Test
   each separately; a passing unit suite does not establish answer quality.

## Questions you should expect

**Why choose your model instead of downloading a small model?**
They should download a trusted small model: that is what this product uses. The differentiation is
the private system around it—ingestion/versioning, access-controlled retrieval, graph provenance,
citation inspection, adapter workflow and deployment—not claiming a superior new foundation model.

**Why should it work?**
The base supplies language ability; retrieval supplies relevant company evidence at query time.
Graph links can retrieve related evidence across documents. It works only to the extent that text
extraction, retrieval, graph extraction and the generator are reliable; benchmark those stages.

**Is it always cheaper than an online LLM?**
No. Reusing the base avoids pretraining, adapters reduce trainable parameters, quantization reduces
base-memory needs during QLoRA, and RAG avoids retraining for every document change. But local GPU,
electricity, utilization, engineering and maintenance cost money. Compare cost per acceptable answer
and privacy requirements on the company's workload, not just API token price.

**Does GraphRAG eliminate fine-tuning or hallucinations?**
No. It makes document facts available without putting every update into weights. Fine-tuning can
still help behavior. Retrieval/graph errors and hallucinations remain; citation syntax is not proof
of entailment. This implementation is entity-graph RAG, not graph-community summarization.

**How do you prevent secret documents leaking through the graph?**
Permission filtering happens before index construction, and graph edges/chunks are authorized on
every lookup. A graph edge must have visible evidence. Company-specific adapters are isolated
because access control cannot undo knowledge learned into shared weights.

**What did you verify yourself?**
The regression suite, production frontend build, real local-model upload-to-cited-answer path,
Compose configuration variants, and real tiny-model LoRA training/save/reload/evaluation mechanics.
Docker containers could not be run on the development laptop because it has no Docker engine.
The random-model training test is not evidence of a useful adapter or model-quality improvement.
Use the current evaluation guide for measured results; do not repeat unverified resume numbers.

**What would you improve next?**
A representative multi-hop/abstention benchmark, semantic citation faithfulness checks, retrieval
threshold tuning, better alias resolution, upload quotas, enterprise identity/TLS, and persistent
permission-filtered indexes for larger collections. Keep the graph only if its measured gain
justifies extraction and latency costs.
