# Local customization: LoRA and QLoRA

## What is trained

Use a downloaded instruction model, **not training a transformer from scratch**. The base stays
frozen and small low-rank adapter matrices are updated. QLoRA additionally loads the base in 4-bit
form during training. Lower trainable parameter count reduces optimizer memory; it does not remove
activation memory, data preparation, evaluation or GPU costs. QLoRA training does not automatically
make the separate serving path 4-bit: this implementation serves HF adapters in FP16 on GPU.

Use RAG for changing facts; use adapters for consistent answer style, terminology or task behavior.
Training on sensitive facts can cause memorization. A model cannot enforce document access control
over knowledge stored in its weights, so company-trained adapters must run in their own company
installation. Even within one company, use a training corpus appropriate for every intended user.

## 1. Prepare dependencies and base weights while connected

From the project root, create a separate training environment:

```powershell
uv venv .training-venv --python backend/.venv/Scripts/python.exe
uv pip install --python .training-venv/Scripts/python.exe -r training/requirements-qlora.txt
```

The default Windows wheel may be CPU-only. For GPU training, install the matching CUDA-enabled
PyTorch build from the official PyTorch instructions and check `torch.cuda.is_available()` before
starting. Linux Docker training uses the supplied PyTorch CUDA image. QLoRA also requires a supported
bitsandbytes/CUDA combination. This laptop's GPU inference is tested; full-model GPU training is not.

Ollama's quantized GGUF weights are for its runtime; the HF training scripts need separate
Transformers-compatible **safetensors** weights/tokenizer. Resolve a trusted model revision/commit
from the official model repository, then pin it:

```powershell
.\.training-venv\Scripts\python.exe scripts/prepare_models.py --model Qwen/Qwen3-1.7B --revision <verified-commit> --output models/base
```

Replace `<verified-commit>` with an actual repository revision. A 0.6B instruction model is an
alternative for smaller hardware; pass its official repository instead. Models are not interchangeable
with arbitrary adapters. Never download private company data or enable remote model code at runtime.

## 2. Create reviewed train/validation/test datasets

Create separate JSONL files in `training/datasets/`. Every row has `question`, `context` and `answer`;
answers cite numbered context sources. Optional `required_facts` supports a simple held-out check.

```json
{"question":"How long is retention?","context":"SOURCE [1]: The policy says 47 days.","answer":"Retention is 47 days [1].","required_facts":["47"]}
```

Use reviewed examples, including unsupported questions and adversarial instructions in evidence.
The current trainer requires a cited answer on each row; an abstention can cite the source that
establishes the absence/limit. It does not support uncited abstention-only training rows yet.
Split by document/topic/time, not merely random questions: question overlap checks alone cannot
detect paraphrase or semantic leakage. The eight bundled synthetic rows are workflow examples,
**not** an adequate dataset for useful company adaptation. Never use training examples as your
evaluation questions. Avoid overlong examples; evidence is not silently truncated.

Validate without loading a model:

```powershell
.\backend\.venv\Scripts\python.exe -m training.train_adapter --model-dir models/base --train training/datasets/train.jsonl --validation training/datasets/validation.jsonl --output data/private/training-runs/company-v1 --dry-run
```

## 3. Train offline

Close competing GPU workloads. On a prepared CUDA installation:

```powershell
.\.training-venv\Scripts\python.exe -m training.train_adapter --model-dir models/base --train training/datasets/train.jsonl --validation training/datasets/validation.jsonl --output data/private/training-runs/company-v1 --workspace default --qlora --epochs 2 --sequence-length 512
```

CPU LoRA omits `--qlora` and adds `--cpu`; small fixtures can run, but large-model CPU training can
be slow and memory-intensive. All model loading uses local files; HF offline/telemetry-disable
flags are set. The base is frozen, prompts are masked from loss, only assistant tokens are trained,
rank defaults to 8, seed is 42, batch size is 1 with four-step accumulation. Non-reentrant gradient
checkpointing avoids frozen-input gradient problems. Output directories must be new.

Artifacts: `adapter/adapter_model.safetensors`, adapter configuration/tokenizer, and `manifest.json`
with dataset hashes, base configuration hash, parameter counts, settings and validation loss.
Training traces do not print example text; datasets/adapters/manifests still require protection.

### Docker training

Prepare the trainer image while connected:

```powershell
docker compose -f deploy/compose.yaml --profile training build trainer
```

After transferring the model/datasets through your approved process, run with networking disabled:

```powershell
docker compose -f deploy/compose.yaml --profile training run --rm trainer --model-dir /models/base --train /datasets/train.jsonl --validation /datasets/validation.jsonl --output /data/training-runs/company-v1 --workspace default --qlora
```

The training image is not included in the default runtime-only offline bundle. Export it and any
optional adapter-serving image separately through your trusted image-transfer process.

## 4. Compare on held-out data; approve only after review

```powershell
.\.training-venv\Scripts\python.exe -m training.evaluate_adapter --model-dir models/base --run-dir data/private/training-runs/company-v1 --test training/datasets/test.jsonl
```

This runs the same prompts with and without the adapter, checks question/file overlap, and records
string fact recall and citation syntax. These checks are **not** semantic faithfulness, security or
business quality. Review individual outputs privately and test extraction errors, access leakage,
prompt injection, abstention, latency and representative document updates. The script stores only
aggregate metrics, not generated answers; add a secure review harness for a real company pilot.

After that review, rerun with `--approve-for-serving`. Approval additionally requires valid source
markers on every evaluation example and no fact-recall regression against the base. This is a
minimum gate, not a production guarantee. Failed candidates remain unapproved. The API refuses an
unapproved adapter or mismatched base configuration, and refuses mixed-company serving.

## 5. Serve the approved adapter

Native: install the training dependencies into the API environment, then set these fields in
`.env.local` and restart:

```dotenv
RESEARCHGRAPH_LLM_PROVIDER=transformers
RESEARCHGRAPH_MODEL_DIR=./models/base
RESEARCHGRAPH_ADAPTER_DIR=./data/private/training-runs/company-v1/adapter
RESEARCHGRAPH_GRAPH_PROVIDER=rules
```

Local nomic embeddings still use Ollama. The rules graph mode reduces competing GPU use but is
co-occurrence-based, not typed LLM extraction; set `GRAPH_PROVIDER=ollama` only with sufficient
hardware and the original local Qwen extraction model available. Only the answer generator gets
the HF adapter. Native transformer input is bounded to 3072 model tokens.

Docker (build the adapter-serving target while connected; runtime remains internal):

```powershell
$env:RESEARCHGRAPH_ADAPTER_RUN='company-v1'
docker compose -f deploy/compose.yaml -f deploy/compose.adapter.yaml build api
docker compose -f deploy/compose.yaml -f deploy/compose.adapter.yaml up -d --no-build --pull never
```

The run must exist in the Docker knowledge volume, not only the native data directory. Preserve
base/adapter pairing and installation workspace. Keep the unadapted base as your fallback.

## Verified training workflow smoke test

```powershell
.\.training-venv\Scripts\python.exe -m scripts.verify_training --output data/training-smoke-new
```

This creates a tiny random Qwen2-compatible local transformer, trains a real LoRA adapter for two
steps, saves/reloads it and evaluates base versus adapter offline on CPU. It has been executed
successfully. Its zero fact/citation scores are expected: it verifies code mechanics, **not** useful
language ability. No company adapter or performance improvement is claimed from this smoke test.
