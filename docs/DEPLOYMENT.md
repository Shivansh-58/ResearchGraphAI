# Deployment

## Native Windows installation

Prerequisites: Python 3.12, uv, Node 24 and sufficient free disk/RAM for the selected models. An NVIDIA
GPU improves latency but is not required for base inference. The development laptop has a 6 GB RTX
3050. Qwen3 1.7B is a practical starting point, not a universal hardware guarantee.

From the project root:

```powershell
.\scripts\start-native.ps1 -Prepare
```

This installs missing application dependencies, builds the interface, prepares hashed credentials,
downloads the portable Ollama runtime and public models, starts the local model service hidden, then
serves the application on http://127.0.0.1:8000. Internet is needed for preparation only. To restart
an already prepared installation, omit `-Prepare`. Do not run a second API process against the same
database. A separately running Ollama service is reused; verify that it is local and cloud-disabled.

Sign in using `secrets/admin.token`. The token is never printed by setup. Do not commit or email it.
The runtime/model directory is `.runtime/ollama-0.35.1` and `models/ollama/models`, respectively.
The model service remains running if the API terminal closes; stop it through your process manager
when no longer needed. Keep its port loopback-only.

## Docker connected preparation

Install/start Docker Desktop with WSL2 on Windows, or Docker Engine + Compose on Linux. GPU use also
requires compatible drivers and Docker GPU support (NVIDIA Container Toolkit on Linux). Containers
cannot run merely because Compose configuration validates; a functioning engine is required.

```powershell
.\scripts\install.ps1 -Gpu
```

For CPU use: `.\scripts\install.ps1`. Linux CPU: `sh scripts/install.sh`.
The installer checks the engine, preserves existing credentials, builds API/web images, temporarily
starts Ollama with a connected bootstrap network, downloads the two public models, stops the setup
stack **without removing its data volume**, and starts the private runtime with `--no-build --pull never`.

The runtime publishes only `127.0.0.1:8080`. API/model ports are not published. The Compose network
is `internal: true`, and cloud features are disabled. Do not use the setup override for ordinary
operation. Internet-disconnected execution should be tested before ingesting real company material.

```powershell
docker compose -f deploy/compose.yaml -f deploy/compose.gpu.yaml ps
.\backend\.venv\Scripts\python.exe scripts/verify_local.py --url http://localhost:8080
```

Omit the GPU override for CPU installations. The verifier requires the application's `httpx`
dependency; installing Python packages on an air-gapped machine must be prepared separately.

## Offline transfer to another company machine

On a connected preparation machine, first finish Docker installation/building and model downloads.
Then export:

```powershell
python scripts/offline_bundle.py create --output offline-bundles/researchgraph-0.2.0.tar
```

The bundle contains the API/web/Ollama runtime images, installation files and only blobs referenced
by the default public Qwen3 1.7B and nomic manifests. It excludes company documents, SQLite databases,
credentials, Ollama keys, datasets and private adapters. Verify licensing before distribution.
This export needs substantial free disk space; the image archive temporarily exists twice.

Transfer the bundle **and the `scripts/offline_bundle.py` importer** through your approved medium.
On an offline target with Docker and Python already installed:

```powershell
python offline_bundle.py import researchgraph-0.2.0.tar --destination D:\ResearchGraph
cd D:\ResearchGraph
.\scripts\install.ps1 -Offline -Gpu
```

Linux uses `python3` and `sh scripts/install.sh --offline` (CPU default).
The importer verifies inventory, sizes and SHA256 hashes before writing, refuses path traversal,
links and an existing destination, then loads images locally. Target setup creates fresh company
credentials. Internal checksums detect corruption, **not authenticity**; transfer an independently
verified archive hash/signature through a trusted channel. Docker images execute code, so import
only a trusted build. Optional training dependencies/base weights are not in the default bundle.

## Users, workspace administration and revocation

```powershell
python scripts/bootstrap.py --user analyst --workspace default --role member --groups engineering
```

Existing users/tokens are preserved. Distribute `secrets/analyst.token` securely. To revoke a user,
set their `disabled` field to true in `secrets/users.json`; new requests reload the configuration.
For rotation, generate a new high-entropy token and replace its SHA256 digest plus the securely
distributed token file. Do not change workspace identifiers casually: they are document boundaries.
Use separate installations for companies, especially when serving a trained company adapter.

## Persistence, backup and shutdown

Native data: `data/private`; Docker data: named volume `researchgraph_knowledge`. Back up source files
and SQLite together while the API is stopped (WAL writes can otherwise make raw copies inconsistent).
Back up hashed users configuration separately; keep plaintext tokens and model/dataset backups
access-restricted. To stop Docker: `docker compose -f deploy/compose.yaml down`; **do not add `-v`**
unless you intentionally want permanent deletion of company knowledge. Uninstalling images is
different from deleting persistent volumes.

The native and Docker installations do not automatically share their document databases. There is
no automatic migration from the old unsigned/hashing index to semantic embeddings: preserve the old
data directory, select a fresh one and reingest. Changing an embedding configuration requires this
reindexing; changing the answer model alone does not.

For LAN access, add an approved TLS reverse proxy, network firewall and enterprise identity layer;
do not simply expose the default unencrypted token endpoint to the public internet.
