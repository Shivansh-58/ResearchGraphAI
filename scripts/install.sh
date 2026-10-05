#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
docker info >/dev/null
python3 scripts/bootstrap.py
if [ "${1:-}" != "--offline" ]; then
    docker compose -f deploy/compose.yaml build api web
    docker compose -f deploy/compose.yaml -f deploy/compose.setup.yaml up -d ollama
    docker compose -f deploy/compose.yaml -f deploy/compose.setup.yaml exec ollama ollama pull qwen3:1.7b
    docker compose -f deploy/compose.yaml -f deploy/compose.setup.yaml exec ollama ollama pull nomic-embed-text
    docker compose -f deploy/compose.yaml -f deploy/compose.setup.yaml down
fi
docker compose -f deploy/compose.yaml up -d --no-build --pull never
printf '%s\n' 'ResearchGraph is starting at http://localhost:8080. Sign in with secrets/admin.token.'
