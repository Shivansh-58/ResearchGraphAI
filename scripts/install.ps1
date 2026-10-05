param([switch]$Gpu, [switch]$Offline)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install Docker Desktop with the WSL2 backend, start it, then rerun this installer.' }
docker info | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is not running.' }
python scripts/bootstrap.py
if ($LASTEXITCODE -ne 0) { throw 'Credential bootstrap failed.' }
$files = @('-f', 'deploy/compose.yaml')
if ($Gpu) { $files += @('-f', 'deploy/compose.gpu.yaml') }
if (-not $Offline) {
    docker compose @files build api web
    if ($LASTEXITCODE -ne 0) { throw 'Image build failed.' }
    docker compose @files -f deploy/compose.setup.yaml up -d ollama
    if ($LASTEXITCODE -ne 0) { throw 'Model bootstrap service failed.' }
    docker compose @files -f deploy/compose.setup.yaml exec ollama ollama pull qwen3:1.7b
    if ($LASTEXITCODE -ne 0) { throw 'Language model download failed.' }
    docker compose @files -f deploy/compose.setup.yaml exec ollama ollama pull nomic-embed-text
    if ($LASTEXITCODE -ne 0) { throw 'Embedding model download failed.' }
    docker compose @files -f deploy/compose.setup.yaml down
}
docker compose @files up -d --no-build --pull never
if ($LASTEXITCODE -ne 0) { throw 'Private runtime startup failed.' }
Write-Output 'ResearchGraph is starting at http://localhost:8080. Sign in with secrets/admin.token.'
