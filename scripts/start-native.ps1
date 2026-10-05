param([switch]$Prepare)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pythonPath = Join-Path $projectRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) {
    if (-not $Prepare) { throw 'Run with -Prepare once to install dependencies, or use Docker.' }
    uv sync --project backend --extra dev --extra faiss
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $pythonPath scripts/bootstrap.py
if ($LASTEXITCODE -ne 0) { throw 'Credential setup failed.' }
if ($Prepare -or -not (Test-Path frontend/dist/index.html)) {
    Push-Location frontend
    try {
        npm ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependencies failed.' }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
$modelServiceReady = $false
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null; $modelServiceReady = $true } catch {}
if (-not $modelServiceReady) {
    $ollamaPath = Join-Path $projectRoot '.runtime\ollama-0.35.1\ollama.exe'
    if (-not (Test-Path $ollamaPath)) {
        if (-not $Prepare) { throw 'Run with -Prepare once to download the local model runtime.' }
        & $PSScriptRoot/download-ollama.ps1
    }
    $env:OLLAMA_HOST = '127.0.0.1:11434'
    $env:OLLAMA_MODELS = Join-Path $projectRoot 'models\ollama\models'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_MAX_LOADED_MODELS = '1'
    Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; $modelServiceReady = $true; break } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $modelServiceReady) { throw 'Local model service did not start.' }
}
if ($Prepare) {
    & $pythonPath scripts/pull_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
}
Write-Output 'Open http://127.0.0.1:8000. Use secrets/admin.token to sign in. Close this terminal to stop the API.'
& $pythonPath -m uvicorn app.main:app --app-dir backend --env-file .env.local --host 127.0.0.1 --port 8000 --no-access-log
