param([string]$Version = '0.35.1')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw 'Invalid release version.' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot '.runtime'
$target = Join-Path $runtimeRoot "ollama-$Version"
New-Item -ItemType Directory -Force $runtimeRoot | Out-Null
if (Test-Path (Join-Path $target 'ollama.exe')) { Write-Output "Runtime already present: $target"; exit 0 }
$asset = 'ollama-windows-amd64.zip'
$base = "https://github.com/ollama/ollama/releases/download/v$Version"
$zip = Join-Path $runtimeRoot "ollama-$Version-windows-amd64.zip"
$hashFile = Join-Path $runtimeRoot "ollama-$Version-sha256sum.txt"
if (-not (Test-Path $hashFile)) { Invoke-WebRequest "$base/sha256sum.txt" -OutFile $hashFile }
if (-not (Test-Path $zip)) {
    Write-Output 'Downloading the official portable Ollama runtime (approximately 1.5 GB).'
    Invoke-WebRequest "$base/$asset" -OutFile $zip
}
$expected = ((Get-Content $hashFile | Where-Object { $_ -match "\s+\*?(\./)?$([regex]::Escape($asset))$" }) -split '\s+')[0]
if (-not $expected -or (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.ToLowerInvariant()) { throw 'Runtime checksum mismatch.' }
Expand-Archive -LiteralPath $zip -DestinationPath $target
Write-Output "Verified portable runtime: $target"
