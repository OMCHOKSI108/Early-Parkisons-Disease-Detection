# Simple PowerShell runner for development
# Usage (from repository root):
#   .\app\run.ps1

$env:PORT = $env:PORT -or "8000"
$env:HOST = $env:HOST -or "0.0.0.0"
# Use SQLite fallback by default for local dev
$env:ALLOW_SQLITE_FALLBACK = $env:ALLOW_SQLITE_FALLBACK -or "true"

Push-Location -Path (Join-Path $PSScriptRoot '.')
python main.py
Pop-Location
