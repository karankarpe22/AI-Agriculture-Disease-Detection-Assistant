Write-Host "===================================================" -ForegroundColor Green
Write-Host "  Starting AI Agriculture Assistant Web Server..." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

Set-Location $PSScriptRoot
& ".\.venv-windows\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
