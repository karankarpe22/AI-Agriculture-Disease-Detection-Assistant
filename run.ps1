Write-Host "===================================================" -ForegroundColor Green
Write-Host "  Starting AI Agriculture Assistant Web Server..." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

if (Test-Path "$PSScriptRoot\AI_Argriculture_Assistant") {
    Set-Location "$PSScriptRoot\AI_Argriculture_Assistant"
} elseif (Test-Path "$PSScriptRoot\files-pasted-by-the-user-you") {
    Set-Location "$PSScriptRoot\files-pasted-by-the-user-you"
}

if (Test-Path ".\.venv-windows\Scripts\python.exe") {
    & ".\.venv-windows\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
}