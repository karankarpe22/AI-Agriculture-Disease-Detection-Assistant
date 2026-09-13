@echo off
if exist "%~dp0AI_Argriculture_Assistant" (
    cd /d "%~dp0AI_Argriculture_Assistant"
) else if exist "%~dp0files-pasted-by-the-user-you" (
    cd /d "%~dp0files-pasted-by-the-user-you"
) else (
    cd /d "%~dp0"
)
echo ===================================================
echo  Starting AI Agriculture Assistant Web Server...
echo ===================================================
echo.
echo Open your browser at: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.

if exist ".venv-windows\Scripts\python.exe" (
    ".venv-windows\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
)
pause