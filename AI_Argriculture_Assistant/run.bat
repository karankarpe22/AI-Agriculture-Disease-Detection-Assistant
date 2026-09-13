@echo off
cd /d "%~dp0"
echo ===================================================
echo  Starting AI Agriculture Assistant Web Server...
echo ===================================================
echo.
echo Open your browser at: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.

".venv-windows\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
