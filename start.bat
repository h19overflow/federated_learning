@echo off
title Federated Pneumonia Detection - Launcher
echo.
echo Starting Federated Pneumonia Detection...
echo.

:: Start backend in new terminal (run from project root)
echo [Backend] Starting FastAPI server on http://127.0.0.1:8001
start "Backend - FastAPI" cmd /k "cd /d %~dp0 && uv run uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001"

:: Small delay to let backend start first
timeout /t 2 /nobreak > nul

:: Start frontend in new terminal
echo [Frontend] Starting Vite dev server on http://localhost:5173
start "Frontend - Vite" cmd /k "cd /d %~dp0xray-vision-ai-forge && npm run dev"

echo.
echo Both servers starting in separate terminals!
echo.
echo   Backend:  http://127.0.0.1:8001
echo   Frontend: http://localhost:5173
echo.
pause
