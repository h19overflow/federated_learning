# Start script for Federated Pneumonia Detection project
# Runs both frontend (npm) and backend (uvicorn) in parallel

$ErrorActionPreference = "Stop"

Write-Host "Starting Federated Pneumonia Detection..." -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start backend in new terminal (run from project root)
Write-Host "[Backend] Starting FastAPI server on http://127.0.0.1:8001" -ForegroundColor Green
$backendCmd = "cd '$scriptDir'; uv run uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

# Start frontend in new terminal
Write-Host "[Frontend] Starting Vite dev server on http://localhost:5173" -ForegroundColor Green
$frontendCmd = "cd '$scriptDir\xray-vision-ai-forge'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "Both servers starting in separate terminals!" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend:  http://127.0.0.1:8001" -ForegroundColor Yellow
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit this window..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
