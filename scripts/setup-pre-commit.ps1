# Pre-Commit Hook Setup Script (PowerShell)
# Automates installation and verification of pre-commit hooks

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Pre-Commit Hook Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if pre-commit is installed
Write-Host "Checking pre-commit installation..." -ForegroundColor Yellow
try {
    $version = pre-commit --version 2>&1
    Write-Host "✅ pre-commit already installed ($version)" -ForegroundColor Green
} catch {
    Write-Host "❌ pre-commit not found" -ForegroundColor Red
    Write-Host "Installing pre-commit..." -ForegroundColor Yellow
    pip install pre-commit
    Write-Host "✅ pre-commit installed" -ForegroundColor Green
}

Write-Host ""

# Check if pytest is installed
Write-Host "Checking pytest installation..." -ForegroundColor Yellow
try {
    $version = pytest --version 2>&1 | Select-Object -First 1
    Write-Host "✅ pytest already installed ($version)" -ForegroundColor Green
} catch {
    Write-Host "❌ pytest not found" -ForegroundColor Red
    Write-Host "Installing pytest and dependencies..." -ForegroundColor Yellow
    pip install pytest pytest-asyncio
    Write-Host "✅ pytest installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing git hooks..." -ForegroundColor Yellow
pre-commit install

Write-Host ""
Write-Host "Running hooks on all files (test installation)..." -ForegroundColor Yellow
try {
    pre-commit run --all-files
} catch {
    Write-Host ""
    Write-Host "⚠️  Some hooks failed (expected on first run)" -ForegroundColor Yellow
    Write-Host "This is normal - hooks may auto-fix issues" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "What happens now:" -ForegroundColor White
Write-Host "  • Unit tests run automatically before each commit" -ForegroundColor Gray
Write-Host "  • Hooks check code quality, syntax, security" -ForegroundColor Gray
Write-Host "  • Commits blocked if tests fail" -ForegroundColor Gray
Write-Host ""
Write-Host "To test manually:" -ForegroundColor White
Write-Host "  pre-commit run --all-files" -ForegroundColor Gray
Write-Host ""
Write-Host "To skip hooks (use sparingly):" -ForegroundColor White
Write-Host "  git commit --no-verify" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation: docs/operations/PRE_COMMIT_AUTOMATION.md" -ForegroundColor Cyan
Write-Host ""
