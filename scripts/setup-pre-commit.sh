#!/usr/bin/env bash
# Pre-Commit Hook Setup Script
# Automates installation and verification of pre-commit hooks

set -e  # Exit on error

echo "========================================="
echo "Pre-Commit Hook Setup"
echo "========================================="
echo ""

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "❌ pre-commit not found"
    echo "Installing pre-commit..."
    pip install pre-commit
    echo "✅ pre-commit installed"
else
    echo "✅ pre-commit already installed ($(pre-commit --version))"
fi

echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found"
    echo "Installing pytest and dependencies..."
    pip install pytest pytest-asyncio
    echo "✅ pytest installed"
else
    echo "✅ pytest already installed ($(pytest --version | head -n1))"
fi

echo ""
echo "Installing git hooks..."
pre-commit install

echo ""
echo "Running hooks on all files (test installation)..."
pre-commit run --all-files || {
    echo ""
    echo "⚠️  Some hooks failed (expected on first run)"
    echo "This is normal - hooks may auto-fix issues"
    echo ""
}

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "What happens now:"
echo "  • Unit tests run automatically before each commit"
echo "  • Hooks check code quality, syntax, security"
echo "  • Commits blocked if tests fail"
echo ""
echo "To test manually:"
echo "  pre-commit run --all-files"
echo ""
echo "To skip hooks (use sparingly):"
echo "  git commit --no-verify"
echo ""
echo "Documentation: docs/operations/PRE_COMMIT_AUTOMATION.md"
echo ""
