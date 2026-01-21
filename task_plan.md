# Task Plan: Set Up Ruff Automation (Big Tech Style)

## Goal
Implement a complete Ruff automation setup with pre-commit hooks, IDE integration, CI/CD pipeline, and project configuration for the FYP2 monorepo.

## Phases
- [ ] Phase 1: Project configuration - Add Ruff config to pyproject.toml
- [ ] Phase 2: Pre-commit setup - Install and configure pre-commit hooks
- [ ] Phase 3: IDE integration - Create VS Code settings for format-on-save
- [ ] Phase 4: CI/CD pipeline - Set up GitHub Actions workflow
- [ ] Phase 5: Documentation - Update README with Ruff usage instructions

## Key Questions
1. Should we configure Ruff at the root level or per-subproject?
   - **Decision**: Root level with per-directory overrides if needed
2. What rule sets should we enable?
   - **Decision**: Start with E, F, I, N, W (intermediate rules)
3. Should pre-commit run on all files or just staged?
   - **Decision**: Just staged files (standard pre-commit behavior)

## Decisions Made
- **Line length**: 88 (Black-compatible)
- **Target Python**: 3.11+ (based on project requirements)
- **Pre-commit**: Auto-fix enabled for developer convenience
- **CI**: Blocking checks to maintain code quality
- **VS Code**: Format on save + organize imports on save

## File Structure
```
FYP2/
├── .pre-commit-config.yaml          (Phase 2)
├── .github/workflows/ruff.yml       (Phase 4)
├── .vscode/settings.json            (Phase 3)
├── pyproject.toml                   (Phase 1 - update existing)
└── federated_pneumonia_detection/
    └── pyproject.toml               (Phase 1 - check if separate config needed)
```

## Agents Assignment
- **Phase 1**: `general-purpose` - Configuration files
- **Phase 2**: `general-purpose` - Pre-commit setup
- **Phase 3**: `frontend-architect` - VS Code settings (IDE expertise)
- **Phase 4**: `general-purpose` - GitHub Actions workflow
- **Phase 5**: `general-purpose` - Documentation updates

## Errors Encountered
(None yet)

## Status
**Currently in Phase 1** - About to check existing pyproject.toml and add Ruff configuration
