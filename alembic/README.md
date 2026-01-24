# Alembic Database Migrations

**Last Updated**: 2026-01-24

This directory contains Alembic migrations for the Federated Pneumonia Detection database.

## Overview

Alembic is a database migration tool for SQLAlchemy that provides:
- Version-controlled schema changes
- Automated migration generation from model changes
- Rollback capabilities for schema changes
- SQL script generation for manual review

## Directory Structure

```
alembic/
├── README.md                  # This file
├── env.py                     # Alembic environment configuration
├── script.py.mako             # Migration template
└── versions/                  # Migration files (auto-generated)
    ├── .gitkeep
    └── 20260119_0930_..._initial_migration_with_all_6_models.py
```

## Database Models Tracked

The following 6 SQLAlchemy models are tracked by Alembic:

1. **Run** - Training session records (centralized or federated)
2. **Client** - Federated learning client information
3. **Round** - Federated learning round data
4. **RunMetric** - Training metrics per epoch
5. **ServerEvaluation** - Server-side evaluation results
6. **ChatSession** - Chat conversation session data

## Common Commands

All commands should be run from the project root directory (`C:\Users\User\Projects\FYP2`).

### Check Current Migration Status

```bash
uv run alembic current
```

Shows the current database revision.

### View Migration History

```bash
uv run alembic history --verbose
```

Shows all available migrations and their status.

### Apply All Pending Migrations

```bash
uv run alembic upgrade head
```

Applies all migrations up to the latest version.

### Rollback One Migration

```bash
uv run alembic downgrade -1
```

Rolls back the most recent migration.

### Rollback to Specific Revision

```bash
uv run alembic downgrade <revision_id>
```

Rolls back to a specific migration revision.

### Generate New Migration (Auto-detect Changes)

```bash
uv run alembic revision --autogenerate -m "Description of changes"
```

Automatically generates a migration by comparing models to database schema.

### Create Empty Migration (Manual)

```bash
uv run alembic revision -m "Description of changes"
```

Creates an empty migration file for manual SQL operations.

### Generate SQL Without Applying

```bash
uv run alembic upgrade head --sql
```

Generates SQL statements without executing them (useful for review).

## Migration Workflow

### Development Workflow

1. **Modify SQLAlchemy models** in `federated_pneumonia_detection/src/boundary/models/`

2. **Generate migration**:
   ```bash
   uv run alembic revision --autogenerate -m "Add new column to Run model"
   ```

3. **Review generated migration** in `alembic/versions/`
   - Check upgrade() and downgrade() functions
   - Verify all changes are captured correctly
   - Add any manual operations if needed

4. **Apply migration**:
   ```bash
   uv run alembic upgrade head
   ```

5. **Test rollback** (optional):
   ```bash
   uv run alembic downgrade -1
   uv run alembic upgrade head
   ```

### Production Deployment

1. **Set environment variable**:
   ```bash
   export USE_ALEMBIC=true  # Linux/Mac
   set USE_ALEMBIC=true     # Windows
   ```

2. **Apply migrations**:
   ```bash
   uv run alembic upgrade head
   ```

3. **Verify schema**:
   ```bash
   uv run alembic current
   ```

## Migration vs create_all()

The project supports two schema management approaches:

### Approach 1: Alembic Migrations (Production)

```python
# Set in .env or environment
USE_ALEMBIC=true

# In code
from federated_pneumonia_detection.src.boundary.engine import get_engine
engine = get_engine()
# Tables managed by Alembic - don't call create_tables()
```

```bash
# Apply migrations
uv run alembic upgrade head
```

### Approach 2: Direct create_all() (Development)

```python
# Unset or set to false in .env
USE_ALEMBIC=false

# In code
from federated_pneumonia_detection.src.boundary.engine import create_tables
engine = create_tables()  # Creates tables directly
```

### Force Mode (Override)

```python
# Force create_all() even if USE_ALEMBIC=true
from federated_pneumonia_detection.src.boundary.engine import create_tables
engine = create_tables(force=True)
```

## Configuration

### Database Connection

The database URL is automatically loaded from `federated_pneumonia_detection.config.settings.Settings`:

- Uses `Settings().get_postgres_db_uri()`
- Configured via `.env` file (POSTGRES_DB_URI)
- No hardcoded credentials in alembic.ini

### Alembic Settings (alembic.ini)

- **script_location**: `alembic` (migration directory)
- **file_template**: Timestamp-based naming (YYYYMMDD_HHMM_revision_slug)
- **timezone**: UTC
- **truncate_slug_length**: 40 characters

## Initial Migration

The initial migration (`7036676c922c`) is a baseline migration that:

- Assumes tables already exist (created by `Base.metadata.create_all()`)
- Serves as a starting point for future migrations
- No operations in upgrade() (tables pre-existed)
- Downgrade() drops all 6 tables

This was created using:

```bash
uv run alembic stamp head
```

## Troubleshooting

### "Table already exists" error

If you get this error, the database has tables but Alembic doesn't know about them.

**Solution**: Stamp the database with the initial revision:

```bash
uv run alembic stamp head
```

### "No module named 'federated_pneumonia_detection'"

The project needs to be installed in editable mode.

**Solution**:

```bash
uv pip install -e .
```

### Autogenerate doesn't detect changes

Possible causes:
1. Models not imported in `alembic/env.py`
2. Base.metadata not properly configured
3. Database connection issues

**Solution**: Verify that:
- `from federated_pneumonia_detection.src.boundary.models import Base` is in `alembic/env.py`
- `target_metadata = Base.metadata` is set
- Database URL is correct in `.env`

### Migration conflicts

If you have multiple developers creating migrations:

1. **Pull latest migrations** from git
2. **Check for conflicts**:
   ```bash
   uv run alembic branches
   ```
3. **Merge conflicting branches** if needed (advanced)

## Best Practices

1. **Always review auto-generated migrations** before applying
2. **Test rollback** in development before deploying to production
3. **Use descriptive migration messages** (e.g., "Add recall_threshold to Run model")
4. **Never edit applied migrations** - create a new migration instead
5. **Commit migrations to version control** alongside model changes
6. **Run migrations in CI/CD pipelines** before deploying code
7. **Back up database** before major migrations in production

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/orm/tutorial.html)
- [Database Migration Best Practices](https://www.alembic.sqlalchemy.org/en/latest/tutorial.html#auto-generating-migrations)
