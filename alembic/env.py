"""
Alembic environment configuration for federated pneumonia detection.

This module configures the Alembic migration environment to work with:
- SQLAlchemy models from federated_pneumonia_detection.src.boundary.models
- Database URL from federated_pneumonia_detection.config.settings.Settings
- Both offline (SQL generation) and online (direct DB) migration modes

CRITICAL RULES:
- ALWAYS import Base from boundary.models to ensure all models are registered
- ALWAYS use Settings().get_postgres_db_uri() for database connection
- NEVER hardcode database credentials
- ALWAYS include target_metadata = Base.metadata for autogenerate support
"""

import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import Settings for database URL
from federated_pneumonia_detection.config.settings import Settings

# Import Base - this registers all models with SQLAlchemy metadata
# CRITICAL: This import ensures all 6 models are included in autogenerate
from federated_pneumonia_detection.src.boundary.models import Base

# Initialize settings
settings = Settings()

# Alembic Config object - provides access to alembic.ini values
config = context.config

# Setup Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# Target metadata for autogenerate support
# This tells Alembic which models to track for schema changes
target_metadata = Base.metadata

# Override sqlalchemy.url from alembic.ini with Settings value
# This ensures migrations always use the correct database URL from environment
config.set_main_option("sqlalchemy.url", settings.get_postgres_db_uri())


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    In offline mode, Alembic generates SQL scripts without connecting to the database.
    This is useful for:
    - Generating migration SQL for manual review
    - Applying migrations on systems without direct database access
    - Creating deployment scripts for production environments

    The SQL is printed to stdout for redirection to a file.

    Example:
        alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    logger.info("Running migrations in OFFLINE mode (SQL generation)")
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In online mode, Alembic connects directly to the database and applies migrations.
    This is the standard mode for development and automated deployments.

    Connection pooling is disabled (poolclass=pool.NullPool) because Alembic
    only needs a single connection for migration operations.

    Example:
        alembic upgrade head
    """
    # Create engine from alembic.ini settings + overridden URL
    # NullPool: No connection pooling (migrations don't need it)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    logger.info("Running migrations in ONLINE mode (direct database connection)")
    logger.info(f"Target database: {settings.POSTGRES_DB}")

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare types to detect column type changes
            compare_type=True,
            # Compare server defaults to detect default value changes
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# Determine migration mode and execute
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
