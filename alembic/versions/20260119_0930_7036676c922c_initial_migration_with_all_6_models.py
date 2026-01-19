"""Initial migration with all 6 models

This migration captures the existing database schema for the 6 core models:
- Run (training sessions)
- Client (federated learning clients)
- Round (federated rounds)
- RunMetric (training metrics)
- ServerEvaluation (server-side evaluations)
- ChatSession (chat conversation sessions)

Note: This migration assumes tables already exist (created by Base.metadata.create_all).
It serves as a baseline for future migrations using Alembic.

Revision ID: 7036676c922c
Revises:
Create Date: 2026-01-19 09:30:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7036676c922c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Apply schema changes to upgrade the database.

    This is a baseline migration - tables already exist from Base.metadata.create_all().
    No operations needed for upgrade.
    """
    # Baseline migration - no operations needed
    # Tables were created by Base.metadata.create_all() in engine.py
    pass


def downgrade() -> None:
    """
    Revert schema changes to downgrade the database.

    For a baseline migration, downgrade drops all tables.
    WARNING: This will delete all data in these tables!
    """
    # Drop all tables in reverse dependency order
    op.drop_table('server_evaluations')
    op.drop_table('rounds')
    op.drop_table('run_metrics')
    op.drop_table('clients')
    op.drop_table('chat_sessions')
    op.drop_table('runs')
