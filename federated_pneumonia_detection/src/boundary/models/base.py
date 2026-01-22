"""
SQLAlchemy declarative base for all ORM models.

This module defines the Base class used by all models to prevent circular imports.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
