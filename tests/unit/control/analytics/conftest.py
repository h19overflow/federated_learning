"""Pytest configuration and fixtures for analytics tests."""

import os
import sys
from unittest.mock import Mock

# Set required environment variables before any imports
os.environ["POSTGRES_DB"] = "test_db"
os.environ["POSTGRES_USER"] = "test_user"
os.environ["POSTGRES_PASSWORD"] = "test_password"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_DB_URI"] = (
    "postgresql://test_user:test_password@localhost:5432/test_db"
)
os.environ["GEMINI_API_KEY"] = "test_key"
os.environ["GOOGLE_API_KEY"] = "test_key"
os.environ["BASE_LLM"] = "gemini-1.5-flash"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "test_key"
