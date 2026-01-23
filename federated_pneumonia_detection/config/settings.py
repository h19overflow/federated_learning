from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database Configuration
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_PORT: int
    POSTGRES_DB_URI: str

    # API Configuration
    API_BASE_URL: str = Field(default="http://localhost")
    API_VERSION: str = Field(default="v1")
    API_PREFIX: str = Field(default="/api")

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:8080",
            "http://localhost:8081",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8081",
        ]
    )

    # WebSocket Configuration
    WEBSOCKET_HOST: str = Field(default="localhost")
    WEBSOCKET_PORT: int = Field(default=8765)

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")

    # Federated Learning Configuration
    FL_NUM_ROUNDS: int = Field(default=3)
    FL_NUM_CLIENTS: int = Field(default=2)
    FL_FRACTION_FIT: float = Field(default=1.0)
    FL_FRACTION_EVALUATE: float = Field(default=1.0)

    # AI/LLM Configuration
    GEMINI_API_KEY: str
    GOOGLE_API_KEY: str
    BASE_LLM: str

    # LangSmith Configuration
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                import json

                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("WEBSOCKET_PORT")
    @classmethod
    def validate_websocket_port(cls, v: int) -> int:
        if not 1024 <= v <= 65535:
            raise ValueError("WEBSOCKET_PORT must be between 1024 and 65535")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    def get_postgres_db_uri(self) -> str:
        return self.POSTGRES_DB_URI

    @property
    def full_api_url(self) -> str:
        """Compute full API URL"""
        return f"{self.API_BASE_URL}{self.API_PREFIX}/{self.API_VERSION}"

    @property
    def websocket_uri(self) -> str:
        """Compute WebSocket URI"""
        return f"ws://{self.WEBSOCKET_HOST}:{self.WEBSOCKET_PORT}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)"""
    return Settings()
