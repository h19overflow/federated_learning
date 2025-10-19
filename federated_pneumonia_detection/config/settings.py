from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_PORT: int
    GEMINI_API_KEY: str
    GOOGLE_API_KEY: str
    BASE_LLM: str
    POSTGRES_DB_URI: str

    def get_postgres_db_uri(self) -> str:
        return self.POSTGRES_DB_URI
