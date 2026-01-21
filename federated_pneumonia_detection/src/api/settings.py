from pydantic_settings import BaseSettings
from pydantic import Field

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_VERSION = "v1"
DEFAULT_API_PREFIX = "/api"


class Settings(BaseSettings):
    """Settings for the API"""

    BASE_URL: str = Field(default=DEFAULT_BASE_URL)
    API_VERSION: str = Field(default=DEFAULT_API_VERSION)
    API_PREFIX: str = Field(default=DEFAULT_API_PREFIX)
