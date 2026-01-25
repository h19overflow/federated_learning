"""
Unit tests for API settings configuration.

Tests environment variable loading and default values.
"""

from federated_pneumonia_detection.config.settings import get_settings


class TestSettingsDefaults:
    """Test default values for Settings class."""

    def test_default_base_url(self):
        """Settings should have default API_BASE_URL."""
        settings = get_settings()
        assert settings.API_BASE_URL == "http://localhost"

    def test_default_api_version(self):
        """Settings should have default API_VERSION."""
        settings = get_settings()
        assert settings.API_VERSION == "v1"

    def test_default_api_prefix(self):
        """Settings should have default API_PREFIX."""
        settings = get_settings()
        assert settings.API_PREFIX == "/api"

    def test_default_websocket_config(self):
        """Settings should have default WebSocket configuration."""
        settings = get_settings()
        assert settings.WEBSOCKET_HOST == "localhost"
        assert settings.WEBSOCKET_PORT == 8765

    def test_default_cors_origins(self):
        """Settings should have default CORS origins."""
        settings = get_settings()
        assert len(settings.CORS_ORIGINS) > 0
        assert "http://localhost:5173" in settings.CORS_ORIGINS

    def test_default_log_level(self):
        """Settings should have default log level."""
        settings = get_settings()
        assert settings.LOG_LEVEL == "INFO"

    def test_full_api_url_property(self):
        """Settings should compute full API URL correctly."""
        settings = get_settings()
        expected_url = (
            f"{settings.API_BASE_URL}{settings.API_PREFIX}/{settings.API_VERSION}"
        )
        assert settings.full_api_url == expected_url

    def test_websocket_uri_property(self):
        """Settings should compute WebSocket URI correctly."""
        settings = get_settings()
        expected_uri = f"ws://{settings.WEBSOCKET_HOST}:{settings.WEBSOCKET_PORT}"
        assert settings.websocket_uri == expected_uri


class TestSettingsFromEnv:
    """Test Settings loading from environment variables."""

    def test_api_base_url_from_env(self, monkeypatch):
        """Settings should load API_BASE_URL from environment."""
        monkeypatch.setenv("API_BASE_URL", "https://example.com")
        # Clear cache to get fresh instance
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.API_BASE_URL == "https://example.com"

    def test_api_version_from_env(self, monkeypatch):
        """Settings should load API_VERSION from environment."""
        monkeypatch.setenv("API_VERSION", "v2")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.API_VERSION == "v2"

    def test_api_prefix_from_env(self, monkeypatch):
        """Settings should load API_PREFIX from environment."""
        monkeypatch.setenv("API_PREFIX", "/api/v2")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.API_PREFIX == "/api/v2"

    def test_multiple_env_vars(self, monkeypatch):
        """Settings should load multiple environment variables."""
        monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("API_VERSION", "v3")
        monkeypatch.setenv("API_PREFIX", "/api/v3")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.API_BASE_URL == "https://api.example.com"
        assert settings.API_VERSION == "v3"
        assert settings.API_PREFIX == "/api/v3"


class TestSettingsValidation:
    """Test Settings validation and type checking."""

    def test_api_base_url_must_be_string(self, monkeypatch):
        """API_BASE_URL must be a string."""
        monkeypatch.setenv("API_BASE_URL", "12345")
        get_settings.cache_clear()
        # Pydantic will convert to string, this tests type coercion
        settings = get_settings()
        assert isinstance(settings.API_BASE_URL, str)

    def test_api_version_must_be_string(self, monkeypatch):
        """API_VERSION must be a string."""
        monkeypatch.setenv("API_VERSION", "v1")
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings.API_VERSION, str)

    def test_api_prefix_must_be_string(self, monkeypatch):
        """API_PREFIX must be a string."""
        monkeypatch.setenv("API_PREFIX", "/api")
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings.API_PREFIX, str)

    def test_log_level_validation(self, monkeypatch):
        """LOG_LEVEL must be one of the valid levels."""
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.LOG_LEVEL in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_websocket_port_validation(self, monkeypatch):
        """WEBSOCKET_PORT must be between 1024 and 65535."""
        get_settings.cache_clear()
        settings = get_settings()
        assert 1024 <= settings.WEBSOCKET_PORT <= 65535


class TestSettingsSingleton:
    """Test Settings behaves as singleton using get_settings()."""

    def test_get_settings_returns_singleton(self):
        """get_settings() should return the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_settings_instances_are_equal(self):
        """Multiple get_settings() calls should return same values."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1.API_BASE_URL == settings2.API_BASE_URL
        assert settings1.API_VERSION == settings2.API_VERSION
        assert settings1.API_PREFIX == settings2.API_PREFIX


class TestSettingsIntegration:
    """Test Settings integration with environment."""

    def test_env_cleared_uses_defaults(self, monkeypatch):
        """Clearing env vars should fall back to defaults."""
        monkeypatch.delenv("API_BASE_URL", raising=False)
        monkeypatch.delenv("API_VERSION", raising=False)
        monkeypatch.delenv("API_PREFIX", raising=False)
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.API_BASE_URL == "http://localhost"
        assert settings.API_VERSION == "v1"
        assert settings.API_PREFIX == "/api"
