"""
Unit tests for API settings configuration.

Tests environment variable loading and default values.
"""

from federated_pneumonia_detection.src.api.settings import (
    DEFAULT_API_PREFIX,
    DEFAULT_API_VERSION,
    DEFAULT_BASE_URL,
    Settings,
)


class TestSettingsDefaults:
    """Test default values for Settings class."""

    def test_default_base_url(self):
        """Settings should have default BASE_URL."""
        settings = Settings()
        assert settings.BASE_URL == DEFAULT_BASE_URL
        assert settings.BASE_URL == "http://localhost:8000"

    def test_default_api_version(self):
        """Settings should have default API_VERSION."""
        settings = Settings()
        assert settings.API_VERSION == DEFAULT_API_VERSION
        assert settings.API_VERSION == "v1"

    def test_default_api_prefix(self):
        """Settings should have default API_PREFIX."""
        settings = Settings()
        assert settings.API_PREFIX == DEFAULT_API_PREFIX
        assert settings.API_PREFIX == "/api"


class TestSettingsFromEnv:
    """Test Settings loading from environment variables."""

    def test_base_url_from_env(self, monkeypatch):
        """Settings should load BASE_URL from environment."""
        monkeypatch.setenv("BASE_URL", "https://example.com")
        settings = Settings()
        assert settings.BASE_URL == "https://example.com"

    def test_api_version_from_env(self, monkeypatch):
        """Settings should load API_VERSION from environment."""
        monkeypatch.setenv("API_VERSION", "v2")
        settings = Settings()
        assert settings.API_VERSION == "v2"

    def test_api_prefix_from_env(self, monkeypatch):
        """Settings should load API_PREFIX from environment."""
        monkeypatch.setenv("API_PREFIX", "/api/v2")
        settings = Settings()
        assert settings.API_PREFIX == "/api/v2"

    def test_multiple_env_vars(self, monkeypatch):
        """Settings should load multiple environment variables."""
        monkeypatch.setenv("BASE_URL", "https://api.example.com")
        monkeypatch.setenv("API_VERSION", "v3")
        monkeypatch.setenv("API_PREFIX", "/api/v3")
        settings = Settings()
        assert settings.BASE_URL == "https://api.example.com"
        assert settings.API_VERSION == "v3"
        assert settings.API_PREFIX == "/api/v3"


class TestSettingsValidation:
    """Test Settings validation and type checking."""

    def test_base_url_must_be_string(self, monkeypatch):
        """BASE_URL must be a string."""
        monkeypatch.setenv("BASE_URL", "12345")
        # Pydantic will convert to string, this tests type coercion
        settings = Settings()
        assert isinstance(settings.BASE_URL, str)

    def test_api_version_must_be_string(self, monkeypatch):
        """API_VERSION must be a string."""
        monkeypatch.setenv("API_VERSION", "v1")
        settings = Settings()
        assert isinstance(settings.API_VERSION, str)

    def test_api_prefix_must_be_string(self, monkeypatch):
        """API_PREFIX must be a string."""
        monkeypatch.setenv("API_PREFIX", "/api")
        settings = Settings()
        assert isinstance(settings.API_PREFIX, str)


class TestSettingsSingleton:
    """Test Settings behaves as singleton (though it's not enforced)."""

    def test_settings_instances_are_equal(self):
        """Multiple Settings instances should have same defaults."""
        settings1 = Settings()
        settings2 = Settings()
        assert settings1.BASE_URL == settings2.BASE_URL
        assert settings1.API_VERSION == settings2.API_VERSION
        assert settings1.API_PREFIX == settings2.API_PREFIX


class TestSettingsIntegration:
    """Test Settings integration with environment."""

    def test_env_cleared_uses_defaults(self, monkeypatch):
        """Clearing env vars should fall back to defaults."""
        monkeypatch.delenv("BASE_URL", raising=False)
        monkeypatch.delenv("API_VERSION", raising=False)
        monkeypatch.delenv("API_PREFIX", raising=False)
        settings = Settings()
        assert settings.BASE_URL == DEFAULT_BASE_URL
        assert settings.API_VERSION == DEFAULT_API_VERSION
        assert settings.API_PREFIX == DEFAULT_API_PREFIX

    def test_empty_string_env_uses_defaults(self, monkeypatch):
        """Empty string env vars should use defaults."""
        monkeypatch.setenv("BASE_URL", "")
        monkeypatch.setenv("API_VERSION", "")
        monkeypatch.setenv("API_PREFIX", "")
        settings = Settings()
        # Pydantic defaults override empty strings
        assert settings.BASE_URL == DEFAULT_BASE_URL
        assert settings.API_VERSION == DEFAULT_API_VERSION
        assert settings.API_PREFIX == DEFAULT_API_PREFIX
