"""
Centralized logging configuration for the application.

Configures third-party library log levels and provides consistent
logger initialization across the codebase.
"""

import logging
import os


def configure_logging() -> None:
    """
    Configure application logging with proper third-party library silencing.

    This function should be called once at application startup (in main.py's
    lifespan handler) to configure all logging behavior.

    Log Level Strategy:
        - INFO: Critical business events (start/end of operations, errors, warnings)
        - DEBUG: Detailed flow, chunk counts, intermediate steps
        - WARNING: Graceful degradations, non-critical issues
        - ERROR: Critical failures

    Third-Party Libraries:
        - langchain_google_genai: WARNING (silences API key warnings)
        - google.genai: WARNING (silences AFC messages)
        - uvicorn.access: WARNING (silences HTTP request logs)
        - uvicorn: INFO (keep important server logs)
    """
    # Get log level from environment variable, default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger format
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(message)s - %(filename)s - %(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Silence third-party library noise
    _configure_third_party_loggers()

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={log_level_str}, "
        f"third-party libs silenced (langchain_google_genai, google.genai, uvicorn.access)"
    )


def _configure_third_party_loggers() -> None:
    """
    Configure third-party library loggers to reduce noise.

    These libraries log too much at INFO level, drowning out important
    application logs. We set them to WARNING or higher.
    """
    # LangChain Google Generative AI: Silences "Both GOOGLE_API_KEY and GEMINI_API_KEY"
    logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

    # Google GenAI: Silences "AFC is enabled with max remote calls: 10"
    logging.getLogger("google.genai").setLevel(logging.WARNING)

    # Uvicorn Access: Silences HTTP request logs (GET /chat/history, etc.)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Keep uvicorn server logs at INFO (important startup/shutdown messages)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is the preferred way to get loggers in the codebase.

    Args:
        name: Usually __name__ for module-level loggers

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
