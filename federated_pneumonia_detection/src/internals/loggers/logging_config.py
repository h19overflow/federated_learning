import logging
import logging.config
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict

import yaml

# Context variable to store the request ID
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """
    Logging filter that injects the current request ID into the log record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(
    config_path: str = "federated_pneumonia_detection/config/logging.yaml",
) -> None:
    """
    Configure logging using a YAML configuration file.

    Args:
        config_path: Path to the logging configuration file relative to the
            repository root.
    """
    # Determine the repository root (assuming this file is in src/internals/loggers/)
    # federated_pneumonia_detection/src/internals/loggers/logging_config.py
    # -> 5 levels up to root
    root_dir = Path(__file__).resolve().parents[4]
    abs_config_path = root_dir / config_path

    if not abs_config_path.exists():
        # Fallback to current working directory if not found relative to root
        abs_config_path = Path(config_path).resolve()

    if not abs_config_path.exists():
        print(
            f"Logging configuration file not found at {abs_config_path}. "
            f"Using basic configuration."
        )
        logging.basicConfig(level=logging.INFO)
        return

    try:
        with open(abs_config_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Inject RequestIdFilter into the configuration
        if "filters" not in config:
            config["filters"] = {}

        config["filters"]["request_id"] = {
            "()": "federated_pneumonia_detection.src.internals.loggers."
            "logging_config.RequestIdFilter"
        }

        # Add the filter to the console handler
        if "handlers" in config and "console" in config["handlers"]:
            if "filters" not in config["handlers"]["console"]:
                config["handlers"]["console"]["filters"] = []
            if "request_id" not in config["handlers"]["console"]["filters"]:
                config["handlers"]["console"]["filters"].append("request_id")

        logging.config.dictConfig(config)

        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured successfully from {abs_config_path}")

    except Exception as e:
        print(f"Failed to configure logging from {abs_config_path}: {e}")
        logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    """
    return logging.getLogger(name)
