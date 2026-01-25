import logging

logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
