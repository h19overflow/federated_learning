import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a wrapper around logging.getLogger to maintain a consistent interface.
    """
    return logging.getLogger(name)


def setup_logger(name: str) -> logging.Logger:
    """
    Legacy setup_logger. Now just calls get_logger as configuration is
    handled centrally.
    """
    return get_logger(name)
