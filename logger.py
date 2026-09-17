"""Centralized logging so every module shares the same format."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Create a logger configured with the project's shared output format.

    Args:
        name: Name included in each log record, usually ``__name__``.

    Returns:
        A configured logger for the requested module name.

    Example:
        >>> module_logger = get_logger(__name__)
        >>> module_logger.info("Dataset loaded")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger