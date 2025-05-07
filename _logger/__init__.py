"""
Centralized logging configuration for clembench.
"""

import logging

import colorlog


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting and configuration.

    Args:
        name: The name of the logger (typically __name__ of the module)

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create console handler with colored formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

    return logger
