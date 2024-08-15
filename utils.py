"""
Utility functions for the Chat99 project.
"""

import logging
from config import LOG_LEVEL, DEBUG


def setup_logging():
    """Set up logging for the project."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length."""
    return text[:max_length] + "..." if len(text) > max_length else text


# Add more utility functions as needed
