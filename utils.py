import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os


def setup_logging(script_filename, profile_name, user_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "../logs")

    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_{profile_name}_{user_name}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        f"%(asctime)s.%(msecs)03d - {script_filename} - {profile_name} - {user_name} - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
