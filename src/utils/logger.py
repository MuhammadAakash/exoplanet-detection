"""
logger.py
=========
Centralised logging setup for the Exoplanet Vetting project.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting preprocessing pipeline…")

All modules should use this logger rather than bare print() statements so
that output can be uniformly redirected to file and console, and log levels
can be controlled from the central config.
"""

import logging
import sys
from pathlib import Path

from src.utils.config import LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance configured with:
      - StreamHandler → stdout (colour-friendly for notebooks/terminals)
      - FileHandler   → results/experiment.log (persistent record)

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # ------------------------------------------------------------------ #
    # Formatter — show timestamp, module, level and message
    # ------------------------------------------------------------------ #
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------ #
    # Console handler
    # ------------------------------------------------------------------ #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # File handler — write to results/experiment.log
    # ------------------------------------------------------------------ #
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning(f"Could not initialise file logging: {exc}")

    return logger
