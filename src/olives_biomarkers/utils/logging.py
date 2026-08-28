"""Consistent logging setup for scripts and notebooks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class LoggerFactory:
    """Creates loggers that write to stdout and optionally to a run log file."""

    FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    DATEFMT = "%H:%M:%S"
    _configured: set[str] = set()

    @classmethod
    def get(
        cls,
        name: str = "olives",
        level: int | str = logging.INFO,
        log_file: str | Path | None = None,
    ) -> logging.Logger:
        """Return a configured logger, attaching handlers only once per name."""
        logger = logging.getLogger(name)
        if name in cls._configured:
            return logger

        logger.setLevel(level)
        logger.propagate = False
        formatter = logging.Formatter(cls.FORMAT, datefmt=cls.DATEFMT)

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._configured.add(name)
        return logger

    @classmethod
    def reset(cls) -> None:
        """Drop handler bookkeeping; used by tests and notebook reloads."""
        for name in cls._configured:
            logging.getLogger(name).handlers.clear()
        cls._configured.clear()
