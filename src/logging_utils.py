from __future__ import annotations

import logging


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
