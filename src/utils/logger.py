"""
PRISM Logger
"""

import logging
import os
import sys
from datetime import datetime


class AutoFlushFileHandler(logging.FileHandler):
    """FileHandler that flushes to disk immediately after every emitted log record."""
    def emit(self, record):
        super().emit(record)
        self.flush()


class AutoFlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes immediately after every emitted log record."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(
    name: str = "prism",
    log_dir: str = "results/logs",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Create a logger that writes to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = AutoFlushStreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = AutoFlushFileHandler(
            os.path.join(log_dir, f"{name}_{timestamp}.log")
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
