from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str | Path | None = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("psoct_processing")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
