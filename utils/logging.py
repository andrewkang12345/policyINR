import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

_CONFIGURED = False


def get_logger(name: str = "inr") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(os.environ.get("INR_LOG_LEVEL", "INFO"))
        _CONFIGURED = True
    return logger


class JSONLLogger:
    """Append-only JSONL metric logger."""

    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")
