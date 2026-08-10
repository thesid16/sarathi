"""Logging setup.

Kept in one place so every module gets the same format and so the benchmark
harness can quiet everything down to WARNING without hunting for loggers.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"


def configure(level: int | str | None = None, *, fmt: str = _DEFAULT_FORMAT) -> None:
    """Install the root handler. Idempotent; later calls only adjust the level."""
    global _CONFIGURED

    if level is None:
        level = os.environ.get("SARATHI_LOG", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("sarathi")
    root.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    # Module names arrive as "sarathi.sources.mjpeg"; keep them under the
    # "sarathi" root so one level change covers the whole package.
    return logging.getLogger(name if name.startswith("sarathi") else f"sarathi.{name}")
