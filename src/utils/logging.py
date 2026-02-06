"""Project logger configuration based on Loguru."""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def _resolve_stdout_sink():
    """Return a robust log sink for environments with wrapped stdout streams."""
    sink = sys.stdout
    enc = getattr(sink, "encoding", None)
    if not enc or enc.lower() != "utf-8":
        try:
            reconfigure = getattr(sink, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            # Wrapped notebook streams may not support reconfiguration.
            pass

        enc = getattr(sink, "encoding", None)
        if not enc:
            return lambda message: print(message, end="", file=sys.stdout)

    return sink


def _configure_logger(level: str = "INFO") -> None:
    """Configure global Loguru handlers once.

    Parameters
    ----------
    level : str, default="INFO"
        Minimum log level for stdout output.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    sink = _resolve_stdout_sink()
    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    logger.add(sink, level=level, format=fmt, enqueue=False, backtrace=False)
    _CONFIGURED = True


def get_logger(name: str = __name__):
    """Return a configured logger instance.

    Parameters
    ----------
    name : str, default=__name__
        Logger binding name.

    Returns
    -------
    loguru.Logger
        Bound logger instance.
    """
    _configure_logger()
    return logger.bind(name=name)
