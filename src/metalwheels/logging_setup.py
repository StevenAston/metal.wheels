"""Central logging configuration.

Levels follow the project convention: DEBUG for trace, INFO for milestones,
WARNING for recoverable issues (e.g. a fit evaluated outside its source data
range), ERROR for failures.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; later calls only adjust the level."""
    global _CONFIGURED
    root = logging.getLogger()
    if not _CONFIGURED:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(handler)
        _CONFIGURED = True
    root.setLevel(level.upper())
    logging.getLogger(__name__).debug("Logging configured at %s", level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (thin wrapper so call sites stay uniform)."""
    return logging.getLogger(name)
