"""Generic helper functions."""

from __future__ import annotations


def ensure_directory(path):
    """Create a directory if it does not exist."""
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)
    return path
