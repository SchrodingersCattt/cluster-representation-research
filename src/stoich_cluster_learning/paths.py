"""Path helpers for the standalone code release."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the root of the code-availability repository."""
    return Path(__file__).resolve().parents[2]
