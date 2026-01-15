"""Utility functions for the GKE Log Processor."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Get the current UTC datetime.

    This function replaces the deprecated datetime.utcnow() with
    the recommended timezone-aware approach.

    Returns:
        datetime: Current UTC datetime with timezone information
    """
    return datetime.now(timezone.utc)
