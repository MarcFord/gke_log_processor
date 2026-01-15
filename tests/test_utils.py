"""Tests for utility functions."""

from datetime import datetime, timezone

from gke_log_processor.core.utils import utc_now


class TestUtcNow:
    """Test utc_now utility function."""

    def test_utc_now_returns_datetime(self):
        """Test that utc_now returns a datetime object."""
        result = utc_now()
        assert isinstance(result, datetime)

    def test_utc_now_returns_utc_timezone(self):
        """Test that utc_now returns a UTC timezone-aware datetime."""
        result = utc_now()
        assert result.tzinfo == timezone.utc

    def test_utc_now_is_recent(self):
        """Test that utc_now returns a recent timestamp."""
        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)

        # Should be between before and after calls
        assert before <= result <= after

    def test_utc_now_multiple_calls_progress(self):
        """Test that multiple calls to utc_now progress in time."""
        first = utc_now()
        # Small delay to ensure time progression
        import time

        time.sleep(0.001)
        second = utc_now()

        assert second > first
