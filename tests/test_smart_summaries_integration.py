"""Tests for smart summaries integration in LogAnalysisEngine."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from gke_log_processor.ai.analyzer import LogAnalysisEngine
from gke_log_processor.ai.summarizer import (
    KeyInsight,
    LogSummaryReport,
    SummarizerConfig,
    SummaryType,
    TimeWindowSize,
    TimeWindowSummary,
    TrendAnalysis,
    TrendDirection,
)
from gke_log_processor.core.models import LogEntry, LogLevel, SeverityLevel
from gke_log_processor.core.utils import utc_now


@pytest.fixture
def sample_log_entries():
    """Create sample log entries for testing."""
    base_time = utc_now()

    return [
        LogEntry(
            timestamp=base_time,
            level=LogLevel.INFO,
            message="Service started successfully",
            source="app",
            pod_name="web-server-1",
            namespace="default",
            cluster="test-cluster",
            container_name="app",
            raw_message="Service started successfully"
        ),
        LogEntry(
            timestamp=base_time + timedelta(minutes=2),
            level=LogLevel.WARNING,
            message="High memory usage detected",
            source="app",
            pod_name="web-server-1",
            namespace="default",
            cluster="test-cluster",
            container_name="app",
            raw_message="High memory usage detected"
        ),
        LogEntry(
            timestamp=base_time + timedelta(minutes=5),
            level=LogLevel.ERROR,
            message="Database connection failed",
            source="app",
            pod_name="web-server-2",
            namespace="default",
            cluster="test-cluster",
            container_name="app",
            raw_message="Database connection failed"
        ),
    ]


@pytest.fixture
def mock_summary_report():
    """Create mock summary report for testing."""
    base_time = utc_now()

    window_summary = TimeWindowSummary(
        start_time=base_time,
        end_time=base_time + timedelta(minutes=15),
        window_size=TimeWindowSize.FIFTEEN_MINUTES,
        log_count=5,
        error_count=2,
        warning_count=1,
        overall_severity=SeverityLevel.MEDIUM,
        top_errors=["Database connection failed"],
        summary_text="System experiencing connectivity issues",
        key_events=["Service started", "High memory detected"]
    )

    key_insight = KeyInsight(
        title="Database Connectivity Issues",
        description="Multiple database connection failures detected",
        severity=SeverityLevel.HIGH,
        confidence=0.9,
        affected_windows=[base_time],
        recommendation="Check database server health and network connectivity"
    )

    trend_analysis = TrendAnalysis(
        metric_name="error_rate",
        direction=TrendDirection.INCREASING,
        confidence=0.8,
        change_percentage=45.0,
        significant_changes=["Error rate increased by 45%"],
        recommendation="Investigate root cause of increasing error rate"
    )

    return LogSummaryReport(
        time_range_start=base_time,
        time_range_end=base_time + timedelta(minutes=20),
        total_log_entries=5,
        window_summaries=[window_summary],
        key_insights=[key_insight],
        trend_analyses=[trend_analysis],
        executive_summary="System experiencing moderate issues with database connectivity",
        recommendations=[
            "Check database server health and network connectivity",
            "Investigate root cause of increasing error rate",
            "Monitor memory usage patterns"
        ]
    )


class TestSmartSummaryIntegration:
    """Test smart summary integration functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create LogAnalysisEngine instance for testing."""
        return LogAnalysisEngine()

    @pytest.mark.asyncio
    async def test_generate_smart_summary_default(self, analyzer, sample_log_entries):
        """Test generating smart summary with default configuration."""
        # Mock the summarizer
        analyzer.summarizer.summarize_logs = AsyncMock(return_value=Mock())

        result = await analyzer.generate_smart_summary(sample_log_entries)

        # Verify summarizer was called with correct parameters
        analyzer.summarizer.summarize_logs.assert_called_once_with(
            sample_log_entries, analyzer.summarizer.config
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, analyzer, sample_log_entries, mock_summary_report):
        """Test generating executive-level summary."""
        analyzer.summarizer.summarize_logs = AsyncMock(return_value=mock_summary_report)

        result = await analyzer.generate_executive_summary(
            sample_log_entries, TimeWindowSize.ONE_HOUR
        )

        # Verify the result is the executive summary text
        assert result == mock_summary_report.executive_summary

        # Verify configuration was set correctly
        call_args = analyzer.summarizer.summarize_logs.call_args
        config = call_args[0][1]
        assert config.window_size == TimeWindowSize.ONE_HOUR
        assert config.summary_type == SummaryType.EXECUTIVE

    @pytest.mark.asyncio
    async def test_generate_technical_summary(self, analyzer, sample_log_entries, mock_summary_report):
        """Test generating technical summary."""
        analyzer.summarizer.summarize_logs = AsyncMock(return_value=mock_summary_report)

        result = await analyzer.generate_technical_summary(
            sample_log_entries, TimeWindowSize.FIFTEEN_MINUTES
        )

        # Verify the structure of technical summary
        assert isinstance(result, dict)
        assert "time_range" in result
        assert "overview" in result
        assert "window_summaries" in result
        assert "key_insights" in result
        assert "executive_summary" in result

        # Verify overview details
        assert result["overview"]["total_logs"] == mock_summary_report.total_log_entries

    def test_get_available_window_sizes(self, analyzer):
        """Test getting available time window sizes."""
        sizes = analyzer.get_available_window_sizes()

        assert isinstance(sizes, list)
        assert len(sizes) > 0
        assert TimeWindowSize.ONE_MINUTE in sizes
        assert TimeWindowSize.FIFTEEN_MINUTES in sizes
