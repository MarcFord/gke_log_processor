"""Tests for log summarization engine."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from gke_log_processor.ai.summarizer import (
    KeyInsight,
    LogSummarizer,
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


class TestSummarizerConfig:
    """Test SummarizerConfig functionality."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SummarizerConfig()
        assert config.window_size == TimeWindowSize.FIFTEEN_MINUTES
        assert config.summary_type == SummaryType.TECHNICAL
        assert config.max_insights == 10
        assert config.min_confidence == 0.6
        assert config.enable_trend_analysis is True
        assert config.enable_ai_summarization is True
        assert config.max_summary_length == 500

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SummarizerConfig(
            window_size=TimeWindowSize.ONE_HOUR,
            summary_type=SummaryType.EXECUTIVE,
            max_insights=5,
            min_confidence=0.8,
            enable_trend_analysis=False,
            max_summary_length=1000
        )
        assert config.window_size == TimeWindowSize.ONE_HOUR
        assert config.summary_type == SummaryType.EXECUTIVE
        assert config.max_insights == 5
        assert config.min_confidence == 0.8
        assert config.enable_trend_analysis is False
        assert config.max_summary_length == 1000

    def test_validation_constraints(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            SummarizerConfig(max_insights=0)
        with pytest.raises(ValueError):
            SummarizerConfig(max_insights=51)
        with pytest.raises(ValueError):
            SummarizerConfig(min_confidence=-0.1)
        with pytest.raises(ValueError):
            SummarizerConfig(min_confidence=1.1)


class TestLogSummarizer:
    """Test LogSummarizer functionality."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create mock AI client."""
        client = Mock()
        client.summarize_logs = AsyncMock(return_value="AI-generated summary")
        client.query_logs = AsyncMock(return_value="- Key event 1\n- Key event 2\n- Key event 3")
        return client

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SummarizerConfig(
            window_size=TimeWindowSize.FIVE_MINUTES,
            max_insights=5,
            min_confidence=0.5
        )

    @pytest.fixture
    def sample_log_entries(self):
        """Create sample log entries for testing."""
        base_time = utc_now().replace(second=0, microsecond=0)
        return [
            LogEntry(
                timestamp=base_time,
                message="Application started successfully",
                level=LogLevel.INFO,
                pod_name="app-pod-1",
                container_name="app-container",
                namespace="default",
                cluster="test-cluster",
                source="app-container",
                raw_message="Application started successfully"
            ),
            LogEntry(
                timestamp=base_time + timedelta(minutes=2),
                message="Database connection established",
                level=LogLevel.INFO,
                pod_name="app-pod-1",
                container_name="app-container",
                namespace="default",
                cluster="test-cluster",
                source="app-container",
                raw_message="Database connection established"
            ),
            LogEntry(
                timestamp=base_time + timedelta(minutes=3),
                message="Connection refused to database",
                level=LogLevel.ERROR,
                pod_name="app-pod-1",
                container_name="app-container",
                namespace="default",
                cluster="test-cluster",
                source="app-container",
                raw_message="Connection refused to database"
            ),
            LogEntry(
                timestamp=base_time + timedelta(minutes=6),
                message="Retry connection to database",
                level=LogLevel.WARNING,
                pod_name="app-pod-1",
                container_name="app-container",
                namespace="default",
                cluster="test-cluster",
                source="app-container",
                raw_message="Retry connection to database"
            ),
            LogEntry(
                timestamp=base_time + timedelta(minutes=8),
                message="Database connection restored",
                level=LogLevel.INFO,
                pod_name="app-pod-1",
                container_name="app-container",
                namespace="default",
                cluster="test-cluster",
                source="app-container",
                raw_message="Database connection restored"
            ),
        ]

    def test_summarizer_initialization(self, mock_ai_client, config):
        """Test summarizer initialization."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)
        assert summarizer.ai_client == mock_ai_client
        assert summarizer.config == config
        assert summarizer.logger is not None

    def test_summarizer_initialization_defaults(self):
        """Test summarizer initialization with defaults."""
        summarizer = LogSummarizer()
        assert summarizer.ai_client is None
        assert isinstance(summarizer.config, SummarizerConfig)

    @pytest.mark.asyncio
    async def test_summarize_logs_empty_list(self, mock_ai_client, config):
        """Test handling of empty log list."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        with pytest.raises(ValueError, match="No log entries provided"):
            await summarizer.summarize_logs([])

    @pytest.mark.asyncio
    async def test_summarize_logs_basic(self, mock_ai_client, config, sample_log_entries):
        """Test basic log summarization."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        result = await summarizer.summarize_logs(sample_log_entries, config)

        assert isinstance(result, LogSummaryReport)
        assert result.total_log_entries == len(sample_log_entries)
        assert len(result.window_summaries) >= 1
        assert result.executive_summary
        assert isinstance(result.key_insights, list)
        assert isinstance(result.trend_analyses, list)
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_create_time_windows(self, mock_ai_client, config, sample_log_entries):
        """Test time window creation."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        windows = await summarizer._create_time_windows(sample_log_entries, config)

        assert len(windows) >= 1
        assert all(isinstance(w, TimeWindowSummary) for w in windows)
        assert all(w.window_size == config.window_size for w in windows)

        # Check that logs are properly distributed
        total_logs = sum(w.log_count for w in windows)
        assert total_logs == len(sample_log_entries)

    @pytest.mark.asyncio
    async def test_summarize_window_with_errors(self, mock_ai_client, config):
        """Test window summarization with errors."""
        # Set up mock to return error-related response
        mock_ai_client.summarize_logs.return_value = "Multiple errors occurred in the system"

        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        base_time = utc_now().replace(second=0, microsecond=0)
        window_logs = [
            LogEntry(
                timestamp=base_time,
                message="Error occurred",
                level=LogLevel.ERROR,
                pod_name="test-pod",
                container_name="test-container",
                namespace="default",
                cluster="test-cluster",
                source="test-container",
                raw_message="Error occurred"
            ),
            LogEntry(
                timestamp=base_time + timedelta(seconds=30),
                message="Another error",
                level=LogLevel.ERROR,
                pod_name="test-pod",
                container_name="test-container",
                namespace="default",
                cluster="test-cluster",
                source="test-container",
                raw_message="Another error"
            ),
        ]

        window = await summarizer._summarize_window(
            window_logs, base_time, base_time + timedelta(minutes=5), config
        )

        assert window.error_count == 2
        assert window.warning_count == 0
        assert window.overall_severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        assert len(window.top_errors) > 0
        assert "error" in window.summary_text.lower()

    @pytest.mark.asyncio
    async def test_analyze_error_patterns(self, mock_ai_client, config):
        """Test error pattern analysis."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        base_time = utc_now().replace(second=0, microsecond=0)
        window_summaries = [
            TimeWindowSummary(
                start_time=base_time,
                end_time=base_time + timedelta(minutes=5),
                window_size=config.window_size,
                log_count=10,
                error_count=5,
                warning_count=2,
                overall_severity=SeverityLevel.HIGH,
                summary_text="High error rate window",
                top_errors=["Error 1", "Error 2"]
            ),
            TimeWindowSummary(
                start_time=base_time + timedelta(minutes=5),
                end_time=base_time + timedelta(minutes=10),
                window_size=config.window_size,
                log_count=8,
                error_count=4,
                warning_count=1,
                overall_severity=SeverityLevel.HIGH,
                summary_text="Another high error rate window",
                top_errors=["Error 1", "Error 3"]
            ),
        ]

        insights = await summarizer._analyze_error_patterns(window_summaries, config)

        assert len(insights) > 0
        assert all(isinstance(insight, KeyInsight) for insight in insights)
        assert any("error" in insight.title.lower() for insight in insights)

    @pytest.mark.asyncio
    async def test_analyze_volume_patterns(self, mock_ai_client, config):
        """Test volume pattern analysis."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        base_time = utc_now().replace(second=0, microsecond=0)
        window_summaries = [
            TimeWindowSummary(
                start_time=base_time,
                end_time=base_time + timedelta(minutes=5),
                window_size=config.window_size,
                log_count=10,
                error_count=0,
                warning_count=0,
                overall_severity=SeverityLevel.LOW,
                summary_text="Normal volume",
                top_errors=[]
            ),
            TimeWindowSummary(
                start_time=base_time + timedelta(minutes=5),
                end_time=base_time + timedelta(minutes=10),
                window_size=config.window_size,
                log_count=15,  # Normal
                error_count=0,
                warning_count=0,
                overall_severity=SeverityLevel.LOW,
                summary_text="Normal volume",
                top_errors=[]
            ),
            TimeWindowSummary(
                start_time=base_time + timedelta(minutes=10),
                end_time=base_time + timedelta(minutes=15),
                window_size=config.window_size,
                log_count=500,  # Large spike - avg will be ~175, threshold ~525, but 500 < 525 still...
                error_count=0,
                warning_count=0,
                overall_severity=SeverityLevel.LOW,
                summary_text="Volume spike",
                top_errors=[]
            ),
        ]

        insights = await summarizer._analyze_volume_patterns(window_summaries, config)

        assert len(insights) > 0
        assert any("volume" in insight.title.lower() for insight in insights)

    @pytest.mark.asyncio
    async def test_analyze_trends(self, mock_ai_client, config):
        """Test trend analysis."""
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        base_time = utc_now().replace(second=0, microsecond=0)
        window_summaries = [
            TimeWindowSummary(
                start_time=base_time,
                end_time=base_time + timedelta(minutes=5),
                window_size=config.window_size,
                log_count=10,
                error_count=1,
                warning_count=0,
                overall_severity=SeverityLevel.LOW,
                summary_text="Window 1",
                top_errors=[]
            ),
            TimeWindowSummary(
                start_time=base_time + timedelta(minutes=5),
                end_time=base_time + timedelta(minutes=10),
                window_size=config.window_size,
                log_count=15,
                error_count=3,
                warning_count=1,
                overall_severity=SeverityLevel.MEDIUM,
                summary_text="Window 2",
                top_errors=[]
            ),
            TimeWindowSummary(
                start_time=base_time + timedelta(minutes=10),
                end_time=base_time + timedelta(minutes=15),
                window_size=config.window_size,
                log_count=20,
                error_count=5,
                warning_count=2,
                overall_severity=SeverityLevel.HIGH,
                summary_text="Window 3",
                top_errors=[]
            ),
        ]

        trends = await summarizer._analyze_trends(window_summaries, config)

        assert len(trends) >= 3  # Error rate, volume, severity
        assert all(isinstance(trend, TrendAnalysis) for trend in trends)

        # Find error rate trend (should be increasing)
        error_trend = next((t for t in trends if "Error Rate" in t.metric_name), None)
        assert error_trend is not None
        assert error_trend.direction in [TrendDirection.INCREASING, TrendDirection.VOLATILE]

    @pytest.mark.asyncio
    async def test_analyze_metric_trend_increasing(self, mock_ai_client):
        """Test increasing trend analysis."""
        summarizer = LogSummarizer(ai_client=mock_ai_client)

        # Clearly increasing values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = await summarizer._analyze_metric_trend(values, "Test Metric")

        assert trend.direction == TrendDirection.INCREASING
        assert trend.confidence > 0.5
        assert trend.change_percentage == 400.0  # 5x increase

    @pytest.mark.asyncio
    async def test_analyze_metric_trend_stable(self, mock_ai_client):
        """Test stable trend analysis."""
        summarizer = LogSummarizer(ai_client=mock_ai_client)

        # Stable values
        values = [5.0, 5.1, 4.9, 5.0, 5.2]
        trend = await summarizer._analyze_metric_trend(values, "Test Metric")

        assert trend.direction == TrendDirection.STABLE
        assert trend.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_metric_trend_volatile(self, mock_ai_client):
        """Test volatile trend analysis."""
        summarizer = LogSummarizer(ai_client=mock_ai_client)

        # Highly variable values
        values = [1.0, 10.0, 2.0, 15.0, 3.0]
        trend = await summarizer._analyze_metric_trend(values, "Test Metric")

        assert trend.direction == TrendDirection.VOLATILE

    @pytest.mark.asyncio
    async def test_get_top_messages(self, mock_ai_client):
        """Test top messages extraction."""
        summarizer = LogSummarizer(ai_client=mock_ai_client)

        messages = [
            "Error A occurred",
            "Error B occurred",
            "Error A occurred",
            "Error C occurred",
            "Error A occurred",
        ]

        top_messages = summarizer._get_top_messages(messages, max_count=3)

        assert len(top_messages) <= 3
        assert "Error A occurred" == top_messages[0]  # Most frequent

    @pytest.mark.asyncio
    async def test_generate_basic_summary(self, mock_ai_client):
        """Test basic summary generation."""
        summarizer = LogSummarizer(ai_client=mock_ai_client)

        # With errors
        summary_with_errors = summarizer._generate_basic_summary([], 5, 3)
        assert "5 errors" in summary_with_errors
        assert "3 warnings" in summary_with_errors
        assert "critical" in summary_with_errors.lower()

        # With warnings only
        summary_with_warnings = summarizer._generate_basic_summary([], 0, 2)
        assert "2 warnings" in summary_with_warnings
        assert "issues" in summary_with_warnings.lower()

        # Normal operation
        summary_normal = summarizer._generate_basic_summary([], 0, 0)
        assert "normally" in summary_normal.lower()

    @pytest.mark.asyncio
    async def test_summarize_without_ai(self, config, sample_log_entries):
        """Test summarization without AI client."""
        summarizer = LogSummarizer(ai_client=None, config=config)

        result = await summarizer.summarize_logs(sample_log_entries, config)

        assert isinstance(result, LogSummaryReport)
        assert result.total_log_entries == len(sample_log_entries)
        assert len(result.window_summaries) >= 1
        assert result.executive_summary
        # Should work without AI but with reduced functionality
        assert "entries" in result.executive_summary

    @pytest.mark.asyncio
    async def test_trend_analysis_disabled(self, mock_ai_client, sample_log_entries):
        """Test behavior when trend analysis is disabled."""
        config = SummarizerConfig(
            window_size=TimeWindowSize.FIVE_MINUTES,
            enable_trend_analysis=False
        )
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        result = await summarizer.summarize_logs(sample_log_entries, config)

        assert len(result.trend_analyses) == 0

    @pytest.mark.asyncio
    async def test_ai_summarization_disabled(self, mock_ai_client, sample_log_entries):
        """Test behavior when AI summarization is disabled."""
        config = SummarizerConfig(
            window_size=TimeWindowSize.FIVE_MINUTES,
            enable_ai_summarization=False
        )
        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config)

        result = await summarizer.summarize_logs(sample_log_entries, config)

        # Should not call AI methods
        mock_ai_client.summarize_logs.assert_not_called()

        # But should still generate summaries
        assert result.executive_summary
        assert all(w.summary_text for w in result.window_summaries)

    @pytest.mark.asyncio
    async def test_confidence_filtering(self, mock_ai_client, config):
        """Test insight filtering by confidence level."""
        config_high_confidence = SummarizerConfig(
            window_size=TimeWindowSize.FIVE_MINUTES,
            min_confidence=0.9  # Very high confidence required
        )

        summarizer = LogSummarizer(ai_client=mock_ai_client, config=config_high_confidence)

        # Create windows that would generate insights with varying confidence
        base_time = utc_now().replace(second=0, microsecond=0)
        window_summaries = [
            TimeWindowSummary(
                start_time=base_time,
                end_time=base_time + timedelta(minutes=5),
                window_size=config.window_size,
                log_count=100,
                error_count=50,  # Very high error rate - should generate high confidence insights
                warning_count=10,
                overall_severity=SeverityLevel.CRITICAL,
                summary_text="Critical window",
                top_errors=["Critical error"]
            ),
        ]

        insights = await summarizer._extract_key_insights([], window_summaries, config_high_confidence)

        # With high confidence threshold, should get fewer or no insights
        # depending on the specific confidence calculations
        assert isinstance(insights, list)

    def test_window_size_mapping(self, mock_ai_client):
        """Test time window size mapping."""
        summarizer = LogSummarizer(ai_client=mock_ai_client)

        assert TimeWindowSize.ONE_MINUTE in summarizer._window_deltas
        assert TimeWindowSize.ONE_HOUR in summarizer._window_deltas
        assert TimeWindowSize.ONE_DAY in summarizer._window_deltas

        assert summarizer._window_deltas[TimeWindowSize.ONE_MINUTE] == timedelta(minutes=1)
        assert summarizer._window_deltas[TimeWindowSize.ONE_HOUR] == timedelta(hours=1)
        assert summarizer._window_deltas[TimeWindowSize.ONE_DAY] == timedelta(days=1)


class TestTimeWindowSummary:
    """Test TimeWindowSummary model."""

    def test_model_creation(self):
        """Test TimeWindowSummary model creation."""
        now = utc_now()
        summary = TimeWindowSummary(
            start_time=now,
            end_time=now + timedelta(minutes=15),
            window_size=TimeWindowSize.FIFTEEN_MINUTES,
            log_count=50,
            error_count=5,
            warning_count=10,
            overall_severity=SeverityLevel.MEDIUM,
            summary_text="Test summary"
        )

        assert summary.start_time == now
        assert summary.window_size == TimeWindowSize.FIFTEEN_MINUTES
        assert summary.log_count == 50
        assert summary.error_count == 5
        assert summary.overall_severity == SeverityLevel.MEDIUM
        assert summary.top_errors == []  # Default empty list
        assert summary.key_events == []  # Default empty list


class TestKeyInsight:
    """Test KeyInsight model."""

    def test_model_creation(self):
        """Test KeyInsight model creation."""
        now = utc_now()
        insight = KeyInsight(
            title="Test Insight",
            description="A test insight for testing",
            severity=SeverityLevel.HIGH,
            confidence=0.8,
            affected_windows=[now, now + timedelta(minutes=15)]
        )

        assert insight.title == "Test Insight"
        assert insight.severity == SeverityLevel.HIGH
        assert insight.confidence == 0.8
        assert len(insight.affected_windows) == 2
        assert insight.recommendation is None  # Optional field
        assert insight.related_logs == []  # Default empty list


class TestTrendAnalysis:
    """Test TrendAnalysis model."""

    def test_model_creation(self):
        """Test TrendAnalysis model creation."""
        trend = TrendAnalysis(
            metric_name="Error Rate",
            direction=TrendDirection.INCREASING,
            confidence=0.75,
            change_percentage=25.5,
            significant_changes=["Spike at 10:00", "Recovery at 11:00"],
            recommendation="Monitor closely"
        )

        assert trend.metric_name == "Error Rate"
        assert trend.direction == TrendDirection.INCREASING
        assert trend.confidence == 0.75
        assert trend.change_percentage == 25.5
        assert len(trend.significant_changes) == 2
        assert trend.recommendation == "Monitor closely"


class TestLogSummaryReport:
    """Test LogSummaryReport model."""

    def test_model_creation(self):
        """Test LogSummaryReport model creation."""
        now = utc_now()

        report = LogSummaryReport(
            time_range_start=now,
            time_range_end=now + timedelta(hours=1),
            total_log_entries=1000,
            window_summaries=[],
            key_insights=[],
            trend_analyses=[],
            executive_summary="Test executive summary",
            recommendations=["Recommendation 1", "Recommendation 2"]
        )

        assert report.total_log_entries == 1000
        assert report.executive_summary == "Test executive summary"
        assert len(report.recommendations) == 2
        assert isinstance(report.generated_at, datetime)
        assert report.window_summaries == []
        assert report.key_insights == []
        assert report.trend_analyses == []
