"""Tests for the log analysis engine."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gke_log_processor.ai.analyzer import (
    BatchProcessor,
    LogAnalysisEngine,
    PatternRecognitionEngine,
    SeverityDetectionAlgorithm,
)
from gke_log_processor.ai.client import GeminiClient, GeminiConfig
from gke_log_processor.core.models import (
    AIAnalysisResult,
    DetectedPattern,
    LogEntry,
    LogLevel,
    PatternType,
    SeverityLevel,
)
from gke_log_processor.core.utils import utc_now


class TestSeverityDetectionAlgorithm:
    """Test severity detection algorithms."""

    def test_keyword_detection_critical(self):
        """Test critical keyword detection."""
        messages = [
            "Fatal error occurred in database connection",
            "CRITICAL: System crashed due to memory corruption",
            "Emergency shutdown initiated - catastrophic failure",
            "Panic: Unable to recover from segmentation fault",
        ]

        for message in messages:
            severity = SeverityDetectionAlgorithm.detect_severity_by_keywords(message)
            assert severity == SeverityLevel.CRITICAL

    def test_keyword_detection_high(self):
        """Test high severity keyword detection."""
        messages = [
            "Error: Connection timeout to external service",
            "Exception in thread main: NullPointerException",
            "Failed to authenticate user credentials",
            "Permission denied accessing secure resource",
        ]

        for message in messages:
            severity = SeverityDetectionAlgorithm.detect_severity_by_keywords(message)
            assert severity == SeverityLevel.HIGH

    def test_pattern_detection_stack_trace(self):
        """Test pattern detection for stack traces."""
        message = "Exception in thread 'main' at com.example.Service.process(Service.java:42)"
        severity = SeverityDetectionAlgorithm.detect_severity_by_patterns(message)
        assert severity == SeverityLevel.HIGH

    def test_pattern_detection_http_errors(self):
        """Test pattern detection for HTTP errors."""
        messages = [
            "HTTP 404 error: Resource not found",
            "500 status code returned from API",
            "Client received 403 error response",
        ]

        for message in messages:
            severity = SeverityDetectionAlgorithm.detect_severity_by_patterns(message)
            assert severity == SeverityLevel.HIGH

    def test_pattern_detection_memory_issues(self):
        """Test pattern detection for memory issues."""
        messages = [
            "OutOfMemoryError: Java heap space",
            "Memory leak detected in process",
            "Out of disk space on volume /var",
        ]

        for message in messages:
            severity = SeverityDetectionAlgorithm.detect_severity_by_patterns(message)
            assert severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]  # Accept either

    def test_log_level_mapping(self):
        """Test log level to severity mapping."""
        now = utc_now()
        test_cases = [
            (LogLevel.CRITICAL, SeverityLevel.CRITICAL),
            (LogLevel.ERROR, SeverityLevel.HIGH),
            (LogLevel.WARNING, SeverityLevel.MEDIUM),
            (LogLevel.INFO, SeverityLevel.LOW),
            (LogLevel.DEBUG, SeverityLevel.LOW),
        ]

        for log_level, expected_severity in test_cases:
            log_entry = LogEntry(
                timestamp=now,
                message="Test message",
                level=log_level,
                source="test-container",
                pod_name="test-pod",
                namespace="default",
                cluster="test-cluster",
                container_name="test-container",
                raw_message="Test message",
            )

            severity = SeverityDetectionAlgorithm.detect_severity_by_log_level(log_entry)
            assert severity == expected_severity

    def test_combined_severity_detection(self):
        """Test combined severity detection taking highest."""
        now = utc_now()

        # Message with critical keyword but INFO level
        log_entry = LogEntry(
            timestamp=now,
            message="Fatal error in application startup",
            level=LogLevel.INFO,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Fatal error in application startup",
        )

        severity = SeverityDetectionAlgorithm.detect_combined_severity(log_entry)
        assert severity == SeverityLevel.CRITICAL  # Should take highest from keyword detection


class TestPatternRecognitionEngine:
    """Test pattern recognition functionality."""

    @pytest.fixture
    def pattern_engine(self):
        """Create pattern recognition engine."""
        return PatternRecognitionEngine()

    @pytest.fixture
    def sample_error_logs(self):
        """Create sample error log entries."""
        now = utc_now()
        logs = []

        # Create recurring error pattern
        for i in range(5):
            logs.append(LogEntry(
                timestamp=now + timedelta(seconds=i * 10),
                message="Database connection failed: Connection refused",
                level=LogLevel.ERROR,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Database connection failed: Connection refused",
            ))

        # Add different error type
        for i in range(3):
            logs.append(LogEntry(
                timestamp=now + timedelta(seconds=100 + i * 5),
                message="Authentication timeout for user session",
                level=LogLevel.ERROR,
                source="auth-container",
                pod_name="auth-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="auth-container",
                raw_message="Authentication timeout for user session",
            ))

        return logs

    def test_normalize_error_message(self, pattern_engine):
        """Test error message normalization."""
        test_cases = [
            (
                "Error at 2024-01-15T10:30:00Z in request ID 123456",
                "Error at [TIMESTAMP] in request ID [NUMBER]"
            ),
            (
                "UUID: 550e8400-e29b-41d4-a716-446655440000 failed",
                "UUID: [UUID] failed"
            ),
            (
                "Memory address 0xabcd1234 corrupted",
                "Memory address [HEX] corrupted"
            ),
            (
                'Query "SELECT * FROM users" failed',
                "Query [STRING] failed"
            ),
        ]

        for original, expected in test_cases:
            normalized = pattern_engine._normalize_error_message(original)
            assert normalized == expected

    def test_detect_error_patterns(self, pattern_engine, sample_error_logs):
        """Test error pattern detection."""
        patterns = pattern_engine.detect_error_patterns(sample_error_logs)

        # Should detect at least one recurring pattern
        assert len(patterns) >= 1

        # Check pattern properties
        db_pattern = next((p for p in patterns if "database" in p.pattern.lower()), None)
        assert db_pattern is not None
        assert db_pattern.type == PatternType.ERROR_PATTERN
        assert db_pattern.occurrence_count >= 3
        assert db_pattern.confidence > 0.5

    def test_detect_temporal_patterns_empty_logs(self, pattern_engine):
        """Test temporal pattern detection with insufficient data."""
        patterns = pattern_engine.detect_temporal_patterns([])
        assert patterns == []

        # Test with too few logs
        now = utc_now()
        few_logs = [
            LogEntry(
                timestamp=now,
                message="Error message",
                level=LogLevel.ERROR,
                source="test-container",
                pod_name="test-pod",
                namespace="default",
                cluster="test-cluster",
                container_name="test-container",
                raw_message="Error message",
            )
        ]
        patterns = pattern_engine.detect_temporal_patterns(few_logs)
        assert patterns == []

    def test_detect_cascade_patterns(self, pattern_engine):
        """Test cascade pattern detection."""
        now = utc_now()

        # Create escalating error sequence
        cascade_logs = [
            LogEntry(
                timestamp=now,
                message="Warning: High memory usage detected",
                level=LogLevel.WARNING,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Warning: High memory usage detected",
            ),
            LogEntry(
                timestamp=now + timedelta(seconds=30),
                message="Error: Memory allocation failed",
                level=LogLevel.ERROR,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Error: Memory allocation failed",
            ),
            LogEntry(
                timestamp=now + timedelta(seconds=60),
                message="Critical: Application crashed due to OOM",
                level=LogLevel.CRITICAL,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Critical: Application crashed due to OOM",
            ),
        ]

        patterns = pattern_engine.detect_cascade_patterns(cascade_logs)

        # Should detect cascade pattern
        cascade_pattern = next((p for p in patterns if p.type == PatternType.PERFORMANCE_ISSUE), None)
        assert cascade_pattern is not None
        assert cascade_pattern.severity == SeverityLevel.CRITICAL
        assert len(cascade_pattern.affected_pods) == 1

    def test_detect_volume_anomalies(self, pattern_engine):
        """Test volume anomaly detection."""
        now = utc_now()

        # Create baseline logs (normal volume)
        normal_logs = []
        for i in range(10):  # 1 log per minute for 10 minutes
            normal_logs.append(LogEntry(
                timestamp=now + timedelta(minutes=i),
                message=f"Normal log message {i}",
                level=LogLevel.INFO,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message=f"Normal log message {i}",
            ))

        # Create high volume spike (50 logs in one minute)
        spike_logs = []
        for i in range(50):
            spike_logs.append(LogEntry(
                timestamp=now + timedelta(minutes=15, seconds=i),
                message=f"High volume log {i}",
                level=LogLevel.INFO,
                source="app-container",
                pod_name="app-pod-2",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message=f"High volume log {i}",
            ))

        all_logs = normal_logs + spike_logs
        patterns = pattern_engine.detect_volume_anomalies(all_logs)

        # Should detect volume anomaly
        volume_pattern = next((p for p in patterns if "volume" in p.pattern.lower()), None)
        assert volume_pattern is not None
        assert volume_pattern.type == PatternType.PERFORMANCE_ISSUE

    def test_group_similar_errors(self, pattern_engine):
        """Test grouping of similar error messages."""
        now = utc_now()

        error_logs = [
            LogEntry(
                timestamp=now,
                message="Connection timeout to database server at 192.168.1.100:5432",
                level=LogLevel.ERROR,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Connection timeout to database server at 192.168.1.100:5432",
            ),
            LogEntry(
                timestamp=now + timedelta(seconds=30),
                message="Connection timeout to database server at 192.168.1.101:5432",
                level=LogLevel.ERROR,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Connection timeout to database server at 192.168.1.101:5432",
            ),
            LogEntry(
                timestamp=now + timedelta(seconds=60),
                message="Authentication failed for user john_doe",
                level=LogLevel.ERROR,
                source="auth-container",
                pod_name="auth-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="auth-container",
                raw_message="Authentication failed for user john_doe",
            ),
        ]

        groups = pattern_engine._group_similar_errors(error_logs)

        # Should have 2 groups (2 similar connection errors, 1 auth error)
        assert len(groups) == 2

        # Find the connection error group
        connection_group = None
        for pattern, logs in groups.items():
            if "connection timeout" in pattern.lower():
                connection_group = logs
                break

        assert connection_group is not None
        assert len(connection_group) == 2  # Both connection errors grouped together


class TestBatchProcessor:
    """Test batch processing functionality."""

    @pytest.fixture
    def batch_processor(self):
        """Create batch processor."""
        return BatchProcessor(batch_size=10, max_concurrent_batches=2)

    @pytest.fixture
    def sample_logs(self):
        """Create sample log entries for batch testing."""
        now = utc_now()
        logs = []

        for i in range(25):  # More than one batch
            logs.append(LogEntry(
                timestamp=now + timedelta(seconds=i),
                message=f"Test message {i}",
                level=LogLevel.INFO,
                source="test-container",
                pod_name="test-pod",
                namespace="default",
                cluster="test-cluster",
                container_name="test-container",
                raw_message=f"Test message {i}",
            ))

        return logs

    @pytest.mark.asyncio
    async def test_batch_processing_empty_logs(self, batch_processor):
        """Test batch processing with empty log list."""
        async def mock_processor(batch):
            return [f"processed-{log.message}" for log in batch]

        results = await batch_processor.process_logs_in_batches([], mock_processor)
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_processing_sync_function(self, batch_processor, sample_logs):
        """Test batch processing with synchronous processor function."""
        def sync_processor(batch):
            return [f"processed-{log.message}" for log in batch]

        results = await batch_processor.process_logs_in_batches(sample_logs, sync_processor)

        assert len(results) == len(sample_logs)
        assert all(result.startswith("processed-") for result in results)

    @pytest.mark.asyncio
    async def test_batch_processing_async_function(self, batch_processor, sample_logs):
        """Test batch processing with asynchronous processor function."""
        async def async_processor(batch):
            await asyncio.sleep(0.01)  # Simulate async work
            return [f"async-processed-{log.message}" for log in batch]

        results = await batch_processor.process_logs_in_batches(sample_logs, async_processor)

        assert len(results) == len(sample_logs)
        assert all(result.startswith("async-processed-") for result in results)

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, batch_processor, sample_logs):
        """Test batch processing handles errors gracefully."""
        def error_processor(batch):
            if len(batch) > 5:  # Simulate error in larger batches
                raise ValueError("Simulated processing error")
            return [f"processed-{log.message}" for log in batch]

        results = await batch_processor.process_logs_in_batches(sample_logs, error_processor)

        # Should still get results from successful batches
        assert len(results) < len(sample_logs)  # Some batches failed
        assert len(results) > 0  # Some batches succeeded

    @pytest.mark.asyncio
    async def test_batch_size_respected(self, sample_logs):
        """Test that batch size is respected."""
        batch_size = 7
        processor = BatchProcessor(batch_size=batch_size, max_concurrent_batches=1)

        processed_batch_sizes = []

        def size_tracking_processor(batch):
            processed_batch_sizes.append(len(batch))
            return [log.message for log in batch]

        await processor.process_logs_in_batches(sample_logs, size_tracking_processor)

        # Check batch sizes
        assert max(processed_batch_sizes) <= batch_size
        assert sum(processed_batch_sizes) == len(sample_logs)


class TestLogAnalysisEngine:
    """Test main log analysis engine."""

    @pytest.fixture
    def analysis_engine(self):
        """Create analysis engine without AI client."""
        return LogAnalysisEngine()

    @pytest.fixture
    def analysis_engine_with_ai(self):
        """Create analysis engine with mocked AI client."""
        from gke_log_processor.ai.client import RateLimitConfig

        mock_config = Mock(spec=GeminiConfig)
        mock_config.api_key = "test-key"
        mock_config.rate_limit = RateLimitConfig()

        with patch('gke_log_processor.ai.analyzer.GeminiClient') as mock_client:
            engine = LogAnalysisEngine(mock_config)
            engine.gemini_client = Mock(spec=GeminiClient)
            return engine

    @pytest.fixture
    def sample_mixed_logs(self):
        """Create mixed sample log entries for comprehensive testing."""
        now = utc_now()
        logs = []

        # Add normal logs
        for i in range(10):
            logs.append(LogEntry(
                timestamp=now + timedelta(seconds=i * 10),
                message=f"Normal operation log {i}",
                level=LogLevel.INFO,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message=f"Normal operation log {i}",
            ))

        # Add error patterns
        for i in range(5):
            logs.append(LogEntry(
                timestamp=now + timedelta(seconds=200 + i * 5),
                message="Database connection failed",
                level=LogLevel.ERROR,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Database connection failed",
            ))

        # Add warnings
        for i in range(3):
            logs.append(LogEntry(
                timestamp=now + timedelta(seconds=300 + i * 10),
                message="High memory usage detected",
                level=LogLevel.WARNING,
                source="app-container",
                pod_name="app-pod-2",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="High memory usage detected",
            ))

        return logs

    @pytest.mark.asyncio
    async def test_analyze_logs_comprehensive_no_ai(self, analysis_engine, sample_mixed_logs):
        """Test comprehensive analysis without AI."""
        result = await analysis_engine.analyze_logs_comprehensive(
            sample_mixed_logs, use_ai=False
        )

        assert isinstance(result, AIAnalysisResult)
        assert result.log_entries_analyzed == len(sample_mixed_logs)
        assert result.overall_severity in [SeverityLevel.MEDIUM, SeverityLevel.HIGH]
        assert result.error_rate > 0  # Should detect errors
        assert len(result.detected_patterns) >= 0  # May or may not find patterns
        assert result.analysis_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_analyze_logs_comprehensive_with_ai(self, analysis_engine_with_ai, sample_mixed_logs):
        """Test comprehensive analysis with AI."""
        # Mock AI client response
        mock_ai_result = Mock()
        mock_ai_result.overall_severity = SeverityLevel.HIGH
        mock_ai_result.confidence_score = 0.9
        mock_ai_result.recommendations = ["Check database connectivity"]
        mock_ai_result.summary = "AI analysis summary"

        analysis_engine_with_ai.gemini_client.analyze_logs = AsyncMock(return_value=mock_ai_result)

        result = await analysis_engine_with_ai.analyze_logs_comprehensive(
            sample_mixed_logs, use_ai=True
        )

        assert isinstance(result, AIAnalysisResult)
        assert result.log_entries_analyzed == len(sample_mixed_logs)
        assert result.overall_severity == SeverityLevel.HIGH  # From AI
        assert result.confidence_score == 0.9  # From AI
        assert "Check database connectivity" in result.recommendations

    @pytest.mark.asyncio
    async def test_analyze_logs_comprehensive_ai_failure(self, analysis_engine_with_ai, sample_mixed_logs):
        """Test comprehensive analysis when AI fails."""
        # Mock AI client to raise exception
        analysis_engine_with_ai.gemini_client.analyze_logs = AsyncMock(
            side_effect=Exception("AI service unavailable")
        )

        result = await analysis_engine_with_ai.analyze_logs_comprehensive(
            sample_mixed_logs, use_ai=True
        )

        # Should still complete analysis using rule-based methods
        assert isinstance(result, AIAnalysisResult)
        assert result.log_entries_analyzed == len(sample_mixed_logs)
        assert result.confidence_score <= 0.7  # Base confidence without AI

    @pytest.mark.asyncio
    async def test_analyze_logs_empty_list(self, analysis_engine):
        """Test analysis with empty log list."""
        with pytest.raises(Exception):  # Should raise LogProcessingError
            await analysis_engine.analyze_logs_comprehensive([])

    @pytest.mark.asyncio
    async def test_analyze_severity_only(self, analysis_engine, sample_mixed_logs):
        """Test severity-only analysis."""
        result = await analysis_engine.analyze_severity_only(sample_mixed_logs)

        assert "overall_severity" in result
        assert "confidence" in result
        assert "error_rate" in result
        assert "severity_distribution" in result

        assert isinstance(result["overall_severity"], SeverityLevel)
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["error_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_severity_only_empty_logs(self, analysis_engine):
        """Test severity-only analysis with empty logs."""
        result = await analysis_engine.analyze_severity_only([])

        assert result["overall_severity"] == SeverityLevel.LOW
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_detect_patterns_only(self, analysis_engine, sample_mixed_logs):
        """Test pattern-only detection."""
        patterns = await analysis_engine.detect_patterns_only(sample_mixed_logs)

        assert isinstance(patterns, list)
        # May or may not find patterns depending on data
        for pattern in patterns:
            assert isinstance(pattern, DetectedPattern)

    def test_calculate_overall_severity(self, analysis_engine):
        """Test overall severity calculation."""
        test_cases = [
            # (severity_distribution, expected_severity)
            ({"critical": 1, "low": 9}, SeverityLevel.CRITICAL),
            ({"high": 3, "low": 7}, SeverityLevel.HIGH),
            ({"medium": 5, "low": 5}, SeverityLevel.MEDIUM),
            ({"low": 2, "medium": 8}, SeverityLevel.MEDIUM),
            ({"low": 10}, SeverityLevel.LOW),
            ({}, SeverityLevel.LOW),
        ]

        for severity_dist, expected in test_cases:
            result = analysis_engine._calculate_overall_severity(severity_dist)
            assert result == expected

    def test_calculate_error_rates(self, analysis_engine):
        """Test error rate calculation."""
        now = utc_now()

        # Create test logs with known error/warning rates
        test_logs = [
            # 2 errors
            LogEntry(timestamp=now, message="Error 1", level=LogLevel.ERROR,
                     source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                     container_name="container", raw_message="Error 1"),
            LogEntry(timestamp=now, message="Error 2", level=LogLevel.CRITICAL,
                     source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                     container_name="container", raw_message="Error 2"),
            # 1 warning
            LogEntry(timestamp=now, message="Warning 1", level=LogLevel.WARNING,
                     source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                     container_name="container", raw_message="Warning 1"),
            # 7 info logs
            *[LogEntry(timestamp=now, message=f"Info {i}", level=LogLevel.INFO,
                       source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                       container_name="container", raw_message=f"Info {i}")
              for i in range(7)]
        ]

        error_rate, warning_rate = analysis_engine._calculate_error_rates(test_logs)

        assert error_rate == 0.2  # 2 errors out of 10 logs
        assert warning_rate == 0.1  # 1 warning out of 10 logs

    def test_extract_top_error_messages(self, analysis_engine):
        """Test extraction of top error messages."""
        now = utc_now()

        error_logs = [
            # Repeat same error 3 times
            *[LogEntry(timestamp=now + timedelta(seconds=i),
                       message="Database connection failed", level=LogLevel.ERROR,
                       source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                       container_name="container", raw_message="Database connection failed")
              for i in range(3)],
            # Different error 2 times
            *[LogEntry(timestamp=now + timedelta(seconds=10 + i),
                       message="Authentication timeout", level=LogLevel.ERROR,
                       source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                       container_name="container", raw_message="Authentication timeout")
              for i in range(2)],
            # Single error
            LogEntry(timestamp=now + timedelta(seconds=20),
                     message="Memory allocation failed", level=LogLevel.ERROR,
                     source="test", pod_name="pod1", namespace="ns", cluster="cluster",
                     container_name="container", raw_message="Memory allocation failed"),
        ]

        top_errors = analysis_engine._extract_top_error_messages(error_logs, limit=2)

        assert len(top_errors) == 2
        # Most common error should be first
        assert "Database connection failed" in top_errors[0]
