"""Tests for data models."""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from gke_log_processor.core.models import (
    AIAnalysisResult,
    ContainerState,
    ContainerStatus,
    DetectedPattern,
    LogEntry,
    LogLevel,
    LogSummary,
    PatternType,
    PodCondition,
    PodInfo,
    PodPhase,
    SeverityLevel,
    StreamingStats,
)
from gke_log_processor.core.utils import utc_now


class TestContainerStatus:
    """Test ContainerStatus model."""

    def test_container_status_creation(self):
        """Test creating a container status."""
        now = utc_now()
        container = ContainerStatus(
            name="test-container",
            image="nginx:latest",
            state=ContainerState.RUNNING,
            ready=True,
            restart_count=0,
            started_at=now,
        )

        assert container.name == "test-container"
        assert container.image == "nginx:latest"
        assert container.state == ContainerState.RUNNING
        assert container.ready is True
        assert container.restart_count == 0
        assert container.started_at == now
        assert container.is_healthy is True

    def test_container_uptime_calculation(self):
        """Test uptime calculation."""
        start_time = utc_now() - timedelta(seconds=60)
        container = ContainerStatus(
            name="test-container",
            image="nginx:latest",
            state=ContainerState.RUNNING,
            ready=True,
            started_at=start_time,
        )

        uptime = container.uptime_seconds
        assert uptime is not None
        assert 59 <= uptime <= 61  # Allow for small timing differences

    def test_container_not_healthy_states(self):
        """Test non-healthy container states."""
        # Not ready
        container1 = ContainerStatus(
            name="test", image="nginx", state=ContainerState.RUNNING, ready=False
        )
        assert container1.is_healthy is False

        # Not running
        container2 = ContainerStatus(
            name="test", image="nginx", state=ContainerState.WAITING, ready=True
        )
        assert container2.is_healthy is False


class TestPodInfo:
    """Test PodInfo model."""

    def test_pod_info_creation(self):
        """Test creating a pod info."""
        now = utc_now()
        pod = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.RUNNING,
            created_at=now,
            labels={"app": "test"},
            annotations={"version": "1.0"},
        )

        assert pod.name == "test-pod"
        assert pod.namespace == "default"
        assert pod.cluster == "test-cluster"
        assert pod.phase == PodPhase.RUNNING
        assert pod.labels["app"] == "test"
        assert pod.annotations["version"] == "1.0"

    def test_pod_name_validation(self):
        """Test pod name validation."""
        now = utc_now()

        with pytest.raises(ValidationError):
            PodInfo(
                name="",  # Empty name
                namespace="default",
                cluster="test-cluster",
                uid="test-uid-123",
                phase=PodPhase.RUNNING,
                created_at=now,
            )

    def test_pod_readiness(self):
        """Test pod readiness calculation."""
        now = utc_now()

        # Pod with ready condition
        pod_ready = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.RUNNING,
            created_at=now,
            conditions=[
                PodCondition(type="Ready", status="True"),
                PodCondition(type="PodScheduled", status="True"),
            ],
        )
        assert pod_ready.is_ready is True

        # Pod without ready condition
        pod_not_ready = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.PENDING,
            created_at=now,
            conditions=[PodCondition(type="PodScheduled", status="True")],
        )
        assert pod_not_ready.is_ready is False

    def test_container_counts(self):
        """Test container count calculations."""
        now = utc_now()
        pod = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.RUNNING,
            created_at=now,
            containers=[
                ContainerStatus(
                    name="container1",
                    image="nginx",
                    state=ContainerState.RUNNING,
                    ready=True,
                ),
                ContainerStatus(
                    name="container2",
                    image="redis",
                    state=ContainerState.RUNNING,
                    ready=False,
                ),
                ContainerStatus(
                    name="container3",
                    image="postgres",
                    state=ContainerState.WAITING,
                    ready=False,
                ),
            ],
        )

        assert pod.total_containers == 3
        assert pod.ready_containers == 1
        assert pod.container_ready_ratio == "1/3"

    def test_controller_info(self):
        """Test controller information extraction."""
        now = utc_now()
        pod = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.RUNNING,
            created_at=now,
            owner_references=[
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "test-deployment",
                    "uid": "deployment-uid",
                }
            ],
        )

        assert pod.controller_name == "test-deployment"
        assert pod.controller_kind == "Deployment"

    def test_label_operations(self):
        """Test label checking operations."""
        now = utc_now()
        pod = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.RUNNING,
            created_at=now,
            labels={"app": "test", "version": "1.0"},
        )

        assert pod.has_label("app") is True
        assert pod.has_label("app", "test") is True
        assert pod.has_label("app", "wrong") is False
        assert pod.has_label("missing") is False

    def test_get_container(self):
        """Test getting container by name."""
        now = utc_now()
        container1 = ContainerStatus(
            name="container1", image="nginx", state=ContainerState.RUNNING, ready=True
        )
        container2 = ContainerStatus(
            name="container2", image="redis", state=ContainerState.RUNNING, ready=True
        )

        pod = PodInfo(
            name="test-pod",
            namespace="default",
            cluster="test-cluster",
            uid="test-uid-123",
            phase=PodPhase.RUNNING,
            created_at=now,
            containers=[container1, container2],
        )

        found_container = pod.get_container("container1")
        assert found_container is not None
        assert found_container.name == "container1"

        missing_container = pod.get_container("missing")
        assert missing_container is None


class TestLogEntry:
    """Test LogEntry model."""

    def test_log_entry_creation(self):
        """Test creating a log entry."""
        now = utc_now()
        log = LogEntry(
            timestamp=now,
            message="Test log message",
            level=LogLevel.INFO,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Test log message",
        )

        assert log.timestamp == now
        assert log.message == "Test log message"
        assert log.level == LogLevel.INFO
        assert log.source == "test-container"
        assert isinstance(log.id, UUID)

    def test_log_level_normalization(self):
        """Test log level normalization."""
        now = utc_now()

        # Test WARN -> WARNING conversion
        log1 = LogEntry(
            timestamp=now,
            message="Warning message",
            level="WARN",
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Warning message",
        )
        assert log1.level == LogLevel.WARNING

        # Test ERR -> ERROR conversion
        log2 = LogEntry(
            timestamp=now,
            message="Error message",
            level="ERR",
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Error message",
        )
        assert log2.level == LogLevel.ERROR

    def test_message_validation(self):
        """Test message validation and trimming."""
        now = utc_now()
        log = LogEntry(
            timestamp=now,
            message="  Trimmed message  ",
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="  Trimmed message  ",
        )

        assert log.message == "Trimmed message"

    def test_severity_properties(self):
        """Test severity checking properties."""
        now = utc_now()

        # Error log
        error_log = LogEntry(
            timestamp=now,
            message="Error occurred",
            level=LogLevel.ERROR,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Error occurred",
        )
        assert error_log.is_error is True
        assert error_log.is_warning is False
        assert error_log.severity_score == 5

        # Warning log
        warn_log = LogEntry(
            timestamp=now,
            message="Warning occurred",
            level=LogLevel.WARNING,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Warning occurred",
        )
        assert warn_log.is_error is False
        assert warn_log.is_warning is True
        assert warn_log.severity_score == 4

        # Info log
        info_log = LogEntry(
            timestamp=now,
            message="Info message",
            level=LogLevel.INFO,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Info message",
        )
        assert info_log.is_error is False
        assert info_log.is_warning is False
        assert info_log.severity_score == 3

    def test_source_identifier(self):
        """Test source identifier generation."""
        now = utc_now()
        log = LogEntry(
            timestamp=now,
            message="Test message",
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Test message",
        )

        assert log.source_identifier == "test-cluster/default/test-pod/test-container"

    def test_tag_operations(self):
        """Test tag management operations."""
        now = utc_now()
        log = LogEntry(
            timestamp=now,
            message="Test message",
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Test message",
        )

        # Add tags
        log.add_tag("performance")
        log.add_tag("database")
        assert log.has_tag("performance") is True
        assert log.has_tag("database") is True
        assert len(log.tags) == 2

        # Don't add duplicate tags
        log.add_tag("performance")
        assert len(log.tags) == 2

        # Remove tags
        log.remove_tag("performance")
        assert log.has_tag("performance") is False
        assert len(log.tags) == 1


class TestDetectedPattern:
    """Test DetectedPattern model."""

    def test_pattern_creation(self):
        """Test creating a detected pattern."""
        now = utc_now()
        pattern = DetectedPattern(
            type=PatternType.ERROR_PATTERN,
            pattern="Connection refused",
            confidence=0.95,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.HIGH,
        )

        assert pattern.type == PatternType.ERROR_PATTERN
        assert pattern.pattern == "Connection refused"
        assert pattern.confidence == 0.95
        assert pattern.severity == SeverityLevel.HIGH

    def test_confidence_validation(self):
        """Test confidence score validation."""
        now = utc_now()

        # Valid confidence
        pattern = DetectedPattern(
            type=PatternType.ERROR_PATTERN,
            pattern="Test pattern",
            confidence=0.5,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.MEDIUM,
        )
        assert pattern.confidence == 0.5

        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            DetectedPattern(
                type=PatternType.ERROR_PATTERN,
                pattern="Test pattern",
                confidence=1.5,
                first_seen=now,
                last_seen=now,
                severity=SeverityLevel.MEDIUM,
            )

        # Invalid confidence (too low)
        with pytest.raises(ValidationError):
            DetectedPattern(
                type=PatternType.ERROR_PATTERN,
                pattern="Test pattern",
                confidence=-0.1,
                first_seen=now,
                last_seen=now,
                severity=SeverityLevel.MEDIUM,
            )

    def test_add_occurrence(self):
        """Test adding pattern occurrences."""
        now = utc_now()
        pattern = DetectedPattern(
            type=PatternType.ERROR_PATTERN,
            pattern="Connection refused",
            confidence=0.95,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.HIGH,
        )

        # Add occurrence
        pattern.add_occurrence("pod-1", "Connection refused to database")
        assert pattern.occurrence_count == 2  # Started with 1
        assert "pod-1" in pattern.affected_pods
        assert "Connection refused to database" in pattern.sample_messages

        # Add another occurrence from different pod
        pattern.add_occurrence("pod-2", "Another connection refused message")
        assert pattern.occurrence_count == 3
        assert len(pattern.affected_pods) == 2
        assert len(pattern.sample_messages) == 2

        # Test sample message limit (max 5)
        for i in range(5):
            pattern.add_occurrence(f"pod-{i}", f"Message {i}")

        assert len(pattern.sample_messages) == 5
        assert (
            pattern.sample_messages[0] != "Connection refused to database"
        )  # Original should be removed


class TestAIAnalysisResult:
    """Test AIAnalysisResult model."""

    def test_analysis_result_creation(self):
        """Test creating an AI analysis result."""
        start_time = utc_now() - timedelta(hours=1)
        end_time = utc_now()

        result = AIAnalysisResult(
            log_entries_analyzed=1000,
            time_window_start=start_time,
            time_window_end=end_time,
            overall_severity=SeverityLevel.MEDIUM,
            confidence_score=0.85,
            severity_distribution={"INFO": 800, "WARNING": 150, "ERROR": 50},
            error_rate=0.05,
            warning_rate=0.15,
        )

        assert result.log_entries_analyzed == 1000
        assert result.overall_severity == SeverityLevel.MEDIUM
        assert result.confidence_score == 0.85
        assert result.error_rate == 0.05

    def test_analysis_duration_calculation(self):
        """Test analysis duration calculation."""
        start_time = utc_now() - timedelta(hours=2)
        end_time = utc_now()

        result = AIAnalysisResult(
            log_entries_analyzed=1000,
            time_window_start=start_time,
            time_window_end=end_time,
            overall_severity=SeverityLevel.LOW,
            confidence_score=0.9,
        )

        duration = result.analysis_duration_seconds
        assert 7190 <= duration <= 7210  # ~2 hours with some tolerance

    def test_critical_patterns_filtering(self):
        """Test filtering critical patterns."""
        now = utc_now()
        critical_pattern = DetectedPattern(
            type=PatternType.CRASH,
            pattern="Segmentation fault",
            confidence=0.9,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.CRITICAL,
        )

        medium_pattern = DetectedPattern(
            type=PatternType.PERFORMANCE_ISSUE,
            pattern="Slow query",
            confidence=0.8,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.MEDIUM,
        )

        result = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.HIGH,
            confidence_score=0.9,
            detected_patterns=[critical_pattern, medium_pattern],
        )

        critical_patterns = result.critical_patterns
        assert len(critical_patterns) == 1
        assert critical_patterns[0].severity == SeverityLevel.CRITICAL

    def test_high_confidence_patterns(self):
        """Test filtering high confidence patterns."""
        now = utc_now()
        high_confidence = DetectedPattern(
            type=PatternType.ERROR_PATTERN,
            pattern="High confidence error",
            confidence=0.95,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.HIGH,
        )

        low_confidence = DetectedPattern(
            type=PatternType.ERROR_PATTERN,
            pattern="Low confidence error",
            confidence=0.6,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.MEDIUM,
        )

        result = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.MEDIUM,
            confidence_score=0.8,
            detected_patterns=[high_confidence, low_confidence],
        )

        high_conf_patterns = result.high_confidence_patterns
        assert len(high_conf_patterns) == 1
        assert high_conf_patterns[0].confidence == 0.95

    def test_immediate_attention_needed(self):
        """Test immediate attention detection."""
        now = utc_now()

        # High severity
        result1 = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.HIGH,
            confidence_score=0.9,
            error_rate=0.05,
        )
        assert result1.needs_immediate_attention is True

        # High error rate
        result2 = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.LOW,
            confidence_score=0.9,
            error_rate=0.15,  # >10%
        )
        assert result2.needs_immediate_attention is True

        # Critical patterns
        critical_pattern = DetectedPattern(
            type=PatternType.CRASH,
            pattern="System crash",
            confidence=0.9,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.CRITICAL,
        )

        result3 = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.LOW,
            confidence_score=0.9,
            error_rate=0.02,
            detected_patterns=[critical_pattern],
        )
        assert result3.needs_immediate_attention is True

        # Normal case
        result4 = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.LOW,
            confidence_score=0.9,
            error_rate=0.02,
        )
        assert result4.needs_immediate_attention is False

    def test_patterns_by_type(self):
        """Test getting patterns by type."""
        now = utc_now()
        error_pattern = DetectedPattern(
            type=PatternType.ERROR_PATTERN,
            pattern="Error pattern",
            confidence=0.9,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.HIGH,
        )

        performance_pattern = DetectedPattern(
            type=PatternType.PERFORMANCE_ISSUE,
            pattern="Slow response",
            confidence=0.8,
            first_seen=now,
            last_seen=now,
            severity=SeverityLevel.MEDIUM,
        )

        result = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.MEDIUM,
            confidence_score=0.85,
            detected_patterns=[error_pattern, performance_pattern],
        )

        error_patterns = result.get_patterns_by_type(PatternType.ERROR_PATTERN)
        assert len(error_patterns) == 1
        assert error_patterns[0].pattern == "Error pattern"

        perf_patterns = result.get_patterns_by_type(PatternType.PERFORMANCE_ISSUE)
        assert len(perf_patterns) == 1
        assert perf_patterns[0].pattern == "Slow response"

    def test_add_recommendation(self):
        """Test adding recommendations."""
        now = utc_now()
        result = AIAnalysisResult(
            log_entries_analyzed=100,
            time_window_start=now,
            time_window_end=now,
            overall_severity=SeverityLevel.MEDIUM,
            confidence_score=0.85,
        )

        # Add recommendation
        result.add_recommendation("Check database connections")
        assert "Check database connections" in result.recommendations
        assert len(result.recommendations) == 1

        # Don't add duplicate
        result.add_recommendation("Check database connections")
        assert len(result.recommendations) == 1

        # Add different recommendation
        result.add_recommendation("Monitor memory usage")
        assert len(result.recommendations) == 2


class TestLogSummary:
    """Test LogSummary model."""

    def test_log_summary_creation(self):
        """Test creating a log summary."""
        start_time = utc_now() - timedelta(hours=1)
        end_time = utc_now()

        summary = LogSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_log_count=1000,
            pod_count=5,
            namespace_count=2,
            container_count=10,
            log_level_counts={"INFO": 800, "WARNING": 150, "ERROR": 50},
        )

        assert summary.total_log_count == 1000
        assert summary.pod_count == 5
        assert summary.log_level_counts["INFO"] == 800

    def test_summary_duration_calculation(self):
        """Test summary duration calculation."""
        start_time = utc_now() - timedelta(minutes=30)
        end_time = utc_now()

        summary = LogSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_log_count=600,
        )

        duration = summary.summary_duration_minutes
        assert 29.5 <= duration <= 30.5  # ~30 minutes with tolerance

    def test_logs_per_minute_calculation(self):
        """Test logs per minute calculation."""
        start_time = utc_now() - timedelta(minutes=10)
        end_time = utc_now()

        summary = LogSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_log_count=200,
        )

        logs_per_minute = summary.logs_per_minute
        assert 19 <= logs_per_minute <= 21  # ~20 logs/min with tolerance

    def test_error_percentage_calculation(self):
        """Test error percentage calculation."""
        start_time = utc_now()
        end_time = utc_now()

        summary = LogSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_log_count=1000,
            log_level_counts={"INFO": 850, "WARNING": 100, "ERROR": 40, "CRITICAL": 10},
        )

        error_percentage = summary.error_percentage
        assert error_percentage == 5.0  # (40 + 10) / 1000 * 100


class TestStreamingStats:
    """Test StreamingStats model."""

    def test_streaming_stats_creation(self):
        """Test creating streaming stats."""
        stats = StreamingStats(
            total_logs_received=1000,
            logs_per_second=50.0,
            bytes_received=1024000,
            active_pod_count=5,
            buffer_size=100,
            max_buffer_size=1000,
        )

        assert stats.total_logs_received == 1000
        assert stats.logs_per_second == 50.0
        assert stats.active_pod_count == 5

    def test_buffer_utilization_calculation(self):
        """Test buffer utilization calculation."""
        stats = StreamingStats(
            buffer_size=250,
            max_buffer_size=1000,
        )

        assert stats.buffer_utilization == 25.0

    def test_drop_rate_calculation(self):
        """Test drop rate calculation."""
        stats = StreamingStats(
            total_logs_received=950,
            dropped_logs_count=50,
        )

        assert stats.drop_rate == 5.0  # 50 / (950 + 50) * 100

    def test_uptime_calculation(self):
        """Test uptime calculation."""
        start_time = utc_now() - timedelta(seconds=120)
        current_time = utc_now()

        stats = StreamingStats(
            start_time=start_time,
            last_update=current_time,
        )

        uptime = stats.uptime_seconds
        assert 119 <= uptime <= 121  # ~120 seconds with tolerance

    def test_average_logs_per_second(self):
        """Test average logs per second calculation."""
        start_time = utc_now() - timedelta(seconds=60)
        current_time = utc_now()

        stats = StreamingStats(
            start_time=start_time,
            last_update=current_time,
            total_logs_received=1200,
        )

        avg_rate = stats.average_logs_per_second
        assert 19 <= avg_rate <= 21  # ~20 logs/sec with tolerance

    def test_update_stats(self):
        """Test updating statistics."""
        stats = StreamingStats(
            total_logs_received=1000,
            bytes_received=500000,
        )

        initial_update_time = stats.last_update

        # Update stats
        stats.update_stats(
            new_logs=100,
            new_bytes=50000,
            current_buffer_size=75,
            active_pods=3,
            current_rate=25.0,
        )

        assert stats.total_logs_received == 1100
        assert stats.bytes_received == 550000
        assert stats.buffer_size == 75
        assert stats.active_pod_count == 3
        assert stats.logs_per_second == 25.0
        assert stats.last_update > initial_update_time


def test_model_imports():
    """Test that all models can be imported correctly."""
    from gke_log_processor.core.models import (
        AIAnalysisResult,
        ContainerState,
        ContainerStatus,
        DetectedPattern,
        LogEntry,
        LogLevel,
        LogSummary,
        PatternType,
        PodCondition,
        PodInfo,
        PodPhase,
        SeverityLevel,
        StreamingStats,
    )

    # Just test that imports work
    assert AIAnalysisResult is not None
    assert ContainerState is not None
    assert ContainerStatus is not None
    assert DetectedPattern is not None
    assert LogEntry is not None
    assert LogLevel is not None
    assert LogSummary is not None
    assert PatternType is not None
    assert PodCondition is not None
    assert PodInfo is not None
    assert PodPhase is not None
    assert SeverityLevel is not None
    assert StreamingStats is not None
