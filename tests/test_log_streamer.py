"""
Tests for log streaming functionality.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from gke_log_processor.gke.log_streamer import (
    LogEntry, LogLevel, LogStreamer, LogBuffer, RateLimiter, StreamConfig
)
from gke_log_processor.gke.kubernetes_client import PodInfo
from gke_log_processor.core.exceptions import LogProcessingError


class TestLogEntry:
    """Test the LogEntry class."""

    def test_log_entry_initialization(self):
        """Test LogEntry initialization."""
        timestamp = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=timestamp,
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="Test message"
        )
        
        assert entry.timestamp == timestamp
        assert entry.pod_name == "test-pod"
        assert entry.namespace == "default"
        assert entry.container_name == "app"
        assert entry.message == "Test message"
        assert entry.level == LogLevel.INFO  # Default detection

    def test_log_level_detection_error(self):
        """Test log level detection for error messages."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="ERROR: Something went wrong"
        )
        
        assert entry.level == LogLevel.ERROR
        assert entry.is_error is True

    def test_log_level_detection_warn(self):
        """Test log level detection for warning messages."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="WARNING: This is a warning"
        )
        
        assert entry.level == LogLevel.WARN
        assert entry.is_error is False

    def test_log_level_detection_info(self):
        """Test log level detection for info messages."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="Starting application"
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.severity_score == 2

    def test_formatted_timestamp(self):
        """Test timestamp formatting."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0, 123456, timezone.utc)
        entry = LogEntry(
            timestamp=timestamp,
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="Test"
        )
        
        formatted = entry.formatted_timestamp
        assert "2023-01-01 12:00:00.123" in formatted

    def test_string_representation(self):
        """Test string representation."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0, 0, timezone.utc)
        entry = LogEntry(
            timestamp=timestamp,
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="Test message"
        )
        
        str_repr = str(entry)
        assert "test-pod/app" in str_repr
        assert "Test message" in str_repr
        assert "[INFO]" in str_repr


class TestLogBuffer:
    """Test the LogBuffer class."""

    @pytest.mark.asyncio
    async def test_buffer_initialization(self):
        """Test LogBuffer initialization."""
        buffer = LogBuffer(max_size=100, flush_interval=2.0)
        
        assert buffer.max_size == 100
        assert buffer.flush_interval == 2.0
        assert len(buffer._buffer) == 0

    @pytest.mark.asyncio
    async def test_add_entry(self):
        """Test adding entries to buffer."""
        buffer = LogBuffer(max_size=10, flush_interval=10.0)
        
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="Test"
        )
        
        await buffer.add(entry)
        current_entries = buffer.get_current_entries()
        assert len(current_entries) == 1
        assert current_entries[0] == entry

    @pytest.mark.asyncio
    async def test_buffer_flush_on_size(self):
        """Test buffer flushes when max size is reached."""
        callback_called = False
        received_entries = []
        
        def test_callback(entries):
            nonlocal callback_called, received_entries
            callback_called = True
            received_entries = entries
        
        buffer = LogBuffer(max_size=2, flush_interval=10.0)
        buffer.add_callback(test_callback)
        
        # Add entries
        for i in range(3):
            entry = LogEntry(
                timestamp=datetime.now(timezone.utc),
                pod_name="test-pod",
                namespace="default",
                container_name="app",
                message=f"Test {i}"
            )
            await buffer.add(entry)
        
        assert callback_called
        assert len(received_entries) >= 2

    @pytest.mark.asyncio
    async def test_force_flush(self):
        """Test manual buffer flushing."""
        callback_called = False
        received_entries = []
        
        def test_callback(entries):
            nonlocal callback_called, received_entries
            callback_called = True
            received_entries = entries
        
        buffer = LogBuffer(max_size=10, flush_interval=10.0)
        buffer.add_callback(test_callback)
        
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            pod_name="test-pod",
            namespace="default",
            container_name="app",
            message="Test"
        )
        await buffer.add(entry)
        
        # Manual flush
        await buffer.force_flush()
        
        assert callback_called
        assert len(received_entries) == 1


class TestRateLimiter:
    """Test the RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(max_rate=10.0, window=1.0)
        
        assert limiter.max_rate == 10.0
        assert limiter.window == 1.0

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        """Test acquiring tokens under the rate limit."""
        limiter = RateLimiter(max_rate=5.0, window=1.0)
        
        # Should be able to acquire up to max_rate tokens
        for i in range(5):
            can_acquire = await limiter.acquire()
            assert can_acquire is True

    @pytest.mark.asyncio
    async def test_acquire_over_limit(self):
        """Test acquiring tokens over the rate limit."""
        limiter = RateLimiter(max_rate=2.0, window=1.0)
        
        # Acquire up to limit
        for i in range(2):
            can_acquire = await limiter.acquire()
            assert can_acquire is True
        
        # Next acquisition should fail
        can_acquire = await limiter.acquire()
        assert can_acquire is False


class TestLogStreamer:
    """Test the LogStreamer class."""

    @pytest.fixture
    def mock_k8s_client(self):
        """Create a mock Kubernetes client."""
        mock_client = Mock()
        mock_api = Mock()
        mock_client._get_api.return_value = mock_api
        return mock_client

    @pytest.fixture
    def mock_pod_info(self):
        """Create a mock PodInfo."""
        pod_info = Mock()
        pod_info.name = "test-pod"
        pod_info.namespace = "default"
        pod_info.containers = ["app", "sidecar"]
        return pod_info

    @pytest.fixture
    def log_streamer(self, mock_k8s_client):
        """Create a LogStreamer instance."""
        config = StreamConfig(
            max_buffer_size=100,
            max_logs_per_second=50.0,
            min_log_level=LogLevel.INFO
        )
        return LogStreamer(mock_k8s_client, config)

    def test_log_streamer_initialization(self, mock_k8s_client):
        """Test LogStreamer initialization."""
        config = StreamConfig(max_buffer_size=500)
        streamer = LogStreamer(mock_k8s_client, config)
        
        assert streamer.k8s_client == mock_k8s_client
        assert streamer.config.max_buffer_size == 500
        assert len(streamer._active_streams) == 0

    def test_parse_log_line_simple(self, log_streamer):
        """Test parsing a simple log line."""
        line = "2023-01-01T12:00:00Z INFO Starting application"
        
        entry = log_streamer._parse_log_line(
            line, "test-pod", "default", "app"
        )
        
        assert entry.pod_name == "test-pod"
        assert entry.namespace == "default"
        assert entry.container_name == "app"
        assert "Starting application" in entry.message
        assert entry.raw_line == line

    def test_parse_log_line_with_error(self, log_streamer):
        """Test parsing a log line with error level."""
        line = "ERROR: Database connection failed"
        
        entry = log_streamer._parse_log_line(
            line, "test-pod", "default", "app"
        )
        
        assert entry.level == LogLevel.ERROR
        assert entry.is_error is True
        assert "Database connection failed" in entry.message

    @pytest.mark.asyncio
    async def test_get_recent_logs_success(self, log_streamer, mock_k8s_client, mock_pod_info):
        """Test getting recent logs successfully."""
        # Mock API response
        mock_api = mock_k8s_client._get_api.return_value
        mock_api.read_namespaced_pod_log.return_value = (
            "2023-01-01T12:00:00Z Starting app\n"
            "2023-01-01T12:00:01Z App ready\n"
        )
        
        logs = await log_streamer.get_recent_logs([mock_pod_info], lines=10)
        
        assert len(logs) == 2
        assert logs[0].message == "Starting app"
        assert logs[1].message == "App ready"
        assert all(log.pod_name == "test-pod" for log in logs)

    @pytest.mark.asyncio
    async def test_get_recent_logs_empty_pods(self, log_streamer):
        """Test getting recent logs with empty pod list."""
        logs = await log_streamer.get_recent_logs([], lines=10)
        assert logs == []

    @pytest.mark.asyncio
    async def test_stop_all_streams(self, log_streamer):
        """Test stopping all streams."""
        # Create an async task to simulate a stream
        async def dummy_stream():
            try:
                await asyncio.sleep(10)  # Long running task
            except asyncio.CancelledError:
                pass  # Expected when cancelled
        
        # Start a real task
        task = asyncio.create_task(dummy_stream())
        log_streamer._active_streams["test_stream"] = task
        
        await log_streamer.stop_all_streams()
        
        assert log_streamer._shutdown_event.is_set()
        assert len(log_streamer._active_streams) == 0
        assert task.cancelled() or task.done()

    def test_get_active_streams(self, log_streamer):
        """Test getting active stream IDs."""
        # Add mock streams
        log_streamer._active_streams["stream1"] = Mock()
        log_streamer._active_streams["stream2"] = Mock()
        
        active = log_streamer.get_active_streams()
        assert len(active) == 2
        assert "stream1" in active
        assert "stream2" in active


class TestStreamConfig:
    """Test the StreamConfig class."""

    def test_stream_config_defaults(self):
        """Test default StreamConfig values."""
        config = StreamConfig()
        
        assert config.max_buffer_size == 1000
        assert config.buffer_flush_interval == 1.0
        assert config.max_logs_per_second == 100.0
        assert config.min_log_level == LogLevel.INFO
        assert config.follow_logs is True
        assert config.tail_lines == 100

    def test_stream_config_custom_values(self):
        """Test custom StreamConfig values."""
        config = StreamConfig(
            max_buffer_size=500,
            max_logs_per_second=50.0,
            min_log_level=LogLevel.WARN,
            follow_logs=False
        )
        
        assert config.max_buffer_size == 500
        assert config.max_logs_per_second == 50.0
        assert config.min_log_level == LogLevel.WARN
        assert config.follow_logs is False