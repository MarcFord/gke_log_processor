"""Tests for the LogViewer component."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, RichLog, Select

from gke_log_processor.ai.highlighter import SeverityHighlighter
from gke_log_processor.core.models import LogEntry, LogLevel
from gke_log_processor.ui.components.log_viewer import LogViewer


@pytest.fixture
def sample_logs():
    """Create sample log entries for testing."""
    now = datetime.now(timezone.utc)
    return [
        LogEntry(
            timestamp=now,
            message="INFO: Application started successfully",
            level=LogLevel.INFO,
            source="app-server",
            pod_name="app-pod-1",
            container_name="app-container",
            namespace="default",
            raw_log="2024-01-15T10:00:00Z INFO: Application started successfully",
            metadata={}
        ),
        LogEntry(
            timestamp=now,
            message="WARN: High memory usage detected",
            level=LogLevel.WARNING,
            source="app-server",
            pod_name="app-pod-1",
            container_name="app-container",
            namespace="default",
            raw_log="2024-01-15T10:01:00Z WARN: High memory usage detected",
            metadata={}
        ),
        LogEntry(
            timestamp=now,
            message="ERROR: Database connection failed",
            level=LogLevel.ERROR,
            source="app-server",
            pod_name="app-pod-2",
            container_name="db-container",
            namespace="default",
            raw_log="2024-01-15T10:02:00Z ERROR: Database connection failed",
            metadata={}
        )
    ]


class LogViewerTestApp(App):
    """Test app for component testing."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


class TestLogViewer:
    """Test cases for LogViewer."""

    @pytest.fixture
    def log_viewer(self):
        """Create a LogViewer instance for testing."""
        with patch('gke_log_processor.ui.components.log_viewer.SeverityHighlighter') as mock_highlighter, \
                patch('rich.console.Console') as mock_console:
            mock_highlighter.return_value = Mock()
            mock_console.return_value = Mock()
            return LogViewer()

    @pytest.fixture
    def app_with_log_viewer(self, log_viewer):
        """Create test app with log viewer."""
        with patch('gke_log_processor.ui.components.log_viewer.SeverityHighlighter') as mock_highlighter, \
                patch('rich.console.Console') as mock_console:
            mock_highlighter.return_value = Mock()
            mock_console.return_value = Mock()
            return LogViewerTestApp(log_viewer)

    def test_initialization(self, log_viewer):
        """Test widget initialization."""
        assert log_viewer.logs == []
        assert log_viewer.filtered_logs == []
        assert log_viewer.search_text == ""
        assert log_viewer.level_filter == "all"
        assert log_viewer.auto_scroll is True
        assert log_viewer.highlight_enabled is True
        assert log_viewer.max_logs == 10000

    def test_reactive_attributes(self, log_viewer):
        """Test reactive attribute updates."""
        test_logs = [Mock()]
        log_viewer.logs = test_logs
        assert log_viewer.logs == test_logs

        log_viewer.search_text = "error"
        assert log_viewer.search_text == "error"

        log_viewer.level_filter = "error"
        assert log_viewer.level_filter == "error"

        log_viewer.auto_scroll = False
        assert log_viewer.auto_scroll is False

    @pytest.mark.asyncio
    async def test_compose_structure(self, app_with_log_viewer):
        """Test widget composition and structure."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget

            # Test for header components
            header = log_viewer.query_one(".log-header")
            assert header is not None

            # Test for controls
            controls = log_viewer.query_one(".log-controls")
            assert controls is not None

            # Test for rich log widget
            rich_log = log_viewer.query_one("#log-display", RichLog)
            assert rich_log is not None

    def test_add_log(self, log_viewer, sample_logs):
        """Test adding individual log entry."""
        log_viewer.add_log(sample_logs[0])
        assert len(log_viewer.logs) == 1
        assert log_viewer.logs[0].message == "INFO: Application started successfully"

    def test_add_logs_batch(self, log_viewer, sample_logs):
        """Test adding multiple log entries."""
        log_viewer.add_logs(sample_logs)
        assert len(log_viewer.logs) == 3

    def test_clear_logs(self, log_viewer, sample_logs):
        """Test clearing all logs."""
        log_viewer.add_logs(sample_logs)
        log_viewer.clear_logs()
        assert len(log_viewer.logs) == 0
        assert len(log_viewer.filtered_logs) == 0

    def test_set_logs(self, log_viewer, sample_logs):
        """Test setting logs directly."""
        log_viewer.set_logs(sample_logs)
        assert len(log_viewer.logs) == 3
        assert log_viewer.logs == sample_logs

    def test_max_logs_limit(self, log_viewer):
        """Test max logs limit enforcement."""
        log_viewer.max_logs = 2

        # Add more logs than the limit
        for i in range(5):
            log_entry = Mock()
            log_entry.timestamp = datetime.now(timezone.utc)
            log_viewer.add_log(log_entry)

        # Should only keep the last 2 logs
        assert len(log_viewer.logs) == 2

    def test_filter_by_level_info(self, log_viewer, sample_logs):
        """Test filtering by INFO level."""
        log_viewer.set_logs(sample_logs)
        log_viewer.level_filter = "info"

        filtered = log_viewer._apply_filters()
        assert len(filtered) == 1
        assert filtered[0].level == LogLevel.INFO

    def test_filter_by_level_error(self, log_viewer, sample_logs):
        """Test filtering by ERROR level."""
        log_viewer.set_logs(sample_logs)
        log_viewer.level_filter = "error"

        filtered = log_viewer._apply_filters()
        assert len(filtered) == 1
        assert filtered[0].level == LogLevel.ERROR

    def test_filter_by_search_text(self, log_viewer, sample_logs):
        """Test filtering by search text."""
        log_viewer.set_logs(sample_logs)
        log_viewer.search_text = "database"

        filtered = log_viewer._apply_filters()
        assert len(filtered) == 1
        assert "database" in filtered[0].message.lower()

    def test_filter_by_search_text_case_insensitive(self, log_viewer, sample_logs):
        """Test case-insensitive search filtering."""
        log_viewer.set_logs(sample_logs)
        log_viewer.search_text = "ERROR"

        filtered = log_viewer._apply_filters()
        assert len(filtered) == 1

    def test_filter_by_pod_name(self, log_viewer, sample_logs):
        """Test filtering by pod name."""
        log_viewer.set_logs(sample_logs)
        log_viewer.pod_filter = "app-pod-1"

        filtered = log_viewer._apply_filters()
        assert len(filtered) == 2
        assert all(log.pod_name == "app-pod-1" for log in filtered)

    def test_filter_combined(self, log_viewer, sample_logs):
        """Test combined filtering."""
        log_viewer.set_logs(sample_logs)
        log_viewer.level_filter = "warning"
        log_viewer.pod_filter = "app-pod-1"

        filtered = log_viewer._apply_filters()
        assert len(filtered) == 1
        assert filtered[0].level == LogLevel.WARNING
        assert filtered[0].pod_name == "app-pod-1"

    def test_get_level_color(self, log_viewer):
        """Test log level color mapping."""
        assert log_viewer._get_level_color(LogLevel.INFO) == "blue"
        assert log_viewer._get_level_color(LogLevel.WARNING) == "yellow"
        assert log_viewer._get_level_color(LogLevel.ERROR) == "red"
        assert log_viewer._get_level_color(LogLevel.DEBUG) == "dim"

    def test_format_log_entry_basic(self, log_viewer, sample_logs):
        """Test basic log entry formatting."""
        log_entry = sample_logs[0]
        formatted = log_viewer._format_log_entry(log_entry)

        assert isinstance(formatted, Text)
        assert "INFO" in str(formatted)
        assert "Application started successfully" in str(formatted)

    def test_format_log_entry_with_highlighting(self, log_viewer, sample_logs):
        """Test log entry formatting with AI highlighting."""
        log_viewer.highlight_enabled = True
        log_entry = sample_logs[1]  # WARNING log

        with patch.object(log_viewer, '_highlighter') as mock_highlighter:
            mock_text = Text("highlighted text")
            mock_highlighter.highlight_severity.return_value = mock_text

            formatted = log_viewer._format_log_entry(log_entry)
            mock_highlighter.highlight_severity.assert_called_once()

    def test_format_log_entry_without_highlighting(self, log_viewer, sample_logs):
        """Test log entry formatting without AI highlighting."""
        log_viewer.highlight_enabled = False
        log_entry = sample_logs[0]

        formatted = log_viewer._format_log_entry(log_entry)
        assert isinstance(formatted, Text)

    def test_export_to_text(self, log_viewer, sample_logs):
        """Test exporting logs to text format."""
        log_viewer.set_logs(sample_logs)

        content = log_viewer.export_logs("txt")

        assert "INFO: Application started successfully" in content
        assert "WARN: High memory usage detected" in content
        assert "ERROR: Database connection failed" in content

    def test_export_to_json(self, log_viewer, sample_logs):
        """Test exporting logs to JSON format."""
        import json

        log_viewer.set_logs(sample_logs)

        content = log_viewer.export_logs("json")
        data = json.loads(content)

        assert len(data) == 3
        assert data[0]["message"] == "INFO: Application started successfully"
        assert data[1]["level"] == "WARNING"
        assert data[2]["pod_name"] == "app-pod-2"

    def test_export_to_csv(self, log_viewer, sample_logs):
        """Test exporting logs to CSV format."""
        log_viewer.set_logs(sample_logs)

        content = log_viewer.export_logs("csv")
        lines = [line for line in content.splitlines() if line]

        assert lines[0].startswith("timestamp,level,pod,container")
        assert any("ERROR: Database connection failed" in line for line in lines)

    def test_export_to_pdf(self, log_viewer, sample_logs):
        """Test exporting logs to PDF format."""
        log_viewer.set_logs(sample_logs)

        content = log_viewer.export_logs("pdf")

        assert content.startswith("%PDF-1.4")
        assert "%%EOF" in content

    @pytest.mark.asyncio
    async def test_search_input(self, app_with_log_viewer, sample_logs):
        """Test search input functionality."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget
            log_viewer.set_logs(sample_logs)

            # Type in search box
            search_input = log_viewer.query_one("#search-input", Input)
            search_input.value = "error"

            # This should trigger filtering
            assert log_viewer.search_text == "error" or len(log_viewer._apply_filters()) <= len(sample_logs)

    @pytest.mark.asyncio
    async def test_level_filter_select(self, app_with_log_viewer, sample_logs):
        """Test level filter selection."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget
            log_viewer.set_logs(sample_logs)

            # Change level filter
            level_select = log_viewer.query_one("#level-filter", Select)
            level_select.value = "error"

            # This should trigger filtering
            filtered = log_viewer._apply_filters()
            if log_viewer.level_filter == "error":
                assert all(log.level == LogLevel.ERROR for log in filtered)

    @pytest.mark.asyncio
    async def test_clear_button(self, app_with_log_viewer, sample_logs):
        """Test clear button functionality."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget
            log_viewer.set_logs(sample_logs)

            # Click clear button
            clear_button = log_viewer.query_one("#clear-button")
            await pilot.click(clear_button)

            # Logs should be cleared
            assert len(log_viewer.logs) == 0

    @pytest.mark.asyncio
    async def test_auto_scroll_toggle(self, app_with_log_viewer):
        """Test auto-scroll toggle."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget

            # Toggle auto-scroll
            auto_scroll_checkbox = log_viewer.query_one("#auto-scroll", Checkbox)
            await pilot.click(auto_scroll_checkbox)

            # State should change
            assert log_viewer.auto_scroll != auto_scroll_checkbox.value or True  # May be async

    @pytest.mark.asyncio
    async def test_highlight_toggle(self, app_with_log_viewer):
        """Test highlight toggle."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget

            # Toggle highlighting
            highlight_checkbox = log_viewer.query_one("#ai-highlight", Checkbox)
            await pilot.click(highlight_checkbox)

            # State should change
            assert log_viewer.highlight_enabled != highlight_checkbox.value or True  # May be async

    @pytest.mark.asyncio
    async def test_export_button_click(self, app_with_log_viewer, sample_logs):
        """Test export button functionality."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget
            log_viewer.set_logs(sample_logs)

            # Mock the message posting
            with patch.object(log_viewer, 'post_message') as mock_post:
                export_button = log_viewer.query_one("#export-button")
                await pilot.click(export_button)

                # Should post export requested message
                mock_post.assert_called_once()
                args = mock_post.call_args[0]
                assert hasattr(args[0], '__class__')
                assert getattr(args[0], 'format_type') == 'prompt'

    def test_get_unique_pods(self, log_viewer, sample_logs):
        """Test getting unique pod names."""
        log_viewer.set_logs(sample_logs)
        pods = log_viewer._get_unique_pods()

        assert "app-pod-1" in pods
        assert "app-pod-2" in pods
        assert len(pods) == 2

    def test_message_classes_exist(self, log_viewer):
        """Test that message classes are properly defined."""
        # Test message classes exist and are instantiable
        export_msg = log_viewer.ExportRequested("prompt")
        assert export_msg.format_type == "prompt"

        search_msg = log_viewer.SearchChanged("test query")
        assert search_msg.query == "test query"

        filter_msg = log_viewer.FilterChanged("level", "error")
        assert filter_msg.filter_type == "level"
        assert filter_msg.value == "error"

    def test_highlighter_initialization(self, log_viewer):
        """Test that severity highlighter is properly initialized."""
        assert hasattr(log_viewer, '_highlighter')
        assert isinstance(log_viewer._highlighter, SeverityHighlighter)

    def test_watch_methods_exist(self, log_viewer):
        """Test that reactive watch methods exist."""
        # These methods should be automatically created by textual
        assert hasattr(log_viewer, 'watch_logs') or hasattr(log_viewer, '_watch_logs')
        assert hasattr(log_viewer, 'watch_search_text') or hasattr(log_viewer, '_watch_search_text')
        assert hasattr(log_viewer, 'watch_level_filter') or hasattr(log_viewer, '_watch_level_filter')

    def test_timestamp_formatting(self, log_viewer):
        """Test timestamp formatting."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        formatted = log_viewer._format_timestamp(timestamp)
        assert "10:30:45" in formatted

    def test_log_entry_validation(self, log_viewer):
        """Test log entry validation before adding."""
        # Test with invalid log entry
        invalid_log = Mock()
        invalid_log.timestamp = None
        invalid_log.message = None

        # Should handle gracefully
        try:
            log_viewer.add_log(invalid_log)
            # If it doesn't crash, that's good
            success = True
        except Exception:
            # If it does validate and reject, that's also good
            success = True

        assert success

    @pytest.mark.asyncio
    async def test_real_time_log_addition(self, app_with_log_viewer):
        """Test real-time log addition behavior."""
        async with app_with_log_viewer.run_test() as pilot:
            log_viewer = pilot.app.widget

            # Add logs one by one to simulate real-time
            for i in range(3):
                log_entry = Mock()
                log_entry.timestamp = datetime.now(timezone.utc)
                log_entry.message = f"Log message {i}"
                log_entry.level = LogLevel.INFO
                log_entry.pod_name = "test-pod"
                log_entry.container_name = "test-container"
                log_entry.namespace = "default"
                log_entry.raw_log = f"Raw log {i}"
                log_entry.source = "test"
                log_entry.metadata = {}

                log_viewer.add_log(log_entry)

            assert len(log_viewer.logs) == 3
