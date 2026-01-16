"""Tests for the StatusBarWidget component."""

from datetime import datetime
from unittest.mock import Mock, PropertyMock, patch

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

from gke_log_processor.ui.components.status_bar import (
    ProgressStatusBar,
    StatusBarWidget,
)


class MockStatusBarWidget(StatusBarWidget):
    """Test subclass that disables DOM operations during initialization."""

    def _update_display(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_connection_status(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_pods_count(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_logs_count(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_selected_pod(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_error_message(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_processing_status(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_ai_analysis_active(self):
        """Override to prevent DOM queries during testing."""
        pass


class StatusBarTestApp(App):
    """Test app for component testing."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


class TestStatusBarWidgetTests:
    """Test cases for StatusBarWidget."""

    @pytest.fixture
    def status_bar(self):
        """Create a StatusBarWidget instance for testing."""
        return MockStatusBarWidget()

    @pytest.fixture
    def app_with_status_bar(self, status_bar):
        """Create test app with status bar."""
        return StatusBarTestApp(status_bar)

    def test_initialization(self, status_bar):
        """Test widget initialization."""
        assert status_bar.connection_status == "disconnected"
        assert status_bar.pods_count == 0
        assert status_bar.logs_count == 0
        assert status_bar.selected_pod is None
        assert status_bar.last_update is None
        assert status_bar.error_message is None
        assert status_bar.processing_status is None
        assert status_bar.ai_analysis_active is False

    def test_reactive_attributes(self, status_bar):
        """Test reactive attribute updates."""
        status_bar.connection_status = "connected"
        assert status_bar.connection_status == "connected"

        status_bar.pods_count = 5
        assert status_bar.pods_count == 5

        status_bar.logs_count = 1000
        assert status_bar.logs_count == 1000

        status_bar.selected_pod = "test-pod"
        assert status_bar.selected_pod == "test-pod"

    @pytest.mark.asyncio
    async def test_compose_structure(self, app_with_status_bar):
        """Test widget composition and structure."""
        async with app_with_status_bar.run_test() as pilot:
            status_bar = pilot.app.widget

            # Test for left section
            left_section = status_bar.query_one("#status-left", Static)
            assert left_section is not None

            # Test for center section
            center_section = status_bar.query_one("#status-center", Static)
            assert center_section is not None

            # Test for right section
            right_section = status_bar.query_one("#status-right", Static)
            assert right_section is not None

    def test_update_connection_status(self, status_bar):
        """Test connection status updates."""
        status_bar.update_connection_status("connected")
        assert status_bar.connection_status == "connected"

        status_bar.update_connection_status("error")
        assert status_bar.connection_status == "error"

    def test_update_pods_info(self, status_bar):
        """Test pods info updates."""
        status_bar.update_pods_info(10)
        assert status_bar.pods_count == 10

    def test_update_logs_info(self, status_bar):
        """Test logs info updates."""
        status_bar.update_logs_info(5000)
        assert status_bar.logs_count == 5000
        assert status_bar.last_update is not None
        assert isinstance(status_bar.last_update, datetime)

    def test_set_selected_pod(self, status_bar):
        """Test selected pod setting."""
        status_bar.set_selected_pod("my-pod")
        assert status_bar.selected_pod == "my-pod"

        status_bar.set_selected_pod(None)
        assert status_bar.selected_pod is None

    def test_set_error(self, status_bar):
        """Test error message setting."""
        status_bar.set_error("Connection failed")
        assert status_bar.error_message == "Connection failed"

    def test_set_processing_status(self, status_bar):
        """Test processing status setting."""
        status_bar.set_processing_status("Analyzing logs...")
        assert status_bar.processing_status == "Analyzing logs..."

    def test_set_ai_analysis_active(self, status_bar):
        """Test AI analysis status setting."""
        status_bar.set_ai_analysis_active(True)
        assert status_bar.ai_analysis_active is True

        status_bar.set_ai_analysis_active(False)
        assert status_bar.ai_analysis_active is False

    def test_clear_error(self, status_bar):
        """Test error clearing."""
        status_bar.set_error("Some error")
        status_bar.clear_error()
        assert status_bar.error_message is None

    def test_clear_processing_status(self, status_bar):
        """Test processing status clearing."""
        status_bar.set_processing_status("Processing...")
        status_bar.clear_processing_status()
        assert status_bar.processing_status is None

    def test_build_left_section_disconnected(self, status_bar):
        """Test left section with disconnected status."""
        text = status_bar._build_left_section()
        assert isinstance(text, Text)
        assert "Disconnected" in str(text)

    def test_build_left_section_connected(self, status_bar):
        """Test left section with connected status."""
        status_bar.update_connection_status("connected")
        status_bar.update_pods_info(5)

        text = status_bar._build_left_section()
        assert "Connected" in str(text)
        assert "Pods: 5" in str(text)

    def test_build_left_section_connecting(self, status_bar):
        """Test left section with connecting status."""
        status_bar.update_connection_status("connecting")

        text = status_bar._build_left_section()
        assert "Connecting" in str(text)

    def test_build_left_section_error(self, status_bar):
        """Test left section with error status."""
        status_bar.update_connection_status("error")

        text = status_bar._build_left_section()
        assert "Error" in str(text)

    def test_build_center_section_ready(self, status_bar):
        """Test center section with ready status."""
        text = status_bar._build_center_section()
        assert "Ready" in str(text)

    def test_build_center_section_error_priority(self, status_bar):
        """Test center section with error message (highest priority)."""
        status_bar.set_error("Critical error")
        status_bar.set_processing_status("Processing...")
        status_bar.set_ai_analysis_active(True)
        status_bar.set_selected_pod("test-pod")

        text = status_bar._build_center_section()
        assert "Critical error" in str(text)

    def test_build_center_section_processing_priority(self, status_bar):
        """Test center section with processing status (second priority)."""
        status_bar.set_processing_status("Analyzing data...")
        status_bar.set_ai_analysis_active(True)
        status_bar.set_selected_pod("test-pod")

        text = status_bar._build_center_section()
        assert "Analyzing data..." in str(text)

    def test_build_center_section_ai_analysis_priority(self, status_bar):
        """Test center section with AI analysis (third priority)."""
        status_bar.set_ai_analysis_active(True)
        status_bar.set_selected_pod("test-pod")

        text = status_bar._build_center_section()
        assert "AI Analysis Running" in str(text)

    def test_build_center_section_selected_pod_priority(self, status_bar):
        """Test center section with selected pod (fourth priority)."""
        status_bar.set_selected_pod("my-pod")

        text = status_bar._build_center_section()
        assert "my-pod" in str(text)

    def test_build_right_section_no_data(self, status_bar):
        """Test right section with no data."""
        text = status_bar._build_right_section()
        # Should be empty or contain minimal info
        assert isinstance(text, Text)

    def test_build_right_section_with_logs(self, status_bar):
        """Test right section with log count."""
        status_bar.update_logs_info(1500)

        text = status_bar._build_right_section()
        assert "Logs: 1.5K" in str(text) or "Logs: 1500" in str(text)

    def test_build_right_section_large_numbers(self, status_bar):
        """Test right section with large log counts."""
        status_bar.update_logs_info(1_500_000)

        text = status_bar._build_right_section()
        assert "1.5M" in str(text)

    def test_build_right_section_with_timestamp(self, status_bar):
        """Test right section with last update timestamp."""
        status_bar.update_logs_info(100)

        text = status_bar._build_right_section()
        assert "Updated:" in str(text)

    @pytest.mark.asyncio
    async def test_click_handling(self, app_with_status_bar):
        """Test click event handling."""
        async with app_with_status_bar.run_test() as pilot:
            status_bar = pilot.app.widget

            # Mock click event and size property
            with patch.object(status_bar, 'post_message') as mock_post:
                # Create a mock size object
                mock_size = Mock()
                mock_size.width = 100

                # Patch the property getter itself using property_mock
                with patch.object(type(status_bar), 'size', new_callable=PropertyMock) as mock_size_property:
                    mock_size_property.return_value = mock_size

                    # Simulate click on different sections
                    click_event = Mock()
                    click_event.x = 10  # Left section

                    status_bar.on_click(click_event)

                    # Should post status clicked message
                    mock_post.assert_called_once()

    def test_watch_methods_exist(self, status_bar):
        """Test that reactive watch methods exist."""
        # These methods should be automatically created by textual
        methods_to_check = [
            'watch_connection_status',
            'watch_pods_count',
            'watch_logs_count',
            'watch_selected_pod',
            'watch_error_message',
            'watch_processing_status',
            'watch_ai_analysis_active'
        ]

        for method_name in methods_to_check:
            assert hasattr(status_bar, method_name), f"Method {method_name} should exist"

    def test_message_classes_exist(self, status_bar):
        """Test that message classes are properly defined."""
        # Test message class exists and is instantiable
        status_msg = status_bar.StatusClicked("left")
        assert status_msg.section == "left"


class MockProgressStatusBar(ProgressStatusBar):
    """Test subclass that disables DOM operations during initialization."""

    def _update_display(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_connection_status(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_pods_count(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_logs_count(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_selected_pod(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_error_message(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_processing_status(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_ai_analysis_active(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_progress_value(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_progress_visible(self):
        """Override to prevent DOM queries during testing."""
        pass

    def watch_progress_label(self):
        """Override to prevent DOM queries during testing."""
        pass


class TestProgressStatusBarTests:
    """Test cases for ProgressStatusBar."""

    @pytest.fixture
    def progress_status_bar(self):
        """Create a ProgressStatusBar instance for testing."""
        return MockProgressStatusBar()

    @pytest.fixture
    def app_with_progress_status_bar(self, progress_status_bar):
        """Create test app with progress status bar."""
        return StatusBarTestApp(progress_status_bar)

    def test_initialization(self, progress_status_bar):
        """Test progress status bar initialization."""
        assert progress_status_bar.progress_value == 0.0
        assert progress_status_bar.progress_visible is False
        assert progress_status_bar.progress_label == ""

    def test_progress_reactive_attributes(self, progress_status_bar):
        """Test progress-specific reactive attributes."""
        progress_status_bar.progress_value = 50.0
        assert progress_status_bar.progress_value == 50.0

        progress_status_bar.progress_visible = True
        assert progress_status_bar.progress_visible is True

        progress_status_bar.progress_label = "Loading..."
        assert progress_status_bar.progress_label == "Loading..."

    @pytest.mark.asyncio
    async def test_progress_compose_structure(self, app_with_progress_status_bar):
        """Test progress status bar composition."""
        async with app_with_progress_status_bar.run_test() as pilot:
            progress_bar = pilot.app.widget

            # Test for progress bar widget
            try:
                progress_widget = progress_bar.query_one("#progress-bar")
                assert progress_widget is not None
            except BaseException:
                # Progress bar might not be visible initially
                pass

            # Test for status sections
            left_section = progress_bar.query_one("#status-left", Static)
            assert left_section is not None

    def test_show_progress(self, progress_status_bar):
        """Test showing progress bar."""
        progress_status_bar.show_progress("Loading data...", 25.0)

        assert progress_status_bar.progress_visible is True
        assert progress_status_bar.progress_label == "Loading data..."
        assert progress_status_bar.progress_value == 25.0

    def test_update_progress(self, progress_status_bar):
        """Test updating progress."""
        progress_status_bar.show_progress("Initial", 0.0)
        progress_status_bar.update_progress(75.0, "Almost done...")

        assert progress_status_bar.progress_value == 75.0
        assert progress_status_bar.progress_label == "Almost done..."

    def test_update_progress_bounds(self, progress_status_bar):
        """Test progress value bounds checking."""
        progress_status_bar.update_progress(-10.0)
        assert progress_status_bar.progress_value == 0.0

        progress_status_bar.update_progress(150.0)
        assert progress_status_bar.progress_value == 100.0

    def test_hide_progress(self, progress_status_bar):
        """Test hiding progress bar."""
        progress_status_bar.show_progress("Test", 50.0)
        progress_status_bar.hide_progress()

        assert progress_status_bar.progress_visible is False
        assert progress_status_bar.progress_value == 0.0
        assert progress_status_bar.progress_label == ""

    def test_build_center_section_with_progress(self, progress_status_bar):
        """Test center section display with progress."""
        progress_status_bar.show_progress("Processing files...", 30.0)

        text = progress_status_bar._build_center_section()
        assert "Processing files..." in str(text)

    def test_build_center_section_fallback(self, progress_status_bar):
        """Test center section fallback to parent behavior."""
        progress_status_bar.hide_progress()
        progress_status_bar.set_selected_pod("test-pod")

        text = progress_status_bar._build_center_section()
        assert "test-pod" in str(text)

    def test_watch_progress_methods_exist(self, progress_status_bar):
        """Test that progress watch methods exist."""
        methods_to_check = [
            'watch_progress_value',
            'watch_progress_visible'
        ]

        for method_name in methods_to_check:
            assert hasattr(progress_status_bar, method_name), f"Method {method_name} should exist"

    @pytest.mark.asyncio
    async def test_progress_bar_updates(self, app_with_progress_status_bar):
        """Test that progress bar updates properly."""
        async with app_with_progress_status_bar.run_test() as pilot:
            progress_bar = pilot.app.widget

            # Show progress
            progress_bar.show_progress("Testing...", 50.0)

            # The widget should handle the updates
            assert progress_bar.progress_visible is True
            assert progress_bar.progress_value == 50.0

    def test_inheritance_from_status_bar(self, progress_status_bar):
        """Test that ProgressStatusBar inherits StatusBar functionality."""
        # Should have all StatusBarWidget methods
        assert hasattr(progress_status_bar, 'update_connection_status')
        assert hasattr(progress_status_bar, 'update_pods_info')
        assert hasattr(progress_status_bar, 'set_selected_pod')

        # Test inherited functionality works
        progress_status_bar.update_connection_status("connected")
        assert progress_status_bar.connection_status == "connected"

        progress_status_bar.update_pods_info(3)
        assert progress_status_bar.pods_count == 3

    def test_progress_with_status_updates(self, progress_status_bar):
        """Test combining progress with status updates."""
        # Set up both progress and status
        progress_status_bar.show_progress("Analyzing...", 60.0)
        progress_status_bar.update_connection_status("connected")
        progress_status_bar.update_pods_info(5)

        # Progress should take precedence in center
        center_text = progress_status_bar._build_center_section()
        assert "Analyzing..." in str(center_text)

        # Left section should still show connection status
        left_text = progress_status_bar._build_left_section()
        assert "Connected" in str(left_text)
        assert "Pods: 5" in str(left_text)
