"""Tests for the PodListWidget component."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from gke_log_processor.core.models import PodInfo
from gke_log_processor.ui.components.pod_list import PodListWidget


@pytest.fixture
def sample_pods():
    """Create sample pod data for testing."""
    return [
        PodInfo(
            name="pod-1",
            namespace="default",
            phase="Running",
            ready=True,
            restart_count=0,
            age_seconds=3600,
            node="node-1",
            container_names=["container-1"],
            labels={"app": "web"},
            annotations={},
            creation_timestamp=datetime.now()
        ),
        PodInfo(
            name="pod-2",
            namespace="kube-system",
            phase="Pending",
            ready=False,
            restart_count=2,
            age_seconds=1800,
            node="node-2",
            container_names=["container-2"],
            labels={"app": "api"},
            annotations={},
            creation_timestamp=datetime.now()
        ),
        PodInfo(
            name="error-pod",
            namespace="default",
            phase="Failed",
            ready=False,
            restart_count=5,
            age_seconds=7200,
            node="node-1",
            container_names=["container-3"],
            labels={"app": "worker"},
            annotations={},
            creation_timestamp=datetime.now()
        )
    ]


class PodListTestApp(App):
    """Test app for component testing."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


class TestPodListWidget:
    """Test cases for PodListWidget."""

    @pytest.fixture
    def pod_widget(self):
        """Create a PodListWidget instance for testing."""
        return PodListWidget()

    @pytest.fixture
    def app_with_pod_widget(self, pod_widget):
        """Create test app with pod widget."""
        return PodListTestApp(pod_widget)

    def test_initialization(self, pod_widget):
        """Test widget initialization."""
        assert pod_widget.pods == []
        assert pod_widget.selected_pod is None
        assert pod_widget.filter_namespace == ""
        assert pod_widget.filter_phase == ""
        assert pod_widget.filter_text == ""
        assert pod_widget.auto_refresh is True

    def test_reactive_attributes(self, pod_widget):
        """Test reactive attribute updates."""
        test_pods = [Mock()]
        pod_widget.pods = test_pods
        assert pod_widget.pods == test_pods

        pod_widget.selected_pod = "test-pod"
        assert pod_widget.selected_pod == "test-pod"

        pod_widget.filter_namespace = "kube-system"
        assert pod_widget.filter_namespace == "kube-system"

    @pytest.mark.asyncio
    async def test_compose_structure(self, app_with_pod_widget):
        """Test widget composition and structure."""
        async with app_with_pod_widget.run_test() as pilot:
            # Check that required widgets are present
            pod_widget = pilot.app.widget

            # Test for header components
            header = pod_widget.query_one(".pod-header")
            assert header is not None

            # Test for control components
            controls = pod_widget.query_one(".pod-controls")
            assert controls is not None

            # Test for data table
            table = pod_widget.query_one("#pod-table", DataTable)
            assert table is not None

    def test_update_pods(self, pod_widget, sample_pods):
        """Test updating pod list."""
        pod_widget.update_pods(sample_pods)
        assert len(pod_widget.pods) == 3
        assert pod_widget.pods[0].name == "pod-1"

    def test_add_pod(self, pod_widget, sample_pods):
        """Test adding individual pod."""
        pod_widget.add_pod(sample_pods[0])
        assert len(pod_widget.pods) == 1
        assert pod_widget.pods[0].name == "pod-1"

    def test_remove_pod(self, pod_widget, sample_pods):
        """Test removing pod."""
        pod_widget.update_pods(sample_pods)
        pod_widget.remove_pod("pod-1")
        assert len(pod_widget.pods) == 2
        assert not any(pod.name == "pod-1" for pod in pod_widget.pods)

    def test_clear_pods(self, pod_widget, sample_pods):
        """Test clearing all pods."""
        pod_widget.update_pods(sample_pods)
        pod_widget.clear_pods()
        assert len(pod_widget.pods) == 0
        assert pod_widget.selected_pod is None

    def test_set_selected_pod(self, pod_widget, sample_pods):
        """Test selecting pod."""
        pod_widget.update_pods(sample_pods)
        pod_widget.set_selected_pod("pod-1")
        assert pod_widget.selected_pod == "pod-1"

    def test_get_filtered_pods_namespace_filter(self, pod_widget, sample_pods):
        """Test namespace filtering."""
        pod_widget.update_pods(sample_pods)
        pod_widget.filter_namespace = "default"

        filtered = pod_widget._get_filtered_pods()
        assert len(filtered) == 2
        assert all(pod.namespace == "default" for pod in filtered)

    def test_get_filtered_pods_phase_filter(self, pod_widget, sample_pods):
        """Test phase filtering."""
        pod_widget.update_pods(sample_pods)
        pod_widget.filter_phase = "Running"

        filtered = pod_widget._get_filtered_pods()
        assert len(filtered) == 1
        assert filtered[0].phase == "Running"

    def test_get_filtered_pods_text_filter(self, pod_widget, sample_pods):
        """Test text filtering."""
        pod_widget.update_pods(sample_pods)
        pod_widget.filter_text = "error"

        filtered = pod_widget._get_filtered_pods()
        assert len(filtered) == 1
        assert "error" in filtered[0].name.lower()

    def test_get_filtered_pods_combined_filters(self, pod_widget, sample_pods):
        """Test combined filtering."""
        pod_widget.update_pods(sample_pods)
        pod_widget.filter_namespace = "default"
        pod_widget.filter_phase = "Running"

        filtered = pod_widget._get_filtered_pods()
        assert len(filtered) == 1
        assert filtered[0].name == "pod-1"

    def test_get_status_icon_running(self, pod_widget):
        """Test status icon for running pod."""
        pod = Mock()
        pod.phase = "Running"
        pod.ready = True
        pod.restart_count = 0

        icon, color = pod_widget._get_status_icon(pod)
        assert icon == "🟢"
        assert color == "green"

    def test_get_status_icon_pending(self, pod_widget):
        """Test status icon for pending pod."""
        pod = Mock()
        pod.phase = "Pending"
        pod.ready = False
        pod.restart_count = 0

        icon, color = pod_widget._get_status_icon(pod)
        assert icon == "🟡"
        assert color == "yellow"

    def test_get_status_icon_failed(self, pod_widget):
        """Test status icon for failed pod."""
        pod = Mock()
        pod.phase = "Failed"
        pod.ready = False
        pod.restart_count = 5

        icon, color = pod_widget._get_status_icon(pod)
        assert icon == "🔴"
        assert color == "red"

    def test_get_status_icon_high_restarts(self, pod_widget):
        """Test status icon for pod with high restarts."""
        pod = Mock()
        pod.phase = "Running"
        pod.ready = True
        pod.restart_count = 10

        icon, color = pod_widget._get_status_icon(pod)
        assert icon == "🟠"
        assert color == "orange"

    def test_format_age_seconds(self, pod_widget):
        """Test age formatting."""
        assert pod_widget._format_age(30) == "30s"
        assert pod_widget._format_age(90) == "1m 30s"
        assert pod_widget._format_age(3661) == "1h 1m"
        assert pod_widget._format_age(86461) == "1d 1m"

    def test_format_age_minutes(self, pod_widget):
        """Test age formatting for minutes."""
        assert pod_widget._format_age(300) == "5m"
        assert pod_widget._format_age(3900) == "1h 5m"

    def test_format_age_hours(self, pod_widget):
        """Test age formatting for hours."""
        assert pod_widget._format_age(7200) == "2h"
        assert pod_widget._format_age(90000) == "1d 1h"

    @pytest.mark.asyncio
    async def test_refresh_button_click(self, app_with_pod_widget):
        """Test refresh button functionality."""
        async with app_with_pod_widget.run_test() as pilot:
            pod_widget = pilot.app.widget

            # Mock the message posting
            with patch.object(pod_widget, 'post_message') as mock_post:
                refresh_button = pod_widget.query_one("#refresh-button")
                await pilot.click(refresh_button)

                # Should post refresh requested message
                mock_post.assert_called_once()
                args = mock_post.call_args[0]
                assert hasattr(args[0], '__class__')

    @pytest.mark.asyncio
    async def test_namespace_filter_input(self, app_with_pod_widget, sample_pods):
        """Test namespace filter input."""
        async with app_with_pod_widget.run_test() as pilot:
            pod_widget = pilot.app.widget
            pod_widget.update_pods(sample_pods)

            # Type in namespace filter
            namespace_input = pod_widget.query_one("#namespace-filter")
            namespace_input.value = "default"

            # This should trigger filtering
            assert len(pod_widget._get_filtered_pods()) == 2

    @pytest.mark.asyncio
    async def test_phase_filter_selection(self, app_with_pod_widget, sample_pods):
        """Test phase filter selection."""
        async with app_with_pod_widget.run_test() as pilot:
            pod_widget = pilot.app.widget
            pod_widget.update_pods(sample_pods)

            # Change phase filter
            phase_select = pod_widget.query_one("#phase-filter")
            phase_select.value = "Running"

            # This should trigger filtering
            assert len(pod_widget._get_filtered_pods()) == 1

    @pytest.mark.asyncio
    async def test_table_row_selection(self, app_with_pod_widget, sample_pods):
        """Test table row selection."""
        async with app_with_pod_widget.run_test() as pilot:
            pod_widget = pilot.app.widget
            pod_widget.update_pods(sample_pods)

            # Simulate table population
            with patch.object(pod_widget, '_update_table_data'):
                pod_widget._update_table_data()

            # Mock row selection
            with patch.object(pod_widget, 'post_message') as mock_post:
                # Simulate clicking on a table row
                table = pod_widget.query_one("#pod-table", DataTable)
                # Since we can't easily simulate row clicks in test,
                # we'll test the method directly
                mock_row = Mock()
                mock_row.get_cell_at = Mock(return_value="pod-1")

                # Test would involve mocking table selection event
                # For now, test the selection method directly
                pod_widget.set_selected_pod("pod-1")
                assert pod_widget.selected_pod == "pod-1"

    def test_get_namespaces(self, pod_widget, sample_pods):
        """Test getting unique namespaces."""
        pod_widget.update_pods(sample_pods)
        namespaces = pod_widget._get_namespaces()

        assert "default" in namespaces
        assert "kube-system" in namespaces
        assert len(namespaces) == 2

    def test_get_phases(self, pod_widget, sample_pods):
        """Test getting unique phases."""
        pod_widget.update_pods(sample_pods)
        phases = pod_widget._get_phases()

        assert "Running" in phases
        assert "Pending" in phases
        assert "Failed" in phases
        assert len(phases) == 3

    def test_message_classes_exist(self, pod_widget):
        """Test that message classes are properly defined."""
        # Test message classes exist and are instantiable
        refresh_msg = pod_widget.RefreshRequested()
        assert refresh_msg is not None

        selection_msg = pod_widget.PodSelected("test-pod")
        assert selection_msg.pod_name == "test-pod"

        filter_msg = pod_widget.FilterChanged("namespace", "test")
        assert filter_msg.filter_type == "namespace"
        assert filter_msg.value == "test"

    @pytest.mark.asyncio
    async def test_auto_refresh_toggle(self, app_with_pod_widget):
        """Test auto-refresh toggle functionality."""
        async with app_with_pod_widget.run_test() as pilot:
            pod_widget = pilot.app.widget

            # Test initial state
            assert pod_widget.auto_refresh is True

            # Toggle auto-refresh
            pod_widget.auto_refresh = False
            assert pod_widget.auto_refresh is False
