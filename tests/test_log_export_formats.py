"""Tests for log export formats."""

from datetime import datetime, timezone

from gke_log_processor.core.models import LogEntry, LogLevel
from gke_log_processor.ui.components.log_viewer import LogViewer


def _sample_entries():
    now = datetime.now(timezone.utc)
    return [
        LogEntry(
            timestamp=now,
            message="INFO: Application started successfully",
            level=LogLevel.INFO,
            source="app",
            pod_name="app-pod-1",
            namespace="default",
            cluster="test-cluster",
            container_name="app-container",
            raw_message="INFO: Application started successfully",
        ),
        LogEntry(
            timestamp=now,
            message="ERROR: Database connection failed",
            level=LogLevel.ERROR,
            source="db",
            pod_name="app-pod-2",
            namespace="default",
            cluster="test-cluster",
            container_name="db-container",
            raw_message="ERROR: Database connection failed",
        ),
    ]


def _viewer_with_entries():
    viewer = LogViewer()
    entries = _sample_entries()
    viewer._filtered_entries = entries  # pylint: disable=protected-access
    return viewer


def test_export_as_text_includes_messages():
    viewer = _viewer_with_entries()
    content = viewer.export_logs("txt")

    assert "Application started successfully" in content
    assert "Database connection failed" in content


def test_export_as_csv_contains_header():
    viewer = _viewer_with_entries()
    content = viewer.export_logs("csv")
    lines = [line for line in content.splitlines() if line]

    assert lines[0] == "timestamp,level,pod,container,namespace,cluster,message"
    assert any("Database connection failed" in line for line in lines[1:])


def test_export_as_pdf_has_pdf_header_and_footer():
    viewer = _viewer_with_entries()
    content = viewer.export_logs("pdf")

    assert content.startswith("%PDF-1.4")
    assert content.rstrip().endswith("%%EOF")
