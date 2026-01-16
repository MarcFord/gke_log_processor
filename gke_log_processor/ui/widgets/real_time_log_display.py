"""Composite widget that provides real-time log display controls."""

from datetime import datetime
from typing import Iterable, Optional, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Static

from ...core.models import LogEntry
from ..components.log_viewer import LogViewer


class RealTimeLogDisplay(Widget):
    """Wrapper around ``LogViewer`` with streaming controls."""

    DEFAULT_CSS = """
    RealTimeLogDisplay {
        border: solid $primary;
        height: 100%;
    }

    RealTimeLogDisplay > .log-display-header {
        dock: top;
        height: 3;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }

    RealTimeLogDisplay > .log-display-toolbar {
        dock: top;
        height: 3;
        background: $surface;
        column-span: 2;
        padding: 0 1;
    }

    RealTimeLogDisplay > LogViewer {
        height: 1fr;
    }
    """

    streaming: reactive[bool] = reactive(False)
    source_name: reactive[Optional[str]] = reactive(None)

    class StartStreaming(Message):
        """Request to start streaming logs."""

    class PauseStreaming(Message):
        """Request to pause streaming."""

    class ClearLogs(Message):
        """Request to clear the current log buffer."""

    class ExportLogs(Message):
        """Request to export logs in the given format."""

        def __init__(self, format_type: str) -> None:
            self.format_type = format_type
            super().__init__()

    def __init__(self, *, name: Optional[str] = None, id: Optional[str] = None, classes: Optional[str] = None) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._viewer = LogViewer(id="real-time-log-viewer")
        self._status = Label("Idle", id="stream-status")

    @property
    def log_viewer(self) -> LogViewer:
        """Expose the underlying log viewer."""

        return self._viewer

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""

        yield Static("📡 Real-Time Logs", classes="log-display-header")
        with Horizontal(classes="log-display-toolbar"):
            yield Button("▶ Start", id="start-stream", variant="success")
            yield Button("⏸ Pause", id="pause-stream", variant="warning")
            yield Button("🧹 Clear", id="clear-logs", variant="default")
            yield Button("💾 Export", id="export-logs", variant="primary")
            yield self._status
        yield self._viewer

    # Proxy methods -----------------------------------------------------
    def set_logs(self, entries: Sequence[LogEntry]) -> None:
        self._viewer.clear_logs()
        self._viewer.add_log_entries(list(entries))

    def add_log_entry(self, entry: LogEntry) -> None:
        self._viewer.add_log_entry(entry)

    def add_log_entries(self, entries: Iterable[LogEntry]) -> None:
        self._viewer.add_log_entries(list(entries))

    def clear_logs(self) -> None:
        self._viewer.clear_logs()

    def export_logs(self, format_type: str = "txt") -> str:
        return self._viewer.export_logs(format_type=format_type)

    def search_logs(self, query: str) -> int:
        return self._viewer.search_logs(query)

    def update_status(self, message: str, *, streaming: Optional[bool] = None, source: Optional[str] = None) -> None:
        """Update status label and reactive fields."""

        if streaming is not None:
            self.streaming = streaming
        if source is not None:
            self.source_name = source
        prefix = "🟢" if self.streaming else "⚪"
        if self.source_name:
            self._status.update(f"{prefix} {message} · {self.source_name}")
        else:
            self._status.update(f"{prefix} {message}")

    # Event handlers ----------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-stream":
            self.streaming = True
            self.post_message(self.StartStreaming())
        elif event.button.id == "pause-stream":
            self.streaming = False
            self.post_message(self.PauseStreaming())
        elif event.button.id == "clear-logs":
            self.post_message(self.ClearLogs())
        elif event.button.id == "export-logs":
            self.post_message(self.ExportLogs("txt"))

    def add_placeholder_log(self, message: str, level: str = "INFO") -> None:
        """Utility for tests: append a synthetic log entry."""

        entry = LogEntry(
            timestamp=datetime.now(),
            message=message,
            level=level,
            pod_name=self.source_name or "unknown",
            container_name="system",
            namespace="default",
            cluster="local",
            source="system",
            raw_message=message,
        )
        self.add_log_entry(entry)
