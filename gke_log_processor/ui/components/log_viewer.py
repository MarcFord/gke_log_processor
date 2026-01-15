"""Log viewer widget with syntax highlighting and real-time updates."""

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
)

from ...ai.highlighter import HighlightTheme, SeverityHighlighter
from ...core.models import LogEntry, LogLevel, SeverityLevel


class LogViewer(Widget):
    """Advanced log viewer with syntax highlighting and filtering."""

    DEFAULT_CSS = """
    LogViewer {
        border: solid $primary;
        height: 100%;
    }

    LogViewer > .log-header {
        dock: top;
        height: 4;
        background: $panel;
    }

    LogViewer > .log-controls {
        dock: top;
        height: 3;
        background: $surface;
    }

    LogViewer > .log-content {
        border: none;
        scrollbar-gutter: stable;
    }

    LogViewer Input {
        margin: 0 1;
    }

    LogViewer Select {
        margin: 0 1;
        max-width: 12;
    }

    LogViewer Button {
        margin: 0 1;
    }

    LogViewer Checkbox {
        margin: 0 1;
    }

    LogViewer Label {
        margin: 0 1;
        content-align: center middle;
    }
    """

    # Reactive attributes
    log_entries: reactive[List[LogEntry]] = reactive(list, layout=True)
    filter_text: reactive[str] = reactive("", layout=True)
    level_filter: reactive[Optional[LogLevel]] = reactive(None, layout=True)
    pod_filter: reactive[Optional[str]] = reactive(None, layout=True)
    auto_scroll: reactive[bool] = reactive(True, layout=True)
    show_timestamps: reactive[bool] = reactive(True, layout=True)
    highlight_enabled: reactive[bool] = reactive(True, layout=True)
    max_lines: reactive[int] = reactive(1000, layout=True)

    class SearchRequested(Message):
        """Message sent when search is requested."""

        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    class ExportRequested(Message):
        """Message sent when log export is requested."""

        def __init__(self, format_type: str) -> None:
            self.format_type = format_type
            super().__init__()

    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        theme: Optional[HighlightTheme] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)

        # Initialize components
        self._log_buffer: deque[LogEntry] = deque(maxlen=self.max_lines)
        self._filtered_entries: List[LogEntry] = []
        self._highlighter = SeverityHighlighter(theme=theme or HighlightTheme.DEFAULT)
        self._console = Console()
        self._search_matches: Set[int] = set()
        self._current_search: str = ""

    def compose(self) -> ComposeResult:
        """Compose the log viewer widget."""
        with Vertical():
            # Header with stats and controls
            with Horizontal(classes="log-header"):
                yield Label("📜 Log Viewer", classes="header-label")
                yield Label("Logs: 0", id="log-count-label")
                yield Label("Filtered: 0", id="filtered-count-label")
                yield Button("📤 Export", id="export-button", variant="default")
                yield Button("🔍 Search", id="search-button", variant="primary")
                yield Button("🗑️ Clear", id="clear-button", variant="warning")

            # Control panel
            with Horizontal(classes="log-controls"):
                yield Input(
                    placeholder="Search logs...",
                    id="search-input",
                    classes="search-input"
                )
                yield Select(
                    [("All Levels", None)] + [(level.value, level) for level in LogLevel],
                    value=None,
                    id="level-select"
                )
                yield Select(
                    [("All Pods", None)],
                    value=None,
                    id="pod-select"
                )
                yield Checkbox("Auto-scroll", value=True, id="auto-scroll-checkbox")
                yield Checkbox("Timestamps", value=True, id="timestamps-checkbox")
                yield Checkbox("Highlight", value=True, id="highlight-checkbox")

            # Log content area
            yield RichLog(
                id="log-content",
                classes="log-content",
                highlight=True,
                markup=True,
                auto_scroll=True,
                max_lines=self.max_lines
            )

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self._update_counts()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-input":
            self.filter_text = event.value
            self._apply_filters()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        if event.input.id == "search-input":
            self.post_message(self.SearchRequested(event.value))
            self._perform_search(event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter select changes."""
        if event.select.id == "level-select":
            self.level_filter = event.value
            self._apply_filters()
        elif event.select.id == "pod-select":
            self.pod_filter = event.value
            self._apply_filters()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        checkbox_id = event.checkbox.id
        if checkbox_id == "auto-scroll-checkbox":
            self.auto_scroll = event.value
            self._update_auto_scroll()
        elif checkbox_id == "timestamps-checkbox":
            self.show_timestamps = event.value
            self._refresh_display()
        elif checkbox_id == "highlight-checkbox":
            self.highlight_enabled = event.value
            self._refresh_display()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "export-button":
            self.post_message(self.ExportRequested("txt"))
        elif button_id == "search-button":
            search_input = self.query_one("#search-input", Input)
            self._perform_search(search_input.value)
        elif button_id == "clear-button":
            self.clear_logs()

    def add_log_entry(self, entry: LogEntry) -> None:
        """Add a single log entry."""
        self._log_buffer.append(entry)

        # Update pod filter options if new pod
        self._update_pod_filter_options()

        # Apply filters and update display
        if self._entry_matches_filters(entry):
            self._add_entry_to_display(entry)

        self._update_counts()

    def add_log_entries(self, entries: List[LogEntry]) -> None:
        """Add multiple log entries efficiently."""
        for entry in entries:
            self._log_buffer.append(entry)

        self._update_pod_filter_options()
        self._apply_filters()
        self._update_counts()

    def clear_logs(self) -> None:
        """Clear all log entries."""
        self._log_buffer.clear()
        self._filtered_entries.clear()
        self._search_matches.clear()

        log_content = self.query_one("#log-content", RichLog)
        log_content.clear()

        self._update_counts()

    def set_theme(self, theme: HighlightTheme) -> None:
        """Update the highlighting theme."""
        self._highlighter.update_theme(theme)
        self._refresh_display()

    def export_logs(self, format_type: str = "txt") -> str:
        """Export current filtered logs to string."""
        if format_type == "txt":
            return self._export_as_text()
        elif format_type == "json":
            return self._export_as_json()
        else:
            return self._export_as_text()

    def search_logs(self, query: str) -> int:
        """Search logs and return number of matches."""
        return self._perform_search(query)

    def _apply_filters(self) -> None:
        """Apply current filters to log entries."""
        self._filtered_entries = [
            entry for entry in self._log_buffer
            if self._entry_matches_filters(entry)
        ]
        self._refresh_display()
        self._update_counts()

    def _entry_matches_filters(self, entry: LogEntry) -> bool:
        """Check if entry matches current filters."""
        # Text filter
        if self.filter_text.strip():
            search_text = self.filter_text.lower()
            if not any(search_text in field.lower() for field in [
                entry.message,
                entry.pod_name,
                entry.container_name,
                entry.source,
                str(entry.level.value if entry.level else "")
            ]):
                return False

        # Level filter
        if self.level_filter and entry.level != self.level_filter:
            return False

        # Pod filter
        if self.pod_filter and entry.pod_name != self.pod_filter:
            return False

        return True

    def _refresh_display(self) -> None:
        """Refresh the entire log display."""
        log_content = self.query_one("#log-content", RichLog)
        log_content.clear()

        for entry in self._filtered_entries:
            self._add_entry_to_display(entry)

    def _add_entry_to_display(self, entry: LogEntry) -> None:
        """Add a single entry to the display."""
        log_content = self.query_one("#log-content", RichLog)

        # Format the log entry
        formatted_text = self._format_log_entry(entry)

        # Add to display
        log_content.write(formatted_text)

    def _format_log_entry(self, entry: LogEntry) -> Text:
        """Format a log entry for display."""
        # Start with base formatting
        if self.highlight_enabled:
            # Use AI-powered highlighting
            formatted = self._highlighter.highlight_log_entry(entry)
        else:
            # Basic formatting without highlighting
            text = Text()

            if self.show_timestamps:
                timestamp_str = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
                text.append(f"[{timestamp_str}] ", style="dim")

            # Log level with color
            if entry.level:
                level_style = self._get_level_style(entry.level)
                text.append(f"[{entry.level.value}] ", style=level_style)

            # Pod and container info
            text.append(f"[{entry.pod_name}", style="blue")
            if entry.container_name:
                text.append(f"/{entry.container_name}", style="blue")
            text.append("] ", style="blue")

            # Message
            text.append(entry.message)

            formatted = text

        # Add search highlighting if active
        if self._current_search and self._current_search.lower() in entry.message.lower():
            # Simple search highlighting (can be enhanced)
            search_style = Style(bgcolor="yellow", color="black")
            # Note: More sophisticated search highlighting would require
            # parsing the Text object and applying styles to matching ranges

        return formatted

    def _get_level_style(self, level: LogLevel) -> str:
        """Get style for log level."""
        level_styles = {
            LogLevel.TRACE: "dim",
            LogLevel.DEBUG: "cyan",
            LogLevel.INFO: "green",
            LogLevel.WARN: "yellow",
            LogLevel.WARNING: "yellow",
            LogLevel.ERROR: "red",
            LogLevel.FATAL: "red bold",
            LogLevel.CRITICAL: "red bold",
        }
        return level_styles.get(level, "white")

    def _perform_search(self, query: str) -> int:
        """Perform search and highlight matches."""
        self._current_search = query
        self._search_matches.clear()

        if not query.strip():
            return 0

        search_text = query.lower()
        matches = 0

        for i, entry in enumerate(self._filtered_entries):
            if search_text in entry.message.lower():
                self._search_matches.add(i)
                matches += 1

        # Refresh display to show search highlighting
        self._refresh_display()

        return matches

    def _update_auto_scroll(self) -> None:
        """Update auto-scroll setting."""
        log_content = self.query_one("#log-content", RichLog)
        log_content.auto_scroll = self.auto_scroll

    def _update_counts(self) -> None:
        """Update the log count labels."""
        total_logs = len(self._log_buffer)
        filtered_logs = len(self._filtered_entries)

        total_label = self.query_one("#log-count-label", Label)
        filtered_label = self.query_one("#filtered-count-label", Label)

        total_label.update(f"Logs: {total_logs}")
        filtered_label.update(f"Filtered: {filtered_logs}")

    def _update_pod_filter_options(self) -> None:
        """Update pod filter options based on current logs."""
        if not self._log_buffer:
            return

        pod_names = sorted(set(entry.pod_name for entry in self._log_buffer))
        pod_options = [("All Pods", None)] + [(pod, pod) for pod in pod_names]

        pod_select = self.query_one("#pod-select", Select)
        current_value = pod_select.value

        # Only update if options changed
        current_options = [option[1] for option in pod_select.options]
        new_options = [option[1] for option in pod_options]

        if current_options != new_options:
            pod_select.set_options(pod_options)
            if current_value in new_options:
                pod_select.value = current_value

    def _export_as_text(self) -> str:
        """Export logs as plain text."""
        lines = []
        for entry in self._filtered_entries:
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            level = entry.level.value if entry.level else "INFO"
            pod_info = f"{entry.pod_name}/{entry.container_name}" if entry.container_name else entry.pod_name
            lines.append(f"[{timestamp}] [{level}] [{pod_info}] {entry.message}")
        return "\\n".join(lines)

    def _export_as_json(self) -> str:
        """Export logs as JSON."""
        import json
        return json.dumps([entry.model_dump() for entry in self._filtered_entries],
                          indent=2, default=str)
