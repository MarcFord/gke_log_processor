"""Status bar widget for displaying system status and quick info."""

from datetime import datetime
from typing import Any, Dict, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static


class StatusBarWidget(Widget):
    """Status bar widget for displaying system information and status."""

    DEFAULT_CSS = """
    StatusBarWidget {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text;
    }

    StatusBarWidget > Horizontal {
        height: 1;
    }

    StatusBarWidget Static {
        height: 1;
        margin: 0;
        padding: 0;
    }

    StatusBarWidget .status-section {
        height: 1;
        margin: 0 1;
    }

    StatusBarWidget .status-left {
        dock: left;
        width: auto;
    }

    StatusBarWidget .status-center {
        margin: 0 2;
    }

    StatusBarWidget .status-right {
        dock: right;
        width: auto;
    }

    StatusBarWidget .status-urgent {
        background: $error;
        color: $text;
    }

    StatusBarWidget .status-warning {
        background: $warning;
        color: $text;
    }

    StatusBarWidget .status-success {
        background: $success;
        color: $text;
    }

    StatusBarWidget .status-info {
        background: $primary;
        color: $text;
    }
    """

    # Reactive attributes
    connection_status: reactive[str] = reactive("disconnected", layout=True)
    pods_count: reactive[int] = reactive(0, layout=True)
    logs_count: reactive[int] = reactive(0, layout=True)
    selected_pod: reactive[Optional[str]] = reactive(None, layout=True)
    last_update: reactive[Optional[datetime]] = reactive(None, layout=True)
    error_message: reactive[Optional[str]] = reactive(None, layout=True)
    processing_status: reactive[Optional[str]] = reactive(None, layout=True)
    ai_analysis_active: reactive[bool] = reactive(False, layout=True)

    class StatusClicked(Message):
        """Message sent when status section is clicked."""

        def __init__(self, section: str) -> None:
            self.section = section
            super().__init__()

    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Compose the status bar."""
        with Horizontal():
            # Left section - Connection and basic status
            yield Static(id="status-left", classes="status-section status-left")

            # Center section - Current activity
            yield Static(id="status-center", classes="status-section status-center")

            # Right section - Time and counts
            yield Static(id="status-right", classes="status-section status-right")

    def on_mount(self) -> None:
        """Initialize the status bar when mounted."""
        self._update_display()

    def watch_connection_status(self) -> None:
        """Update display when connection status changes."""
        self._update_display()

    def watch_pods_count(self) -> None:
        """Update display when pods count changes."""
        self._update_display()

    def watch_logs_count(self) -> None:
        """Update display when logs count changes."""
        self._update_display()

    def watch_selected_pod(self) -> None:
        """Update display when selected pod changes."""
        self._update_display()

    def watch_error_message(self) -> None:
        """Update display when error message changes."""
        self._update_display()

    def watch_processing_status(self) -> None:
        """Update display when processing status changes."""
        self._update_display()

    def watch_ai_analysis_active(self) -> None:
        """Update display when AI analysis status changes."""
        self._update_display()

    def update_connection_status(self, status: str) -> None:
        """Update the connection status."""
        self.connection_status = status

    def update_pods_info(self, count: int) -> None:
        """Update the pods count."""
        self.pods_count = count

    def update_logs_info(self, count: int) -> None:
        """Update the logs count."""
        self.logs_count = count
        self.last_update = datetime.now()

    def set_selected_pod(self, pod_name: Optional[str]) -> None:
        """Set the currently selected pod."""
        self.selected_pod = pod_name

    def set_error(self, message: Optional[str]) -> None:
        """Set an error message."""
        self.error_message = message

    def set_processing_status(self, status: Optional[str]) -> None:
        """Set processing status."""
        self.processing_status = status

    def set_ai_analysis_active(self, active: bool) -> None:
        """Set AI analysis status."""
        self.ai_analysis_active = active

    def clear_error(self) -> None:
        """Clear the current error message."""
        self.error_message = None

    def clear_processing_status(self) -> None:
        """Clear processing status."""
        self.processing_status = None

    def _update_display(self) -> None:
        """Update the status bar display."""
        left_text = self._build_left_section()
        center_text = self._build_center_section()
        right_text = self._build_right_section()

        # Update left section
        left_widget = self.query_one("#status-left", Static)
        left_widget.update(left_text)

        # Update center section
        center_widget = self.query_one("#status-center", Static)
        center_widget.update(center_text)

        # Update right section
        right_widget = self.query_one("#status-right", Static)
        right_widget.update(right_text)

    def _build_left_section(self) -> Text:
        """Build the left section with connection and basic status."""
        text = Text()

        # Connection status
        if self.connection_status == "connected":
            text.append("🟢 Connected", style="green")
        elif self.connection_status == "connecting":
            text.append("🟡 Connecting...", style="yellow")
        elif self.connection_status == "error":
            text.append("🔴 Error", style="red")
        else:
            text.append("⚪ Disconnected", style="dim")

        # Pod count
        if self.pods_count > 0:
            text.append(f" │ Pods: {self.pods_count}", style="blue")

        return text

    def _build_center_section(self) -> Text:
        """Build the center section with current activity."""
        text = Text()

        # Priority: Error > Processing > AI Analysis > Selected Pod
        if self.error_message:
            text.append(f"❌ {self.error_message}", style="red bold")
        elif self.processing_status:
            text.append(f"⚙️ {self.processing_status}", style="yellow")
        elif self.ai_analysis_active:
            text.append("🧠 AI Analysis Running...", style="cyan")
        elif self.selected_pod:
            text.append(f"📋 {self.selected_pod}", style="green")
        else:
            text.append("Ready", style="dim")

        return text

    def _build_right_section(self) -> Text:
        """Build the right section with time and counts."""
        text = Text()

        # Log count
        if self.logs_count > 0:
            if self.logs_count >= 1_000_000:
                count_str = f"{self.logs_count / 1_000_000:.1f}M"
            elif self.logs_count >= 1_000:
                count_str = f"{self.logs_count / 1_000:.1f}K"
            else:
                count_str = str(self.logs_count)

            text.append(f"Logs: {count_str}", style="blue")

        # Last update time
        if self.last_update:
            time_str = self.last_update.strftime("%H:%M:%S")
            if text.plain:
                text.append(" │ ")
            text.append(f"Updated: {time_str}", style="dim")

        return text

    def on_click(self, event) -> None:
        """Handle clicks on the status bar."""
        # Determine which section was clicked based on position
        click_x = event.x
        widget_width = self.size.width

        if click_x < widget_width * 0.3:
            self.post_message(self.StatusClicked("left"))
        elif click_x < widget_width * 0.7:
            self.post_message(self.StatusClicked("center"))
        else:
            self.post_message(self.StatusClicked("right"))


class ProgressStatusBar(StatusBarWidget):
    """Extended status bar with progress indicator support."""

    progress_value: reactive[float] = reactive(0.0, layout=True)
    progress_visible: reactive[bool] = reactive(False, layout=True)
    progress_label: reactive[str] = reactive("", layout=True)

    DEFAULT_CSS = """
    ProgressStatusBar {
        height: 2;
    }

    ProgressStatusBar > Horizontal {
        height: 2;
    }

    ProgressStatusBar .progress-bar {
        dock: top;
        height: 1;
        background: $panel;
    }

    ProgressStatusBar .status-row {
        dock: bottom;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the progress status bar."""
        with Horizontal():
            # Progress bar (top)
            yield ProgressBar(
                id="progress-bar",
                classes="progress-bar",
                show_eta=False,
                show_percentage=True
            )

            # Status row (bottom)
            with Horizontal(classes="status-row"):
                yield Static(id="status-left", classes="status-section status-left")
                yield Static(id="status-center", classes="status-section status-center")
                yield Static(id="status-right", classes="status-section status-right")

    def watch_progress_value(self) -> None:
        """Update progress bar when value changes."""
        if hasattr(self, "query_one"):
            try:
                progress_bar = self.query_one("#progress-bar", ProgressBar)
                progress_bar.progress = self.progress_value
            except BaseException:
                pass

    def watch_progress_visible(self) -> None:
        """Update progress bar visibility."""
        if hasattr(self, "query_one"):
            try:
                progress_bar = self.query_one("#progress-bar", ProgressBar)
                progress_bar.display = self.progress_visible
                # Adjust height based on visibility
                self.styles.height = 2 if self.progress_visible else 1
            except BaseException:
                pass

    def show_progress(self, label: str = "", initial_value: float = 0.0) -> None:
        """Show the progress bar with optional label."""
        self.progress_label = label
        self.progress_value = initial_value
        self.progress_visible = True

    def update_progress(self, value: float, label: str = None) -> None:
        """Update progress value and optionally label."""
        self.progress_value = max(0.0, min(100.0, value))
        if label is not None:
            self.progress_label = label

    def hide_progress(self) -> None:
        """Hide the progress bar."""
        self.progress_visible = False
        self.progress_value = 0.0
        self.progress_label = ""

    def _build_center_section(self) -> Text:
        """Override to include progress label when visible."""
        if self.progress_visible and self.progress_label:
            return Text(f"⚙️ {self.progress_label}", style="yellow")
        return super()._build_center_section()
