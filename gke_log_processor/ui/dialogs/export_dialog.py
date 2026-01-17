"""Dialog for selecting log export format."""

from typing import Iterable, List, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static


class ExportLogsDialog(ModalScreen):
    """Modal dialog that lets the user choose an export format."""

    DEFAULT_CSS = """
    ExportLogsDialog {
        align: center middle;
    }

    ExportLogsDialog > .dialog-panel {
        width: 60;
        height: 12;
        background: $surface;
        border: thick $primary;
        border-title-style: bold;
        border-title-color: $accent;
        padding: 1 2;
    }

    ExportLogsDialog Button {
        min-width: 12;
        margin: 0 1;
    }
    """

    selected_format: reactive[str] = reactive("txt")

    class ExportConfirmed(Message):
        """Message emitted when the user confirms an export format."""

        def __init__(self, format_type: str) -> None:
            self.format_type = format_type
            super().__init__()

    def __init__(
        self,
        available_formats: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        super().__init__()
        self._formats: List[Tuple[str, str]] = list(
            available_formats
            or [
                ("Plain Text (.txt)", "txt"),
                ("JSON (.json)", "json"),
                ("CSV (.csv)", "csv"),
                ("PDF Report (.pdf)", "pdf"),
            ]
        )
        if self._formats:
            self.selected_format = self._formats[0][1]

    def compose(self) -> ComposeResult:
        """Render the export dialog content."""
        with Vertical(classes="dialog-panel"):
            yield Static("💾 Export Logs", classes="dialog-title")
            yield Static("Choose the format for the exported logs:")
            yield Select(self._formats, value=self.selected_format, id="format-select")
            with Horizontal():
                yield Button("Cancel", id="cancel", variant="error")
                yield Button("Export", id="export", variant="success")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "format-select":
            self.selected_format = str(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss()
        elif event.button.id == "export":
            self.post_message(self.ExportConfirmed(self.selected_format))
            self.dismiss()
