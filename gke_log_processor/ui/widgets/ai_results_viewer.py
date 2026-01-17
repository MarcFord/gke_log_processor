"""Composite widget that wraps the AI insights panel with status helpers."""

from typing import Optional, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from ...ai.analyzer import QueryResponse
from ...ai.summarizer import LogSummaryReport
from ...core.models import AIAnalysisResult, LogEntry, SeverityLevel
from ..components.ai_insights_panel import AIInsightsPanel


class AIResultsViewer(Widget):
    """Display AI insights with a concise status banner."""

    DEFAULT_CSS = """
    AIResultsViewer {
        border: solid $primary;
        height: 100%;
    }

    AIResultsViewer > .ai-results-header {
        dock: top;
        height: 3;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }

    AIResultsViewer > .ai-results-status {
        dock: top;
        height: 2;
        background: $surface;
        padding: 0 1;
        content-align: left middle;
    }

    AIResultsViewer > AIInsightsPanel {
        height: 1fr;
    }
    """

    latest_severity: reactive[Optional[SeverityLevel]] = reactive(None)

    def __init__(self, *, name: Optional[str] = None, id: Optional[str] = None, classes: Optional[str] = None) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._status = Label("No analysis yet", id="ai-results-status-text")
        self._panel = AIInsightsPanel(id="ai-results-panel")

    @property
    def panel(self) -> AIInsightsPanel:
        """Expose the underlying insights panel."""

        return self._panel

    def compose(self) -> ComposeResult:
        yield Static("🤖 AI Analysis", classes="ai-results-header")
        with Horizontal(classes="ai-results-status"):
            yield self._status
        yield self._panel

    # Update helpers ----------------------------------------------------
    def update_analysis(self, result: AIAnalysisResult, logs: Sequence[LogEntry]) -> None:
        self.latest_severity = result.overall_severity
        self._panel.update_analysis(result, list(logs))
        self._update_status_from_severity()

    def update_summary(self, summary: LogSummaryReport) -> None:
        self._panel.update_summary(summary)

    def update_query_response(self, response: QueryResponse) -> None:
        self._panel.update_query_response(response)

    def clear(self) -> None:
        self.latest_severity = None
        self._status.update("No analysis yet")
        self._panel.clear_insights()

    def show_message(self, message: str, *, switch_to: Optional[str] = None) -> None:
        """Proxy helper to surface informational messages."""

        self._status.update(message)
        self._panel.show_message(message, switch_to=switch_to)

    # Internal ----------------------------------------------------------
    def _update_status_from_severity(self) -> None:
        if not self.latest_severity:
            self._status.update("No analysis yet")
            return

        severity_emoji = {
            SeverityLevel.LOW: "✅",
            SeverityLevel.MEDIUM: "⚠️",
            SeverityLevel.HIGH: "🔥",
            SeverityLevel.CRITICAL: "🚨",
        }
        emoji = severity_emoji.get(self.latest_severity, "❔")
        label = self.latest_severity.name.title()
        self._status.update(f"{emoji} Overall severity: {label}")
