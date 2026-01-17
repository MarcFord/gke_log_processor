"""AI insights panel for displaying analysis results and recommendations."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Label,
    ProgressBar,
    Select,
    Static,
    Tree,
)

from ...ai.analyzer import QueryResponse
from ...ai.summarizer import LogSummaryReport, SummaryType, TimeWindowSize
from ...core.models import (
    AIAnalysisResult,
    DetectedPattern,
    LogEntry,
    PatternType,
    SeverityLevel,
)


class AIInsightsPanel(Widget):
    """Panel for displaying AI analysis insights and recommendations."""

    DEFAULT_CSS = """
    AIInsightsPanel {
        border: solid $primary;
        height: 100%;
        min-height: 20;
    }

    AIInsightsPanel > .insights-header {
        dock: top;
        height: 3;
        background: $panel;
    }

    AIInsightsPanel > .insights-controls {
        dock: top;
        height: 3;
        background: $surface;
    }

    AIInsightsPanel > .insights-content {
        border: none;
        padding: 1;
    }

    AIInsightsPanel Button {
        margin: 0 1;
    }

    AIInsightsPanel Select {
        margin: 0 1;
        max-width: 15;
    }

    AIInsightsPanel Label {
        margin: 0 1;
        content-align: center middle;
    }

    AIInsightsPanel .severity-critical {
        background: $error;
        color: $text;
    }

    AIInsightsPanel .severity-high {
        background: $warning;
        color: $text;
    }

    AIInsightsPanel .severity-medium {
        background: $accent;
        color: $text;
    }

    AIInsightsPanel .severity-low {
        background: $success;
        color: $text;
    }

    AIInsightsPanel .insight-item {
        border: round $primary;
        margin: 1 0;
        padding: 1;
    }

    AIInsightsPanel .pattern-item {
        border: round $accent;
        margin: 1 0;
        padding: 1;
    }
    """

    # Reactive attributes
    analysis_result: reactive[Optional[AIAnalysisResult]] = reactive(None, layout=True)
    summary_report: reactive[Optional[LogSummaryReport]] = reactive(None, layout=True)
    query_response: reactive[Optional[QueryResponse]] = reactive(None, layout=True)
    display_mode: reactive[str] = reactive("overview", layout=True)
    auto_refresh: reactive[bool] = reactive(True, layout=True)

    class AnalysisRequested(Message):
        """Message sent when new analysis is requested."""

        def __init__(self, analysis_type: str) -> None:
            self.analysis_type = analysis_type
            super().__init__()

    class QueryRequested(Message):
        """Message sent when AI query is requested."""

        def __init__(self, question: str) -> None:
            self.question = question
            super().__init__()

    class RecommendationSelected(Message):
        """Message sent when a recommendation is selected."""

        def __init__(self, recommendation: str) -> None:
            self.recommendation = recommendation
            super().__init__()

    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._current_logs: List[LogEntry] = []

    def compose(self) -> ComposeResult:
        """Compose the AI insights panel."""
        with Vertical():
            # Header
            with Horizontal(classes="insights-header"):
                yield Label("🧠 AI Insights", classes="header-label")
                yield Button("▶ Run", id="run-feature-button", variant="success")
                yield Button("💬 Query", id="query-button", variant="default")

            # Controls
            with Horizontal(classes="insights-controls"):
                yield Select(
                    [
                        ("Overview", "overview"),
                        ("Patterns", "patterns"),
                        ("Summary", "summary"),
                        ("Recommendations", "recommendations"),
                        ("Query Results", "query"),
                    ],
                    value="overview",
                    id="display-mode-select"
                )
                yield Select(
                    [
                        ("Comprehensive", "comprehensive"),
                        ("Quick", "quick"),
                        ("Patterns Only", "patterns"),
                        ("Summary Only", "summary"),
                    ],
                    value="comprehensive",
                    id="analysis-type-select"
                )

            # Content area with scrolling
            with VerticalScroll(classes="insights-content"):
                yield Static(id="insights-display", classes="insights-content")

    def on_mount(self) -> None:
        """Initialize the panel when mounted."""
        self._update_display()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "display-mode-select":
            self.display_mode = event.value
            self._sync_analysis_mode_to_display()

            should_auto_summary = (
                self.display_mode == "summary"
                and self.summary_report is None
                and bool(self._current_logs)
            )

            if should_auto_summary:
                self.show_message("Generating summary...", switch_to="summary")
                self.post_message(self.AnalysisRequested("summary"))
                return

            self._update_display()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "run-feature-button":
            display_select = self.query_one("#display-mode-select", Select)
            analysis_type_select = self.query_one("#analysis-type-select", Select)
            mode = display_select.value

            if mode == "query":
                self._show_query_dialog()
                return

            default_analysis = {
                "overview": "comprehensive",
                "patterns": "patterns",
                "summary": "summary",
                "recommendations": "comprehensive",
            }
            desired = default_analysis.get(mode)
            if desired and analysis_type_select.value != desired:
                analysis_type_select.value = desired

            self.post_message(self.AnalysisRequested(analysis_type_select.value))
        elif button_id == "query-button":
            self._show_query_dialog()

    def update_analysis(self, result: AIAnalysisResult, logs: List[LogEntry]) -> None:
        """Update with new analysis results."""
        self.analysis_result = result
        self._current_logs = logs
        self._update_display()

    def update_summary(self, report: LogSummaryReport) -> None:
        """Update with new summary report."""
        self.summary_report = report
        self._update_display()

    def update_query_response(self, response: QueryResponse) -> None:
        """Update with new query response."""
        self.query_response = response
        # Switch to query view when new response arrives
        display_select = self.query_one("#display-mode-select", Select)
        display_select.value = "query"
        self.display_mode = "query"
        self._update_display()

    def clear_insights(self) -> None:
        """Clear all insights."""
        self.analysis_result = None
        self.summary_report = None
        self.query_response = None
        self._update_display()

    def _sync_analysis_mode_to_display(self) -> None:
        """Align analysis type selection with the display mode."""

        analysis_select = self.query_one("#analysis-type-select", Select)
        mapping = {
            "overview": "comprehensive",
            "patterns": "patterns",
            "summary": "summary",
            "recommendations": "comprehensive",
        }
        desired = mapping.get(self.display_mode)
        if desired and analysis_select.value != desired:
            analysis_select.value = desired

    def _update_display(self) -> None:
        """Update the display based on current mode and data."""
        content = self.query_one("#insights-display", Static)

        render_mapping = {
            "overview": self._render_overview,
            "patterns": self._render_patterns,
            "summary": self._render_summary,
            "recommendations": self._render_recommendations,
            "query": self._render_query_results,
        }

        renderer = render_mapping.get(self.display_mode)
        if renderer is None:
            content.update(RichMarkdown("No content available."))
            return

        content.update(RichMarkdown(renderer()))

    def show_message(self, message: str, *, switch_to: Optional[str] = None) -> None:
        """Display a transient informational message to the user."""

        if switch_to and switch_to != self.display_mode:
            display_select = self.query_one("#display-mode-select", Select)
            display_select.value = switch_to
            self.display_mode = switch_to

        content = self.query_one("#insights-display", Static)
        content.update(RichMarkdown(message))

    def _render_overview(self) -> str:
        """Render overview of all available insights."""
        if not self.analysis_result:
            return "🔍 **No Analysis Available**\\n\\nRun analysis to see AI-powered insights about your logs."

        result = self.analysis_result
        overview_parts = []

        # Overall status
        severity_emoji = {
            SeverityLevel.LOW: "✅",
            SeverityLevel.MEDIUM: "⚠️",
            SeverityLevel.HIGH: "🔥",
            SeverityLevel.CRITICAL: "🚨"
        }

        overview_parts.append(
            f"## {
                severity_emoji.get(
                    result.overall_severity,
                    '❓')} Overall Status: {
                result.overall_severity.value.title()}")
        overview_parts.append(f"**Confidence:** {result.confidence_score:.1%}")
        overview_parts.append(f"**Logs Analyzed:** {result.log_entries_analyzed:,}")
        overview_parts.append("")

        # Key metrics
        overview_parts.append("### 📊 Key Metrics")
        overview_parts.append(f"- **Error Rate:** {result.error_rate:.1%}")
        overview_parts.append(f"- **Warning Rate:** {result.warning_rate:.1%}")
        overview_parts.append(f"- **Analysis Duration:** {result.analysis_duration_seconds:.2f}s")
        overview_parts.append("")

        # Patterns summary
        if result.detected_patterns:
            overview_parts.append("### 🔍 Pattern Summary")
            pattern_counts = {}
            for pattern in result.detected_patterns:
                pattern_counts[pattern.type] = pattern_counts.get(pattern.type, 0) + 1

            for pattern_type, count in pattern_counts.items():
                overview_parts.append(f"- **{pattern_type.value.title()}:** {count} pattern{'s' if count != 1 else ''}")
            overview_parts.append("")

        # Top issues
        if result.top_error_messages:
            overview_parts.append("### ⚠️ Top Issues")
            for i, error in enumerate(result.top_error_messages[:3], 1):
                overview_parts.append(f"{i}. {error}")
            overview_parts.append("")

        # Quick recommendations
        if result.recommendations:
            overview_parts.append("### 💡 Quick Actions")
            for rec in result.recommendations[:3]:
                overview_parts.append(f"- {rec}")

        return "\\n".join(overview_parts)

    def _render_patterns(self) -> str:
        """Render detailed pattern analysis."""
        if not self.analysis_result or not self.analysis_result.detected_patterns:
            return "🔍 **No Patterns Detected**\\n\\nNo recurring patterns found in the current log analysis."

        patterns_parts = []
        patterns_parts.append("## 🔍 Detected Patterns\\n")

        # Group patterns by type
        pattern_groups = {}
        for pattern in self.analysis_result.detected_patterns:
            if pattern.type not in pattern_groups:
                pattern_groups[pattern.type] = []
            pattern_groups[pattern.type].append(pattern)

        for pattern_type, patterns in pattern_groups.items():
            patterns_parts.append(f"### {pattern_type.value.title()} Patterns\\n")

            for i, pattern in enumerate(patterns, 1):
                confidence_bar = "█" * int(pattern.confidence * 10) + "░" * (10 - int(pattern.confidence * 10))

                patterns_parts.append(f"**{i}. {pattern.pattern}**")
                patterns_parts.append(f"- **Confidence:** {pattern.confidence:.1%} `{confidence_bar}`")
                patterns_parts.append(f"- **Severity:** {pattern.severity.value.title()}")
                patterns_parts.append(f"- **Occurrences:** {pattern.occurrence_count}")
                patterns_parts.append(f"- **Affected Pods:** {', '.join(pattern.affected_pods[:3])}")
                if len(pattern.affected_pods) > 3:
                    patterns_parts.append(f"  ...and {len(pattern.affected_pods) - 3} more")

                if pattern.recommendation:
                    patterns_parts.append(f"- **Recommendation:** {pattern.recommendation}")

                patterns_parts.append("")

        return "\\n".join(patterns_parts)

    def _render_summary(self) -> str:
        """Render summary report."""
        if not self.summary_report:
            return "📋 **No Summary Available**\\n\\nGenerate a summary to see time-window analysis and trends."

        summary = self.summary_report
        summary_parts = []

        summary_parts.append("## 📋 Log Summary Report\\n")

        # Time range
        start_time = summary.time_range_start.strftime("%Y-%m-%d %H:%M:%S")
        end_time = summary.time_range_end.strftime("%Y-%m-%d %H:%M:%S")
        summary_parts.append(f"**Analysis Period:** {start_time} to {end_time}")
        summary_parts.append(f"**Total Log Entries:** {summary.total_log_entries:,}")
        summary_parts.append(f"**Time Windows:** {len(summary.window_summaries)}")
        summary_parts.append("")

        # Executive summary
        if summary.executive_summary:
            summary_parts.append("### 📊 Executive Summary")
            summary_parts.append(summary.executive_summary)
            summary_parts.append("")

        # Key insights
        if summary.key_insights:
            summary_parts.append("### 💡 Key Insights")
            for insight in summary.key_insights[:5]:
                confidence_bar = "█" * int(insight.confidence * 5) + "░" * (5 - int(insight.confidence * 5))
                summary_parts.append(f"**{insight.title}**")
                summary_parts.append(f"- {insight.description}")
                summary_parts.append(f"- Confidence: {insight.confidence:.1%} `{confidence_bar}`")
                summary_parts.append("")

        # Trend analysis
        if summary.trend_analyses:
            summary_parts.append("### 📈 Trends")
            for trend in summary.trend_analyses:
                direction_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(
                    trend.direction.value.lower(), "❓"
                )
                summary_parts.append(f"**{direction_emoji} {trend.metric_name}:** {trend.direction.value}")
                if trend.change_percentage is not None:
                    summary_parts.append(f"- Change: {trend.change_percentage:+.1f}%")
                if trend.recommendation:
                    summary_parts.append(f"- Action: {trend.recommendation}")
                summary_parts.append("")

        return "\\n".join(summary_parts)

    def _render_recommendations(self) -> str:
        """Render actionable recommendations."""
        recommendations = []

        # Collect recommendations from various sources
        if self.analysis_result and self.analysis_result.recommendations:
            recommendations.extend(self.analysis_result.recommendations)

        if self.summary_report and self.summary_report.recommendations:
            recommendations.extend(self.summary_report.recommendations)

        if not recommendations:
            return "💡 **No Recommendations Available**\\n\\nRun analysis to get AI-powered recommendations for your logs."

        rec_parts = []
        rec_parts.append("## 💡 AI Recommendations\\n")

        # Prioritize recommendations
        priority_keywords = {
            "immediate": 1,
            "urgent": 1,
            "critical": 1,
            "important": 2,
            "should": 3,
            "consider": 4,
            "may": 5
        }

        def get_priority(rec: str) -> int:
            rec_lower = rec.lower()
            for keyword, priority in priority_keywords.items():
                if keyword in rec_lower:
                    return priority
            return 5

        sorted_recs = sorted(set(recommendations), key=get_priority)

        current_priority = None
        for i, rec in enumerate(sorted_recs, 1):
            rec_priority = get_priority(rec)

            # Add priority headers
            if rec_priority != current_priority:
                current_priority = rec_priority
                if rec_priority == 1:
                    rec_parts.append("### 🚨 Immediate Action Required")
                elif rec_priority == 2:
                    rec_parts.append("### ⚠️ Important")
                elif rec_priority == 3:
                    rec_parts.append("### 💭 Suggested")
                else:
                    rec_parts.append("### 📝 Consider")
                rec_parts.append("")

            rec_parts.append(f"{i}. {rec}")
            rec_parts.append("")

        return "\\n".join(rec_parts)

    def _render_query_results(self) -> str:
        """Render AI query results."""
        if not self.query_response:
            return "💬 **No Query Results**\\n\\nAsk a question about your logs to get AI-powered answers."

        response = self.query_response
        query_parts = []

        query_parts.append("## 💬 AI Query Response\\n")

        # Query info
        query_parts.append(f"**Query Duration:** {response.query_duration_seconds:.2f}s")
        query_parts.append(f"**Sources Analyzed:** {response.sources_analyzed} logs")

        # Confidence indicator
        confidence_bar = "█" * int(response.confidence_score * 10) + "░" * (10 - int(response.confidence_score * 10))
        query_parts.append(f"**Confidence:** {response.confidence_score:.1%} `{confidence_bar}`")
        query_parts.append("")

        # Answer
        query_parts.append("### 🤖 Answer")
        query_parts.append(response.answer)
        query_parts.append("")

        # Related patterns
        if response.related_patterns:
            query_parts.append("### 🔗 Related Patterns")
            for pattern in response.related_patterns:
                query_parts.append(f"- {pattern}")
            query_parts.append("")

        # Follow-up suggestions
        if response.suggested_followups:
            query_parts.append("### ❓ Suggested Follow-ups")
            for followup in response.suggested_followups:
                query_parts.append(f"- {followup}")
            query_parts.append("")

        return "\\n".join(query_parts)

    def _show_query_dialog(self) -> None:
        """Show dialog for AI query input."""
        # This would typically show a modal dialog
        # For now, we'll post a message that can be handled by the parent
        self.post_message(self.QueryRequested("What are the main issues in these logs?"))
