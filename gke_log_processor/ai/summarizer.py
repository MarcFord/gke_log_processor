"""Log summarization engine with time-window analysis and trend detection."""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.models import LogEntry, SeverityLevel
from ..core.utils import utc_now
from .client import GeminiClient


class TimeWindowSize(Enum):
    """Supported time window sizes for summarization."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    SIX_HOURS = "6h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"


class SummaryType(Enum):
    """Types of summaries to generate."""

    EXECUTIVE = "executive"  # High-level overview for management
    TECHNICAL = "technical"  # Detailed technical information
    OPERATIONAL = "operational"  # Focus on operational issues
    SECURITY = "security"  # Security-focused summary
    PERFORMANCE = "performance"  # Performance and resource usage
    BRIEF = "brief"  # Short bullet-point summary


class TrendDirection(Enum):
    """Direction of trends over time."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class TimeWindowSummary(BaseModel):
    """Summary of logs for a specific time window."""

    start_time: datetime = Field(..., description="Start of time window")
    end_time: datetime = Field(..., description="End of time window")
    window_size: TimeWindowSize = Field(..., description="Size of time window")
    log_count: int = Field(..., description="Number of log entries in window")
    error_count: int = Field(..., description="Number of error entries")
    warning_count: int = Field(..., description="Number of warning entries")
    overall_severity: SeverityLevel = Field(..., description="Overall severity for window")
    top_errors: List[str] = Field(default_factory=list, description="Most frequent error messages")
    summary_text: str = Field(..., description="AI-generated summary of the window")
    key_events: List[str] = Field(default_factory=list, description="Key events in this window")


class TrendAnalysis(BaseModel):
    """Analysis of trends over multiple time windows."""

    metric_name: str = Field(..., description="Name of the metric being analyzed")
    direction: TrendDirection = Field(..., description="Overall trend direction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in trend analysis")
    change_percentage: Optional[float] = Field(None, description="Percentage change over period")
    significant_changes: List[str] = Field(default_factory=list, description="Notable changes detected")
    recommendation: Optional[str] = Field(None, description="Recommended action based on trend")


class KeyInsight(BaseModel):
    """A key insight extracted from log analysis."""

    title: str = Field(..., description="Brief title of the insight")
    description: str = Field(..., description="Detailed description")
    severity: SeverityLevel = Field(..., description="Severity level of the insight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the insight")
    affected_windows: List[datetime] = Field(..., description="Time windows where this insight applies")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    related_logs: List[str] = Field(default_factory=list, description="Sample log messages related to insight")


class LogSummaryReport(BaseModel):
    """Complete log summarization report."""

    generated_at: datetime = Field(default_factory=utc_now)
    time_range_start: datetime = Field(..., description="Start of analyzed time range")
    time_range_end: datetime = Field(..., description="End of analyzed time range")
    total_log_entries: int = Field(..., description="Total number of log entries analyzed")
    window_summaries: List[TimeWindowSummary] = Field(..., description="Per-window summaries")
    key_insights: List[KeyInsight] = Field(..., description="Key insights extracted")
    trend_analyses: List[TrendAnalysis] = Field(..., description="Trend analysis results")
    executive_summary: str = Field(..., description="High-level executive summary")
    recommendations: List[str] = Field(default_factory=list, description="Overall recommendations")


class SummarizerConfig(BaseModel):
    """Configuration for the log summarizer."""

    window_size: TimeWindowSize = Field(default=TimeWindowSize.FIFTEEN_MINUTES)
    summary_type: SummaryType = Field(default=SummaryType.TECHNICAL)
    max_insights: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_trend_analysis: bool = Field(default=True)
    enable_ai_summarization: bool = Field(default=True)
    max_summary_length: int = Field(default=500, ge=100, le=2000)


class LogSummarizer:
    """Log summarization engine with time-window analysis and AI integration."""

    def __init__(self, ai_client: Optional[GeminiClient] = None, config: Optional[SummarizerConfig] = None):
        """Initialize the log summarizer.

        Args:
            ai_client: AI client for generating summaries
            config: Summarizer configuration
        """
        self.ai_client = ai_client
        self.config = config or SummarizerConfig()
        self.logger = get_logger(__name__)

        # Time window size mappings
        self._window_deltas = {
            TimeWindowSize.ONE_MINUTE: timedelta(minutes=1),
            TimeWindowSize.FIVE_MINUTES: timedelta(minutes=5),
            TimeWindowSize.FIFTEEN_MINUTES: timedelta(minutes=15),
            TimeWindowSize.THIRTY_MINUTES: timedelta(minutes=30),
            TimeWindowSize.ONE_HOUR: timedelta(hours=1),
            TimeWindowSize.TWO_HOURS: timedelta(hours=2),
            TimeWindowSize.SIX_HOURS: timedelta(hours=6),
            TimeWindowSize.TWELVE_HOURS: timedelta(hours=12),
            TimeWindowSize.ONE_DAY: timedelta(days=1),
        }

    async def summarize_logs(
        self,
        log_entries: List[LogEntry],
        config: Optional[SummarizerConfig] = None
    ) -> LogSummaryReport:
        """Generate a comprehensive summary report from log entries.

        Args:
            log_entries: List of log entries to summarize
            config: Optional configuration override

        Returns:
            Complete summary report
        """
        if not log_entries:
            raise ValueError("No log entries provided for summarization")

        effective_config = config or self.config
        self.logger.info(f"Summarizing {len(log_entries)} log entries with {effective_config.window_size.value} windows")

        # Create time windows
        window_summaries = await self._create_time_windows(log_entries, effective_config)

        # Extract key insights
        key_insights = await self._extract_key_insights(log_entries, window_summaries, effective_config)

        # Analyze trends
        trend_analyses = []
        if effective_config.enable_trend_analysis:
            trend_analyses = await self._analyze_trends(window_summaries, effective_config)

        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            log_entries, window_summaries, key_insights, effective_config
        )

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            key_insights, trend_analyses, effective_config
        )

        return LogSummaryReport(
            time_range_start=min(entry.timestamp for entry in log_entries),
            time_range_end=max(entry.timestamp for entry in log_entries),
            total_log_entries=len(log_entries),
            window_summaries=window_summaries,
            key_insights=key_insights,
            trend_analyses=trend_analyses,
            executive_summary=executive_summary,
            recommendations=recommendations
        )

    async def _create_time_windows(
        self,
        log_entries: List[LogEntry],
        config: SummarizerConfig
    ) -> List[TimeWindowSummary]:
        """Create time-windowed summaries of log entries."""
        if not log_entries:
            return []

        # Sort logs by timestamp
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)
        start_time = sorted_logs[0].timestamp
        end_time = sorted_logs[-1].timestamp

        window_delta = self._window_deltas[config.window_size]
        windows = []

        # Create time windows
        current_start = start_time.replace(second=0, microsecond=0)

        while current_start < end_time:
            current_end = current_start + window_delta

            # Get logs for this window
            window_logs = [
                log for log in sorted_logs
                if current_start <= log.timestamp < current_end
            ]

            if window_logs:  # Only create summaries for windows with logs
                window_summary = await self._summarize_window(
                    window_logs, current_start, current_end, config
                )
                windows.append(window_summary)

            current_start = current_end

        return windows

    async def _summarize_window(
        self,
        window_logs: List[LogEntry],
        start_time: datetime,
        end_time: datetime,
        config: SummarizerConfig
    ) -> TimeWindowSummary:
        """Create a summary for a single time window."""
        error_count = sum(1 for log in window_logs if log.is_error)
        warning_count = sum(1 for log in window_logs if log.is_warning)

        # Determine overall severity
        if error_count > 0:
            if error_count / len(window_logs) > 0.3:  # More than 30% errors
                overall_severity = SeverityLevel.CRITICAL
            elif error_count / len(window_logs) > 0.1:  # More than 10% errors
                overall_severity = SeverityLevel.HIGH
            else:
                overall_severity = SeverityLevel.MEDIUM
        elif warning_count > 0:
            overall_severity = SeverityLevel.MEDIUM if warning_count / len(window_logs) > 0.2 else SeverityLevel.LOW
        else:
            overall_severity = SeverityLevel.LOW

        # Extract top errors
        error_messages = [log.message for log in window_logs if log.is_error]
        top_errors = self._get_top_messages(error_messages, max_count=3)

        # Generate AI summary if enabled
        summary_text = ""
        key_events = []

        if config.enable_ai_summarization and self.ai_client:
            try:
                summary_text = await self._generate_window_summary_ai(window_logs, config)
                key_events = await self._extract_key_events_ai(window_logs, config)
            except Exception as e:
                self.logger.warning(f"AI summarization failed for window: {e}")
                summary_text = self._generate_basic_summary(window_logs, error_count, warning_count)
        else:
            summary_text = self._generate_basic_summary(window_logs, error_count, warning_count)

        return TimeWindowSummary(
            start_time=start_time,
            end_time=end_time,
            window_size=config.window_size,
            log_count=len(window_logs),
            error_count=error_count,
            warning_count=warning_count,
            overall_severity=overall_severity,
            top_errors=top_errors,
            summary_text=summary_text,
            key_events=key_events
        )

    async def _generate_window_summary_ai(
        self,
        window_logs: List[LogEntry],
        config: SummarizerConfig
    ) -> str:
        """Generate AI-powered summary for a time window."""
        if not self.ai_client:
            return self._generate_basic_summary(window_logs, 0, 0)

        # Limit logs for AI processing
        max_logs = 50
        sample_logs = window_logs[:max_logs]

        summary_style = config.summary_type.value
        max_length = min(config.max_summary_length, 300)  # Shorter for window summaries

        return await self.ai_client.summarize_logs(
            sample_logs,
            summary_style=summary_style,
            max_length=max_length
        )

    async def _extract_key_events_ai(
        self,
        window_logs: List[LogEntry],
        config: SummarizerConfig
    ) -> List[str]:
        """Extract key events from window logs using AI."""
        if not self.ai_client or len(window_logs) < 5:
            return []

        try:
            # Use AI to identify key events
            prompt = f"List the 3 most important events from these {len(window_logs)} log entries. Respond with bullet points:"

            # Limit logs for processing
            sample_logs = window_logs[:30]

            response = await self.ai_client.query_logs(
                sample_logs,
                prompt
            )

            # Parse bullet points
            events = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    events.append(line[1:].strip())

            return events[:3]  # Return max 3 events

        except Exception as e:
            self.logger.warning(f"Failed to extract key events: {e}")
            return []

    def _generate_basic_summary(
        self,
        window_logs: List[LogEntry],
        error_count: int,
        warning_count: int
    ) -> str:
        """Generate a basic non-AI summary for a time window."""
        total_logs = len(window_logs)

        if error_count > 0:
            return f"Window contains {total_logs} log entries with {error_count} errors and {warning_count} warnings. Critical issues detected."
        elif warning_count > 0:
            return f"Window contains {total_logs} log entries with {warning_count} warnings. Some issues require attention."
        else:
            return f"Window contains {total_logs} log entries. System operating normally."

    def _get_top_messages(self, messages: List[str], max_count: int = 5) -> List[str]:
        """Get the most frequent messages from a list."""
        if not messages:
            return []

        # Count message frequency
        message_counts = defaultdict(int)
        for message in messages:
            # Truncate long messages
            truncated = message[:100] + "..." if len(message) > 100 else message
            message_counts[truncated] += 1

        # Sort by frequency and return top N
        sorted_messages = sorted(
            message_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [msg for msg, _ in sorted_messages[:max_count]]

    async def _extract_key_insights(
        self,
        log_entries: List[LogEntry],
        window_summaries: List[TimeWindowSummary],
        config: SummarizerConfig
    ) -> List[KeyInsight]:
        """Extract key insights from logs and window summaries."""
        insights = []

        # Analyze error patterns across windows
        error_insights = await self._analyze_error_patterns(window_summaries, config)
        insights.extend(error_insights)

        # Analyze volume patterns
        volume_insights = await self._analyze_volume_patterns(window_summaries, config)
        insights.extend(volume_insights)

        # Analyze severity escalations
        severity_insights = await self._analyze_severity_patterns(window_summaries, config)
        insights.extend(severity_insights)

        # Filter by confidence and limit count
        high_confidence_insights = [
            insight for insight in insights
            if insight.confidence >= config.min_confidence
        ]

        # Sort by severity and confidence
        sorted_insights = sorted(
            high_confidence_insights,
            key=lambda x: (x.severity.value, -x.confidence),
            reverse=True
        )

        return sorted_insights[:config.max_insights]

    async def _analyze_error_patterns(
        self,
        window_summaries: List[TimeWindowSummary],
        config: SummarizerConfig
    ) -> List[KeyInsight]:
        """Analyze error patterns across time windows."""
        insights = []

        if len(window_summaries) < 2:
            return insights

        # Check for persistent errors
        error_windows = [w for w in window_summaries if w.error_count > 0]
        if len(error_windows) >= len(window_summaries) * 0.7:  # 70% of windows have errors
            insights.append(KeyInsight(
                title="Persistent Error Pattern",
                description=f"Errors detected in {len(error_windows)} of {len(window_summaries)} time windows",
                severity=SeverityLevel.HIGH,
                confidence=0.9,
                affected_windows=[w.start_time for w in error_windows],
                recommendation="Investigate recurring error causes and implement fixes"
            ))

        # Check for error spikes
        avg_errors = statistics.mean([w.error_count for w in window_summaries])
        spike_windows = [w for w in window_summaries if w.error_count > avg_errors * 3]

        if spike_windows:
            insights.append(KeyInsight(
                title="Error Rate Spikes",
                description=f"Detected {len(spike_windows)} windows with error rates 3x above average",
                severity=SeverityLevel.MEDIUM,
                confidence=0.8,
                affected_windows=[w.start_time for w in spike_windows],
                recommendation="Review system during spike periods for root causes"
            ))

        return insights

    async def _analyze_volume_patterns(
        self,
        window_summaries: List[TimeWindowSummary],
        config: SummarizerConfig
    ) -> List[KeyInsight]:
        """Analyze log volume patterns."""
        insights = []

        if len(window_summaries) < 3:
            return insights

        volumes = [w.log_count for w in window_summaries]

        # For spike detection, use median instead of mean to avoid skewing
        if len(volumes) >= 3:
            median_volume = statistics.median(volumes)
            # Check for volume spikes - windows that are 5x the median
            spike_threshold = median_volume * 5
            spike_windows = [w for w in window_summaries if w.log_count > spike_threshold]

        if spike_windows:
            insights.append(KeyInsight(
                title="Log Volume Spikes",
                description=f"Detected {len(spike_windows)} windows with 5x normal log volume",
                severity=SeverityLevel.MEDIUM,
                confidence=0.7,
                affected_windows=[w.start_time for w in spike_windows],
                recommendation="Investigate causes of increased logging activity"
            ))

        # Check for suspiciously low volume
        if len(volumes) > 5:
            median_volume = statistics.median(volumes)
            low_threshold = median_volume * 0.1
            low_windows = [w for w in window_summaries if w.log_count < low_threshold]

            if len(low_windows) >= 2:
                insights.append(KeyInsight(
                    title="Reduced Log Activity",
                    description=f"Detected {len(low_windows)} windows with unusually low log volume",
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.6,
                    affected_windows=[w.start_time for w in low_windows],
                    recommendation="Verify system health and logging configuration"
                ))

        return insights

    async def _analyze_severity_patterns(
        self,
        window_summaries: List[TimeWindowSummary],
        config: SummarizerConfig
    ) -> List[KeyInsight]:
        """Analyze severity escalation patterns."""
        insights = []

        if len(window_summaries) < 3:
            return insights

        # Look for severity escalations
        escalations = 0
        for i in range(1, len(window_summaries)):
            if window_summaries[i].overall_severity.value > window_summaries[i - 1].overall_severity.value:
                escalations += 1

        if escalations >= len(window_summaries) * 0.3:  # 30% of transitions are escalations
            insights.append(KeyInsight(
                title="Severity Escalation Pattern",
                description=f"System severity is escalating over time in {escalations} transitions",
                severity=SeverityLevel.HIGH,
                confidence=0.8,
                affected_windows=[w.start_time for w in window_summaries],
                recommendation="Immediate investigation required to prevent further degradation"
            ))

        return insights

    async def _analyze_trends(
        self,
        window_summaries: List[TimeWindowSummary],
        config: SummarizerConfig
    ) -> List[TrendAnalysis]:
        """Analyze trends across time windows."""
        if len(window_summaries) < 3:
            return []

        trends = []

        # Analyze error rate trends
        error_trend = await self._analyze_metric_trend(
            [w.error_count for w in window_summaries],
            "Error Rate"
        )
        trends.append(error_trend)

        # Analyze log volume trends
        volume_trend = await self._analyze_metric_trend(
            [w.log_count for w in window_summaries],
            "Log Volume"
        )
        trends.append(volume_trend)

        # Analyze severity trends - convert to numeric values
        severity_mapping = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }
        severity_values = [severity_mapping.get(w.overall_severity, 1) for w in window_summaries]
        severity_trend = await self._analyze_metric_trend(
            severity_values,
            "Severity Level"
        )
        trends.append(severity_trend)

        return trends

    async def _analyze_metric_trend(
        self,
        values: List[float],
        metric_name: str
    ) -> TrendAnalysis:
        """Analyze trend for a specific metric."""
        if len(values) < 3:
            return TrendAnalysis(
                metric_name=metric_name,
                direction=TrendDirection.UNKNOWN,
                confidence=0.0
            )

        # Calculate trend using simple linear regression approach
        n = len(values)
        x_vals = list(range(n))

        # Calculate slope
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n

        numerator = sum((x_vals[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Calculate correlation coefficient for confidence
        y_variance = sum((values[i] - y_mean) ** 2 for i in range(n))
        if y_variance == 0:
            correlation = 0
        else:
            correlation = abs(numerator) / (denominator ** 0.5 * y_variance ** 0.5)

        # Determine direction
        if abs(slope) < 0.1:  # Essentially flat
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check for volatility
        if len(values) > 3:
            variance = statistics.variance(values)
            mean_val = statistics.mean(values)
            if mean_val > 0 and (variance / mean_val) > 2:  # High coefficient of variation
                direction = TrendDirection.VOLATILE

        # Calculate percentage change
        change_pct = None
        if values[0] != 0:
            change_pct = ((values[-1] - values[0]) / values[0]) * 100

        # Generate recommendation
        recommendation = self._generate_trend_recommendation(
            metric_name, direction, change_pct, correlation
        )

        return TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            confidence=min(correlation, 1.0),
            change_percentage=change_pct,
            recommendation=recommendation
        )

    def _generate_trend_recommendation(
        self,
        metric_name: str,
        direction: TrendDirection,
        change_pct: Optional[float],
        confidence: float
    ) -> Optional[str]:
        """Generate recommendation based on trend analysis."""
        if confidence < 0.5:
            return None

        if metric_name == "Error Rate":
            if direction == TrendDirection.INCREASING:
                return "Error rate is increasing - investigate root causes immediately"
            elif direction == TrendDirection.VOLATILE:
                return "Error rate is unstable - monitor for patterns and investigate spikes"

        elif metric_name == "Log Volume":
            if direction == TrendDirection.INCREASING and change_pct and change_pct > 100:
                return "Log volume increasing rapidly - check for excessive logging or issues"
            elif direction == TrendDirection.DECREASING and change_pct and change_pct < -50:
                return "Log volume decreasing significantly - verify logging configuration"

        elif metric_name == "Severity Level":
            if direction == TrendDirection.INCREASING:
                return "System severity escalating - immediate attention required"

        return None

    async def _generate_executive_summary(
        self,
        log_entries: List[LogEntry],
        window_summaries: List[TimeWindowSummary],
        key_insights: List[KeyInsight],
        config: SummarizerConfig
    ) -> str:
        """Generate an executive summary of the entire analysis."""
        if config.enable_ai_summarization and self.ai_client:
            try:
                return await self._generate_executive_summary_ai(
                    log_entries, window_summaries, key_insights, config
                )
            except Exception as e:
                self.logger.warning(f"AI executive summary generation failed: {e}")

        # Fallback to basic summary
        return self._generate_basic_executive_summary(
            log_entries, window_summaries, key_insights
        )

    async def _generate_executive_summary_ai(
        self,
        log_entries: List[LogEntry],
        window_summaries: List[TimeWindowSummary],
        key_insights: List[KeyInsight],
        config: SummarizerConfig
    ) -> str:
        """Generate AI-powered executive summary."""
        # Sample logs for AI processing
        sample_logs = log_entries[:100]

        return await self.ai_client.summarize_logs(
            sample_logs,
            summary_style="executive",
            max_length=config.max_summary_length
        )

    def _generate_basic_executive_summary(
        self,
        log_entries: List[LogEntry],
        window_summaries: List[TimeWindowSummary],
        key_insights: List[KeyInsight]
    ) -> str:
        """Generate basic executive summary without AI."""
        total_errors = sum(w.error_count for w in window_summaries)
        total_warnings = sum(w.warning_count for w in window_summaries)

        high_severity_insights = [i for i in key_insights if i.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]

        summary = f"Analysis of {len(log_entries)} log entries across {len(window_summaries)} time windows. "

        if total_errors > 0:
            summary += f"Detected {total_errors} errors and {total_warnings} warnings. "

        if high_severity_insights:
            summary += f"Found {len(high_severity_insights)} high-priority issues requiring attention. "
        else:
            summary += "No critical issues detected. "

        summary += "Review detailed analysis for specific recommendations."

        return summary

    async def _generate_recommendations(
        self,
        key_insights: List[KeyInsight],
        trend_analyses: List[TrendAnalysis],
        config: SummarizerConfig
    ) -> List[str]:
        """Generate overall recommendations based on analysis."""
        recommendations = []

        # Add insight recommendations
        for insight in key_insights:
            if insight.recommendation:
                recommendations.append(insight.recommendation)

        # Add trend recommendations
        for trend in trend_analyses:
            if trend.recommendation:
                recommendations.append(trend.recommendation)

        # Add general recommendations based on patterns
        critical_insights = [i for i in key_insights if i.severity == SeverityLevel.CRITICAL]
        if critical_insights:
            recommendations.append("Immediate action required for critical issues - establish incident response")

        high_insights = [i for i in key_insights if i.severity == SeverityLevel.HIGH]
        if len(high_insights) > 3:
            recommendations.append("Multiple high-priority issues detected - prioritize based on business impact")

        return list(set(recommendations))  # Remove duplicates
