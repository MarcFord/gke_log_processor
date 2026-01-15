"""Advanced pattern detection for recurring issues and log analysis."""

import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from ..core.logging import get_logger
from ..core.models import (
    DetectedPattern,
    LogEntry,
    LogLevel,
    PatternType,
    SeverityLevel,
)
from ..core.utils import utc_now

logger = get_logger(__name__)


class PatternDetectionConfig(BaseModel):
    """Configuration for pattern detection algorithms."""

    # Similarity thresholds
    error_similarity_threshold: float = Field(default=0.7, description="Minimum similarity for error grouping")
    cascade_severity_threshold: int = Field(default=2, description="Minimum severity escalation for cascades")

    # Temporal analysis
    spike_multiplier: float = Field(default=3.0, description="Multiplier over average for spike detection")
    spike_minimum_count: int = Field(default=5, description="Minimum error count to consider a spike")
    time_window_minutes: int = Field(default=5, description="Time window for spike analysis (minutes)")

    # Pattern occurrence thresholds
    min_pattern_occurrences: int = Field(default=3, description="Minimum occurrences for pattern detection")
    min_periodic_samples: int = Field(default=5, description="Minimum samples for periodic pattern detection")
    periodic_tolerance: float = Field(default=0.2, description="Tolerance for periodic pattern intervals")

    # Volume analysis
    volume_anomaly_threshold: float = Field(default=2.5, description="Standard deviations for volume anomaly")
    min_volume_for_analysis: int = Field(default=10, description="Minimum logs for volume analysis")

    # Resource pattern detection
    memory_threshold_mb: int = Field(default=100, description="Memory change threshold for patterns (MB)")
    cpu_threshold_percent: float = Field(default=20.0, description="CPU usage threshold for patterns (%)")

    @field_validator("error_similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        """Validate similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v


class PatternSimilarity(BaseModel):
    """Represents similarity between log patterns."""

    pattern1: str = Field(..., description="First pattern")
    pattern2: str = Field(..., description="Second pattern")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    common_keywords: List[str] = Field(default_factory=list, description="Common keywords")
    difference_summary: str = Field(..., description="Summary of differences")


class RecurringIssuePattern(BaseModel):
    """Detailed analysis of a recurring issue pattern."""

    normalized_error: str = Field(..., description="Normalized error pattern")
    occurrence_count: int = Field(..., description="Number of occurrences")
    affected_pods: List[str] = Field(..., description="Pods affected by this pattern")
    affected_namespaces: List[str] = Field(..., description="Namespaces affected")
    affected_clusters: List[str] = Field(..., description="Clusters affected")
    time_span_minutes: int = Field(..., description="Time span from first to last occurrence")
    frequency_per_hour: float = Field(..., description="Average frequency per hour")
    peak_hour: Optional[int] = Field(None, description="Hour of day with most occurrences")
    sample_messages: List[str] = Field(..., description="Sample original messages")
    severity_distribution: Dict[str, int] = Field(..., description="Distribution by severity")
    trend: str = Field(..., description="Trend analysis (increasing/decreasing/stable)")
    impact_score: float = Field(..., description="Calculated impact score (0-100)")


class TemporalPattern(BaseModel):
    """Temporal pattern analysis result."""

    pattern_type: str = Field(..., description="Type of temporal pattern")
    interval_seconds: Optional[float] = Field(None, description="Average interval for periodic patterns")
    interval_variance: Optional[float] = Field(None, description="Variance in intervals")
    peak_times: List[str] = Field(default_factory=list, description="Peak occurrence times")
    duration_minutes: int = Field(..., description="Pattern duration")
    regularity_score: float = Field(..., description="How regular the pattern is (0-1)")
    prediction_confidence: float = Field(..., description="Confidence for future predictions")


class CascadePattern(BaseModel):
    """Cascading failure pattern analysis."""

    cascade_sequence: List[Dict[str, Any]] = Field(..., description="Sequence of cascade events")
    trigger_event: str = Field(..., description="Initial trigger event")
    escalation_time_seconds: int = Field(..., description="Time from trigger to full cascade")
    affected_service_chain: List[str] = Field(..., description="Chain of affected services")
    cascade_depth: int = Field(..., description="Number of escalation levels")
    containment_success: bool = Field(..., description="Whether cascade was contained")
    propagation_rate: float = Field(..., description="How fast the cascade propagated")


class AnomalyPattern(BaseModel):
    """Anomalous behavior pattern analysis."""

    anomaly_type: str = Field(..., description="Type of anomaly detected")
    deviation_score: float = Field(..., description="How much it deviates from normal")
    baseline_metrics: Dict[str, float] = Field(..., description="Normal baseline metrics")
    anomaly_metrics: Dict[str, float] = Field(..., description="Anomalous metrics")
    duration_minutes: int = Field(..., description="Duration of anomaly")
    affected_components: List[str] = Field(..., description="Components showing anomalous behavior")
    potential_causes: List[str] = Field(..., description="Potential root causes")
    recovery_time_minutes: Optional[int] = Field(None, description="Time to recovery if applicable")


class PatternDetectionResult(BaseModel):
    """Complete pattern detection analysis result."""

    analysis_timestamp: datetime = Field(default_factory=utc_now, description="When analysis was performed")
    log_count_analyzed: int = Field(..., description="Number of logs analyzed")
    time_range_analyzed_minutes: int = Field(..., description="Time range of logs analyzed")

    recurring_issues: List[RecurringIssuePattern] = Field(..., description="Recurring issue patterns")
    temporal_patterns: List[TemporalPattern] = Field(..., description="Temporal patterns")
    cascade_patterns: List[CascadePattern] = Field(..., description="Cascade patterns")
    anomaly_patterns: List[AnomalyPattern] = Field(..., description="Anomaly patterns")

    overall_pattern_score: float = Field(..., description="Overall pattern complexity score")
    health_trends: Dict[str, str] = Field(..., description="Health trend analysis")
    recommendations: List[str] = Field(..., description="Pattern-based recommendations")

    @field_validator("overall_pattern_score")
    @classmethod
    def validate_pattern_score(cls, v):
        """Validate pattern score."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("Pattern score must be between 0.0 and 100.0")
        return v


class AdvancedPatternDetector:
    """Advanced pattern detection engine for recurring issue identification."""

    def __init__(self, config: Optional[PatternDetectionConfig] = None):
        """Initialize the pattern detector.

        Args:
            config: Pattern detection configuration
        """
        self.config = config or PatternDetectionConfig()
        self._pattern_cache: Dict[str, Any] = {}
        logger.info("Initialized AdvancedPatternDetector with configuration")

    def detect_all_patterns(self, log_entries: List[LogEntry]) -> PatternDetectionResult:
        """Comprehensive pattern detection analysis.

        Args:
            log_entries: List of log entries to analyze

        Returns:
            Complete pattern detection results
        """
        logger.info(f"Starting comprehensive pattern analysis on {len(log_entries)} log entries")

        if not log_entries:
            return self._create_empty_result()

        # Sort logs by timestamp for temporal analysis
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

        # Calculate time range
        time_range_minutes = self._calculate_time_range_minutes(sorted_logs)

        # Detect different pattern types
        recurring_issues = self.detect_recurring_issues(sorted_logs)
        temporal_patterns = self.detect_temporal_patterns(sorted_logs)
        cascade_patterns = self.detect_cascade_patterns(sorted_logs)
        anomaly_patterns = self.detect_anomaly_patterns(sorted_logs)

        # Calculate overall metrics
        overall_score = self._calculate_overall_pattern_score(
            recurring_issues, temporal_patterns, cascade_patterns, anomaly_patterns
        )

        health_trends = self._analyze_health_trends(sorted_logs)
        recommendations = self._generate_recommendations(
            recurring_issues, temporal_patterns, cascade_patterns, anomaly_patterns
        )

        result = PatternDetectionResult(
            log_count_analyzed=len(log_entries),
            time_range_analyzed_minutes=time_range_minutes,
            recurring_issues=recurring_issues,
            temporal_patterns=temporal_patterns,
            cascade_patterns=cascade_patterns,
            anomaly_patterns=anomaly_patterns,
            overall_pattern_score=overall_score,
            health_trends=health_trends,
            recommendations=recommendations
        )

        logger.info(f"Pattern analysis complete: {len(recurring_issues)} recurring, "
                    f"{len(temporal_patterns)} temporal, {len(cascade_patterns)} cascade, "
                    f"{len(anomaly_patterns)} anomaly patterns detected")

        return result

    def detect_recurring_issues(self, log_entries: List[LogEntry]) -> List[RecurringIssuePattern]:
        """Detect recurring issue patterns with detailed analysis.

        Args:
            log_entries: Log entries to analyze

        Returns:
            List of recurring issue patterns
        """
        patterns = []

        # Filter error and warning logs
        issue_logs = [
            log for log in log_entries
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.WARNING]
        ]

        if len(issue_logs) < self.config.min_pattern_occurrences:
            return patterns

        # Group similar error messages using advanced similarity
        error_groups = self._group_similar_errors_advanced(issue_logs)

        for normalized_error, group_logs in error_groups.items():
            if len(group_logs) >= self.config.min_pattern_occurrences:
                pattern = self._analyze_recurring_pattern(normalized_error, group_logs)
                patterns.append(pattern)

        # Sort by impact score (highest first)
        patterns.sort(key=lambda p: p.impact_score, reverse=True)

        return patterns

    def detect_temporal_patterns(self, log_entries: List[LogEntry]) -> List[TemporalPattern]:
        """Detect temporal patterns in log behavior.

        Args:
            log_entries: Log entries to analyze

        Returns:
            List of temporal patterns
        """
        patterns = []

        if len(log_entries) < self.config.min_periodic_samples:
            return patterns

        # Detect different types of temporal patterns
        patterns.extend(self._detect_periodic_patterns(log_entries))
        patterns.extend(self._detect_spike_patterns(log_entries))
        patterns.extend(self._detect_time_of_day_patterns(log_entries))
        patterns.extend(self._detect_burst_patterns(log_entries))

        return patterns

    def detect_cascade_patterns(self, log_entries: List[LogEntry]) -> List[CascadePattern]:
        """Detect cascading failure patterns.

        Args:
            log_entries: Log entries to analyze

        Returns:
            List of cascade patterns
        """
        patterns = []

        # Group by namespace and pod for cascade analysis
        namespace_logs = defaultdict(list)
        for log in log_entries:
            namespace_logs[log.namespace].append(log)

        for namespace, logs in namespace_logs.items():
            cascades = self._analyze_cascade_sequences(logs)
            patterns.extend(cascades)

        return patterns

    def detect_anomaly_patterns(self, log_entries: List[LogEntry]) -> List[AnomalyPattern]:
        """Detect anomalous patterns in log behavior.

        Args:
            log_entries: Log entries to analyze

        Returns:
            List of anomaly patterns
        """
        patterns = []

        if len(log_entries) < self.config.min_volume_for_analysis:
            return patterns

        # Detect different types of anomalies
        patterns.extend(self._detect_volume_anomalies(log_entries))
        patterns.extend(self._detect_severity_anomalies(log_entries))
        patterns.extend(self._detect_source_anomalies(log_entries))
        patterns.extend(self._detect_message_anomalies(log_entries))

        return patterns

    def _create_empty_result(self) -> PatternDetectionResult:
        """Create empty pattern detection result."""
        return PatternDetectionResult(
            log_count_analyzed=0,
            time_range_analyzed_minutes=0,
            recurring_issues=[],
            temporal_patterns=[],
            cascade_patterns=[],
            anomaly_patterns=[],
            overall_pattern_score=0.0,
            health_trends={"overall": "unknown"},
            recommendations=["No logs available for analysis"]
        )

    def _calculate_time_range_minutes(self, sorted_logs: List[LogEntry]) -> int:
        """Calculate time range of logs in minutes."""
        if len(sorted_logs) < 2:
            return 0

        time_diff = sorted_logs[-1].timestamp - sorted_logs[0].timestamp
        return int(time_diff.total_seconds() / 60)

    def _group_similar_errors_advanced(self, issue_logs: List[LogEntry]) -> Dict[str, List[LogEntry]]:
        """Group similar errors using advanced similarity algorithms."""
        groups = defaultdict(list)

        for log in issue_logs:
            normalized = self._normalize_error_message(log.message)

            # Find existing similar group
            best_match = None
            best_similarity = 0.0

            for existing_normalized in groups.keys():
                similarity = self._calculate_message_similarity(normalized, existing_normalized)
                if similarity > best_similarity and similarity >= self.config.error_similarity_threshold:
                    best_similarity = similarity
                    best_match = existing_normalized

            if best_match:
                groups[best_match].append(log)
            else:
                groups[normalized].append(log)

        return dict(groups)

    def _normalize_error_message(self, message: str) -> str:
        """Advanced error message normalization."""
        # Convert to lowercase for consistency
        normalized = message.lower()

        # Remove common variable elements
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}[.\d]*[z]?', '[timestamp]', normalized)
        normalized = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '[uuid]', normalized)
        normalized = re.sub(r'\b[0-9a-f]{32,64}\b', '[hash]', normalized)
        normalized = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '[ip]', normalized)
        normalized = re.sub(r'\b\d+\b', '[num]', normalized)
        normalized = re.sub(r'\b0x[0-9a-f]+\b', '[hex]', normalized)
        normalized = re.sub(r'[\'"][^\'"]*[\'"]', '[string]', normalized)
        normalized = re.sub(r'/[a-zA-Z0-9/_.-]+', '[path]', normalized)
        normalized = re.sub(r'https?://[^\s]+', '[url]', normalized)

        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _calculate_message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate similarity between two normalized messages."""
        # Use sequence matcher for basic similarity
        seq_similarity = SequenceMatcher(None, msg1, msg2).ratio()

        # Calculate keyword overlap
        words1 = set(msg1.split())
        words2 = set(msg2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        keyword_similarity = len(intersection) / len(union)

        # Combine similarities with weights
        return 0.6 * seq_similarity + 0.4 * keyword_similarity

    def _analyze_recurring_pattern(self, normalized_error: str, group_logs: List[LogEntry]) -> RecurringIssuePattern:
        """Analyze a group of similar logs to create a recurring pattern."""
        # Sort logs by timestamp
        sorted_logs = sorted(group_logs, key=lambda x: x.timestamp)

        # Calculate basic metrics
        time_span = (sorted_logs[-1].timestamp - sorted_logs[0].timestamp).total_seconds() / 60
        frequency_per_hour = (len(group_logs) / max(time_span / 60, 1 / 60))  # Avoid division by zero

        # Analyze affected components
        affected_pods = list(set(log.pod_name for log in group_logs))
        affected_namespaces = list(set(log.namespace for log in group_logs))
        affected_clusters = list(set(log.cluster for log in group_logs))

        # Analyze severity distribution
        severity_counts = Counter()
        for log in group_logs:
            if log.level:
                severity_counts[log.level.value] += 1

        # Analyze peak hour
        hour_counts = Counter()
        for log in group_logs:
            hour_counts[log.timestamp.hour] += 1

        peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

        # Calculate trend
        trend = self._calculate_trend(sorted_logs)

        # Calculate impact score
        impact_score = self._calculate_impact_score(group_logs, len(affected_pods), frequency_per_hour)

        return RecurringIssuePattern(
            normalized_error=normalized_error,
            occurrence_count=len(group_logs),
            affected_pods=affected_pods,
            affected_namespaces=affected_namespaces,
            affected_clusters=affected_clusters,
            time_span_minutes=int(time_span),
            frequency_per_hour=round(frequency_per_hour, 2),
            peak_hour=peak_hour,
            sample_messages=[log.message for log in sorted_logs[:3]],
            severity_distribution=dict(severity_counts),
            trend=trend,
            impact_score=round(impact_score, 2)
        )

    def _calculate_trend(self, sorted_logs: List[LogEntry]) -> str:
        """Calculate trend in log occurrences over time."""
        if len(sorted_logs) < 4:
            return "stable"

        # Split into time windows and count occurrences
        total_time = (sorted_logs[-1].timestamp - sorted_logs[0].timestamp).total_seconds()
        window_size = total_time / 3  # Split into 3 windows

        window_counts = [0, 0, 0]

        for log in sorted_logs:
            time_from_start = (log.timestamp - sorted_logs[0].timestamp).total_seconds()
            window_index = min(int(time_from_start / window_size), 2)
            window_counts[window_index] += 1

        # Analyze trend
        first_half_avg = (window_counts[0] + window_counts[1]) / 2
        last_half_avg = (window_counts[1] + window_counts[2]) / 2

        if last_half_avg > first_half_avg * 1.2:
            return "increasing"
        elif last_half_avg < first_half_avg * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _calculate_impact_score(self, logs: List[LogEntry], pod_count: int, frequency: float) -> float:
        """Calculate impact score for a recurring pattern."""
        # Base score from frequency
        freq_score = min(frequency * 5, 50)  # Max 50 points for frequency

        # Pod spread score
        pod_score = min(pod_count * 5, 25)  # Max 25 points for pod spread

        # Severity score
        severity_score = 0
        for log in logs:
            if log.level == LogLevel.CRITICAL:
                severity_score += 3
            elif log.level == LogLevel.ERROR:
                severity_score += 2
            elif log.level == LogLevel.WARNING:
                severity_score += 1

        severity_score = min(severity_score / len(logs) * 25, 25)  # Max 25 points for severity

        return freq_score + pod_score + severity_score

    def _detect_periodic_patterns(self, log_entries: List[LogEntry]) -> List[TemporalPattern]:
        """Detect periodic patterns in log timing."""
        patterns = []

        # Filter error logs for periodic analysis
        error_logs = [log for log in log_entries if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]

        if len(error_logs) < self.config.min_periodic_samples:
            return patterns

        # Calculate intervals between errors
        intervals = []
        for i in range(1, len(error_logs)):
            interval = (error_logs[i].timestamp - error_logs[i - 1].timestamp).total_seconds()
            intervals.append(interval)

        if not intervals:
            return patterns

        # Check for periodicity
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5

        # Calculate regularity score
        tolerance = avg_interval * self.config.periodic_tolerance
        regular_intervals = sum(1 for interval in intervals if abs(interval - avg_interval) <= tolerance)
        regularity_score = regular_intervals / len(intervals)

        if regularity_score >= 0.7:  # 70% of intervals are regular
            patterns.append(TemporalPattern(
                pattern_type="periodic_errors",
                interval_seconds=round(avg_interval, 2),
                interval_variance=round(variance, 2),
                peak_times=[],
                duration_minutes=self._calculate_time_range_minutes(error_logs),
                regularity_score=round(regularity_score, 2),
                prediction_confidence=min(regularity_score * 1.2, 1.0)
            ))

        return patterns

    def _detect_spike_patterns(self, log_entries: List[LogEntry]) -> List[TemporalPattern]:
        """Detect spike patterns in log volume."""
        patterns = []

        # Create time windows
        window_size = timedelta(minutes=self.config.time_window_minutes)
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

        if len(sorted_logs) < self.config.spike_minimum_count:
            return patterns

        # Calculate log volume in time windows
        windows = []
        current_start = sorted_logs[0].timestamp
        end_time = sorted_logs[-1].timestamp

        while current_start <= end_time:
            window_end = current_start + window_size
            window_logs = [
                log for log in sorted_logs
                if current_start <= log.timestamp < window_end
            ]
            windows.append((current_start, len(window_logs)))
            current_start = window_end

        if len(windows) < 3:
            return patterns

        # Analyze for spikes
        volumes = [count for _, count in windows]
        avg_volume = sum(volumes) / len(volumes)
        threshold = avg_volume * self.config.spike_multiplier

        spike_windows = [(start, count) for start, count in windows if count >=
                         threshold and count >= self.config.spike_minimum_count]

        if spike_windows:
            patterns.append(TemporalPattern(
                pattern_type="volume_spike",
                interval_seconds=None,
                interval_variance=None,
                peak_times=[start.strftime("%H:%M:%S") for start, _ in spike_windows],
                duration_minutes=len(spike_windows) * self.config.time_window_minutes,
                regularity_score=0.0,
                prediction_confidence=0.6
            ))

        return patterns

    def _detect_time_of_day_patterns(self, log_entries: List[LogEntry]) -> List[TemporalPattern]:
        """Detect time-of-day patterns in logs."""
        patterns = []

        # Count logs by hour of day
        hour_counts = Counter()
        for log in log_entries:
            hour_counts[log.timestamp.hour] += 1

        if not hour_counts:
            return patterns

        # Find peak hours
        total_logs = sum(hour_counts.values())
        avg_per_hour = total_logs / 24

        peak_hours = []
        for hour, count in hour_counts.items():
            if count > avg_per_hour * 2:  # More than 2x average
                peak_hours.append(f"{hour:02d}:00")

        if peak_hours:
            patterns.append(TemporalPattern(
                pattern_type="time_of_day",
                interval_seconds=None,
                interval_variance=None,
                peak_times=peak_hours,
                duration_minutes=self._calculate_time_range_minutes(log_entries),
                regularity_score=0.8,
                prediction_confidence=0.9
            ))

        return patterns

    def _detect_burst_patterns(self, log_entries: List[LogEntry]) -> List[TemporalPattern]:
        """Detect burst patterns in logs."""
        patterns = []

        # Sort logs by timestamp
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

        if len(sorted_logs) < 10:
            return patterns

        # Look for bursts (clusters of logs in short time periods)
        burst_threshold = 5  # logs within 60 seconds
        burst_window = timedelta(seconds=60)

        bursts = []
        i = 0
        while i < len(sorted_logs):
            window_end = sorted_logs[i].timestamp + burst_window
            burst_logs = []

            j = i
            while j < len(sorted_logs) and sorted_logs[j].timestamp <= window_end:
                burst_logs.append(sorted_logs[j])
                j += 1

            if len(burst_logs) >= burst_threshold:
                bursts.append((sorted_logs[i].timestamp, len(burst_logs)))
                i = j
            else:
                i += 1

        if len(bursts) >= 3:
            patterns.append(TemporalPattern(
                pattern_type="burst_pattern",
                interval_seconds=None,
                interval_variance=None,
                peak_times=[burst[0].strftime("%H:%M:%S") for burst in bursts[:5]],
                duration_minutes=self._calculate_time_range_minutes(sorted_logs),
                regularity_score=0.5,
                prediction_confidence=0.4
            ))

        return patterns

    def _analyze_cascade_sequences(self, namespace_logs: List[LogEntry]) -> List[CascadePattern]:
        """Analyze cascading failure sequences."""
        patterns = []

        # Sort logs by timestamp
        sorted_logs = sorted(namespace_logs, key=lambda x: x.timestamp)

        if len(sorted_logs) < 5:
            return patterns

        # Look for escalating severity patterns
        cascade_sequences = []
        current_sequence = []
        current_severity = SeverityLevel.LOW

        for log in sorted_logs:
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                # Determine log severity level
                if log.level == LogLevel.CRITICAL:
                    log_severity = SeverityLevel.CRITICAL
                elif log.level == LogLevel.ERROR:
                    log_severity = SeverityLevel.HIGH
                else:
                    log_severity = SeverityLevel.MEDIUM

                # Check if this escalates the current sequence
                if len(current_sequence) == 0 or log_severity.value >= current_severity.value:
                    current_sequence.append({
                        'timestamp': log.timestamp,
                        'severity': log_severity,
                        'pod': log.pod_name,
                        'message': log.message[:100]  # Truncate for storage
                    })
                    current_severity = log_severity
                else:
                    # Sequence broken, check if previous sequence qualifies as cascade
                    if len(current_sequence) >= 3:
                        cascade_sequences.append(current_sequence)

                    # Start new sequence
                    current_sequence = [{
                        'timestamp': log.timestamp,
                        'severity': log_severity,
                        'pod': log.pod_name,
                        'message': log.message[:100]
                    }]
                    current_severity = log_severity

        # Check final sequence
        if len(current_sequence) >= 3:
            cascade_sequences.append(current_sequence)

        # Create cascade patterns from sequences
        for sequence in cascade_sequences:
            if len(sequence) >= 3:
                escalation_time = (sequence[-1]['timestamp'] - sequence[0]['timestamp']).total_seconds()
                affected_pods = list(set(event['pod'] for event in sequence))

                patterns.append(CascadePattern(
                    cascade_sequence=sequence,
                    trigger_event=sequence[0]['message'],
                    escalation_time_seconds=int(escalation_time),
                    affected_service_chain=affected_pods,
                    cascade_depth=len(sequence),
                    containment_success=sequence[-1]['severity'] != SeverityLevel.CRITICAL,
                    propagation_rate=round(len(sequence) / max(escalation_time / 60, 1), 2)
                ))

        return patterns

    def _detect_volume_anomalies(self, log_entries: List[LogEntry]) -> List[AnomalyPattern]:
        """Detect volume anomalies in logs."""
        patterns = []

        if len(log_entries) < self.config.min_volume_for_analysis:
            return patterns

        # Group logs by time windows
        window_size = timedelta(minutes=self.config.time_window_minutes)
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

        # Calculate volume per window
        volumes = []
        current_start = sorted_logs[0].timestamp
        end_time = sorted_logs[-1].timestamp

        while current_start <= end_time:
            window_end = current_start + window_size
            window_logs = [
                log for log in sorted_logs
                if current_start <= log.timestamp < window_end
            ]
            volumes.append(len(window_logs))
            current_start = window_end

        if len(volumes) < 5:
            return patterns

        # Calculate statistics
        mean_volume = sum(volumes) / len(volumes)
        variance = sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
        std_dev = variance ** 0.5

        # Find anomalies
        threshold = mean_volume + (self.config.volume_anomaly_threshold * std_dev)
        anomalous_volumes = [v for v in volumes if v > threshold]

        if anomalous_volumes:
            max_anomaly = max(anomalous_volumes)
            deviation_score = (max_anomaly - mean_volume) / std_dev if std_dev > 0 else 0

            patterns.append(AnomalyPattern(
                anomaly_type="volume_spike",
                deviation_score=round(deviation_score, 2),
                baseline_metrics={'average_volume': round(mean_volume, 2), 'std_dev': round(std_dev, 2)},
                anomaly_metrics={'max_volume': max_anomaly, 'anomaly_count': len(anomalous_volumes)},
                duration_minutes=len(anomalous_volumes) * self.config.time_window_minutes,
                affected_components=['logging_system'],
                potential_causes=['traffic_spike', 'error_burst', 'system_instability'],
                recovery_time_minutes=None
            ))

        return patterns

    def _detect_severity_anomalies(self, log_entries: List[LogEntry]) -> List[AnomalyPattern]:
        """Detect severity level anomalies."""
        patterns = []

        # Count by severity level
        severity_counts = Counter()
        for log in log_entries:
            if log.level:
                severity_counts[log.level.value] += 1

        total_logs = len(log_entries)
        if total_logs < 20:
            return patterns

        # Check for abnormal error ratios
        error_count = severity_counts.get('ERROR', 0) + severity_counts.get('CRITICAL', 0)
        error_ratio = error_count / total_logs

        # Normal error ratio is typically < 10%
        if error_ratio > 0.3:  # More than 30% errors
            patterns.append(AnomalyPattern(
                anomaly_type="high_error_rate",
                deviation_score=round(error_ratio * 10, 2),
                baseline_metrics={'normal_error_ratio': 0.1},
                anomaly_metrics={'error_ratio': round(error_ratio, 3), 'error_count': error_count},
                duration_minutes=self._calculate_time_range_minutes(log_entries),
                affected_components=['application'],
                potential_causes=['service_degradation', 'configuration_error', 'dependency_failure'],
                recovery_time_minutes=None
            ))

        return patterns

    def _detect_source_anomalies(self, log_entries: List[LogEntry]) -> List[AnomalyPattern]:
        """Detect anomalies in log sources."""
        patterns = []

        # Count logs per pod
        pod_counts = Counter()
        for log in log_entries:
            pod_counts[log.pod_name] += 1

        if len(pod_counts) < 3:
            return patterns

        # Calculate statistics
        counts = list(pod_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        std_dev = variance ** 0.5

        # Find pods with anomalous log volumes
        threshold_high = mean_count + (2 * std_dev)
        threshold_low = max(0, mean_count - (2 * std_dev))

        high_volume_pods = [pod for pod, count in pod_counts.items() if count > threshold_high]
        low_volume_pods = [pod for pod, count in pod_counts.items() if count < threshold_low and count > 0]

        if high_volume_pods:
            patterns.append(AnomalyPattern(
                anomaly_type="pod_high_volume",
                deviation_score=round(max(pod_counts[pod] for pod in high_volume_pods) / mean_count, 2),
                baseline_metrics={'average_pod_logs': round(mean_count, 2)},
                anomaly_metrics={'high_volume_pods': len(high_volume_pods)},
                duration_minutes=self._calculate_time_range_minutes(log_entries),
                affected_components=high_volume_pods,
                potential_causes=['pod_restart_loop', 'application_error', 'verbose_logging'],
                recovery_time_minutes=None
            ))

        return patterns

    def _detect_message_anomalies(self, log_entries: List[LogEntry]) -> List[AnomalyPattern]:
        """Detect anomalies in message patterns."""
        patterns = []

        # Count unique message patterns
        message_patterns = Counter()
        for log in log_entries:
            normalized = self._normalize_error_message(log.message)
            message_patterns[normalized] += 1

        if len(message_patterns) < 5:
            return patterns

        # Check message diversity (too many unique messages might indicate instability)
        total_messages = sum(message_patterns.values())
        unique_ratio = len(message_patterns) / total_messages

        if unique_ratio > 0.8:  # More than 80% unique messages
            patterns.append(AnomalyPattern(
                anomaly_type="message_diversity",
                deviation_score=round(unique_ratio * 10, 2),
                baseline_metrics={'normal_diversity_ratio': 0.3},
                anomaly_metrics={'unique_ratio': round(unique_ratio, 3), 'unique_patterns': len(message_patterns)},
                duration_minutes=self._calculate_time_range_minutes(log_entries),
                affected_components=['application_logic'],
                potential_causes=['random_errors', 'data_corruption', 'unstable_system'],
                recovery_time_minutes=None
            ))

        return patterns

    def _calculate_overall_pattern_score(self, recurring: List[RecurringIssuePattern],
                                         temporal: List[TemporalPattern], cascade: List[CascadePattern],
                                         anomaly: List[AnomalyPattern]) -> float:
        """Calculate overall pattern complexity score."""
        score = 0.0

        # Recurring issues impact (0-40 points)
        for pattern in recurring:
            score += min(pattern.impact_score / 5, 8)  # Max 8 points per pattern
        score = min(score, 40)

        # Temporal patterns impact (0-25 points)
        score += min(len(temporal) * 5, 25)

        # Cascade patterns impact (0-25 points)
        score += min(len(cascade) * 8, 25)

        # Anomaly patterns impact (0-10 points)
        score += min(len(anomaly) * 2, 10)

        return round(score, 2)

    def _analyze_health_trends(self, sorted_logs: List[LogEntry]) -> Dict[str, str]:
        """Analyze overall health trends from logs."""
        if len(sorted_logs) < 10:
            return {"overall": "insufficient_data"}

        # Split logs into halves for trend analysis
        mid_point = len(sorted_logs) // 2
        first_half = sorted_logs[:mid_point]
        second_half = sorted_logs[mid_point:]

        # Count errors in each half
        first_errors = sum(1 for log in first_half if log.level in [LogLevel.ERROR, LogLevel.CRITICAL])
        second_errors = sum(1 for log in second_half if log.level in [LogLevel.ERROR, LogLevel.CRITICAL])

        # Calculate error rates
        first_rate = first_errors / len(first_half)
        second_rate = second_errors / len(second_half)

        if second_rate > first_rate * 1.5:
            overall_trend = "degrading"
        elif second_rate < first_rate * 0.7:
            overall_trend = "improving"
        else:
            overall_trend = "stable"

        return {
            "overall": overall_trend,
            "error_rate_trend": f"first_half: {first_rate:.2%}, second_half: {second_rate:.2%}"
        }

    def _generate_recommendations(self, recurring: List[RecurringIssuePattern],
                                  temporal: List[TemporalPattern], cascade: List[CascadePattern],
                                  anomaly: List[AnomalyPattern]) -> List[str]:
        """Generate actionable recommendations based on detected patterns."""
        recommendations = []

        # Recommendations for recurring issues
        high_impact_recurring = [p for p in recurring if p.impact_score > 50]
        if high_impact_recurring:
            recommendations.append(f"Address {len(high_impact_recurring)} high-impact recurring issues immediately")

        frequent_errors = [p for p in recurring if p.frequency_per_hour > 10]
        if frequent_errors:
            recommendations.append("Investigate frequent error patterns occurring >10 times/hour")

        # Recommendations for temporal patterns
        periodic_patterns = [p for p in temporal if p.pattern_type == "periodic_errors"]
        if periodic_patterns:
            recommendations.append("Review scheduled tasks or cron jobs causing periodic errors")

        spike_patterns = [p for p in temporal if p.pattern_type == "volume_spike"]
        if spike_patterns:
            recommendations.append("Implement rate limiting or auto-scaling for volume spikes")

        # Recommendations for cascades
        if cascade:
            recommendations.append("Implement circuit breakers to prevent cascade failures")
            recommendations.append("Review service dependencies and failure isolation")

        # Recommendations for anomalies
        volume_anomalies = [p for p in anomaly if p.anomaly_type == "volume_spike"]
        if volume_anomalies:
            recommendations.append("Monitor and alert on unusual log volume patterns")

        if not recommendations:
            recommendations.append("No critical patterns detected - system appears stable")

        return recommendations
