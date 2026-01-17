"""Log analysis engine for intelligent log processing and pattern detection."""

import asyncio
import inspect
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rich.text import Text

from ..core.exceptions import LogProcessingError
from ..core.logging import get_logger
from ..core.models import (
    AIAnalysisResult,
    DetectedPattern,
    LogEntry,
    LogLevel,
    PatternType,
    QueryConfig,
    QueryRequest,
    QueryResponse,
    QueryType,
    SeverityLevel,
)
from ..core.utils import utc_now
from .client import GeminiClient, GeminiConfig
from .highlighter import HighlighterConfig, HighlightTheme, SeverityHighlighter
from .summarizer import (
    LogSummarizer,
    LogSummaryReport,
    SummarizerConfig,
    SummaryType,
    TimeWindowSize,
)

logger = get_logger(__name__)


class SeverityDetectionAlgorithm:
    """Advanced severity detection algorithms for log analysis."""

    # Keywords for different severity levels
    CRITICAL_KEYWORDS = {
        "fatal", "critical", "emergency", "panic", "abort", "crashed",
        "segmentation fault", "out of memory", "core dump", "deadlock",
        "corrupted", "unrecoverable", "catastrophic", "disaster"
    }

    HIGH_KEYWORDS = {
        "error", "exception", "failed", "failure", "timeout", "refused",
        "denied", "forbidden", "unauthorized", "invalid", "broken",
        "unavailable", "unreachable", "connection lost", "permission denied"
    }

    MEDIUM_KEYWORDS = {
        "warning", "warn", "deprecated", "slow", "retry", "retrying",
        "fallback", "degraded", "limited", "throttled", "delayed"
    }

    LOW_KEYWORDS = {
        "notice", "info", "debug", "trace", "verbose", "started",
        "stopped", "completed", "successful", "ok", "ready"
    }

    @classmethod
    def detect_severity_by_keywords(cls, message: str) -> SeverityLevel:
        """Detect severity based on keyword analysis."""
        message_lower = message.lower()

        # Check for critical keywords
        if any(keyword in message_lower for keyword in cls.CRITICAL_KEYWORDS):
            return SeverityLevel.CRITICAL

        # Check for high severity keywords
        if any(keyword in message_lower for keyword in cls.HIGH_KEYWORDS):
            return SeverityLevel.HIGH

        # Check for medium severity keywords
        if any(keyword in message_lower for keyword in cls.MEDIUM_KEYWORDS):
            return SeverityLevel.MEDIUM

        # Check for low severity keywords
        if any(keyword in message_lower for keyword in cls.LOW_KEYWORDS):
            return SeverityLevel.LOW

        return SeverityLevel.LOW

    @classmethod
    def detect_severity_by_patterns(cls, message: str) -> SeverityLevel:
        """Detect severity using regex patterns."""
        # Stack traces indicate high severity
        if re.search(r'at\s+\w+\.\w+\([^)]*\)', message):
            return SeverityLevel.HIGH

        # HTTP error codes
        if re.search(r'[45]\d{2}\s+(error|status)', message, re.IGNORECASE):
            return SeverityLevel.HIGH

        # Exception patterns
        if re.search(r'(\w+Exception:|Error:|Exception in)', message):
            return SeverityLevel.HIGH

        # Memory/resource patterns
        if re.search(r'out of (memory|disk|space)|memory leak|OutOfMemoryError', message, re.IGNORECASE):
            return SeverityLevel.CRITICAL

        # Connection patterns
        if re.search(r'connection\s+(refused|timeout|lost|failed)', message, re.IGNORECASE):
            return SeverityLevel.HIGH

        return SeverityLevel.LOW

    @classmethod
    def detect_severity_by_log_level(cls, log_entry: LogEntry) -> SeverityLevel:
        """Map log levels to severity levels."""
        if not log_entry.level:
            return SeverityLevel.LOW

        level_mapping = {
            LogLevel.CRITICAL: SeverityLevel.CRITICAL,
            LogLevel.ERROR: SeverityLevel.HIGH,
            LogLevel.WARNING: SeverityLevel.MEDIUM,
            LogLevel.INFO: SeverityLevel.LOW,
            LogLevel.DEBUG: SeverityLevel.LOW,
            LogLevel.TRACE: SeverityLevel.LOW,
        }

        return level_mapping.get(log_entry.level, SeverityLevel.LOW)

    @classmethod
    def detect_combined_severity(cls, log_entry: LogEntry) -> SeverityLevel:
        """Combine multiple detection methods for final severity."""
        keyword_severity = cls.detect_severity_by_keywords(log_entry.message)
        pattern_severity = cls.detect_severity_by_patterns(log_entry.message)
        level_severity = cls.detect_severity_by_log_level(log_entry)

        # Take the highest severity from all methods
        severities = [keyword_severity, pattern_severity, level_severity]
        severity_values = {
            SeverityLevel.LOW: 0,
            SeverityLevel.MEDIUM: 1,
            SeverityLevel.HIGH: 2,
            SeverityLevel.CRITICAL: 3,
        }

        highest = max(severities, key=lambda s: severity_values[s])
        return highest


class PatternRecognitionEngine:
    """Advanced pattern recognition for log analysis."""

    def __init__(self):
        self.patterns_cache: Dict[str, List[DetectedPattern]] = {}

    def detect_error_patterns(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect recurring error patterns."""
        patterns = []
        error_logs = [log for log in log_entries if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]

        if not error_logs:
            return patterns

        # Group similar error messages
        error_groups = self._group_similar_errors(error_logs)

        for error_type, logs in error_groups.items():
            if len(logs) >= 3:  # Minimum threshold for pattern detection
                confidence = min(0.95, len(logs) / len(error_logs))
                severity = SeverityDetectionAlgorithm.detect_combined_severity(logs[0])

                patterns.append(DetectedPattern(
                    type=PatternType.ERROR_PATTERN,
                    pattern=f"Recurring error pattern: {error_type}",
                    confidence=confidence,
                    occurrence_count=len(logs),
                    first_seen=min(log.timestamp for log in logs),
                    last_seen=max(log.timestamp for log in logs),
                    affected_pods=list(set(log.pod_name for log in logs)),
                    sample_messages=[log.message for log in logs[:3]],
                    severity=severity,
                ))

        return patterns

    def detect_temporal_patterns(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect time-based patterns like spikes or cycles."""
        patterns = []

        # Sort logs by timestamp
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

        if len(sorted_logs) < 10:
            return patterns

        # Detect error spikes
        error_spikes = self._detect_error_spikes(sorted_logs)
        patterns.extend(error_spikes)

        # Detect periodic patterns
        periodic_patterns = self._detect_periodic_patterns(sorted_logs)
        patterns.extend(periodic_patterns)

        return patterns

    def detect_cascade_patterns(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect cascading failure patterns."""
        patterns = []

        # Group by pod and sort by timestamp
        pod_logs = defaultdict(list)
        for log in log_entries:
            pod_logs[log.pod_name].append(log)

        for pod_name, logs in pod_logs.items():
            logs.sort(key=lambda x: x.timestamp)

            # Look for sequences of escalating errors
            cascades = self._detect_cascade_sequences(logs)
            patterns.extend(cascades)

        return patterns

    def detect_anomaly_patterns(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect anomalous patterns in log behavior."""
        patterns = []

        # Detect unusual volume spikes
        volume_anomalies = self._detect_volume_anomalies(log_entries)
        patterns.extend(volume_anomalies)

        # Detect unusual pod behavior
        pod_anomalies = self._detect_pod_anomalies(log_entries)
        patterns.extend(pod_anomalies)

        return patterns

    def _group_similar_errors(self, error_logs: List[LogEntry]) -> Dict[str, List[LogEntry]]:
        """Group similar error messages together."""
        groups = defaultdict(list)

        for log in error_logs:
            # Normalize the error message
            normalized = self._normalize_error_message(log.message)
            groups[normalized].append(log)

        return dict(groups)

    def _normalize_error_message(self, message: str) -> str:
        """Normalize error messages for pattern matching."""
        # Remove timestamps, IDs, and other variable parts
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z?', '[TIMESTAMP]', message)
        normalized = re.sub(
            r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
            '[UUID]',
            normalized)
        normalized = re.sub(r'\b\d+\b', '[NUMBER]', normalized)
        normalized = re.sub(r'\b0x[0-9a-fA-F]+\b', '[HEX]', normalized)
        normalized = re.sub(r'[\'"][^\'"]*[\'"]', '[STRING]', normalized)

        return normalized.strip()

    def _detect_error_spikes(self, sorted_logs: List[LogEntry]) -> List[DetectedPattern]:
        """Detect spikes in error frequency."""
        patterns = []
        window_size = timedelta(minutes=5)

        error_logs = [log for log in sorted_logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]

        if len(error_logs) < 5:
            return patterns

        # Calculate error rates in time windows
        windows = []
        current_start = error_logs[0].timestamp
        end_time = error_logs[-1].timestamp

        while current_start <= end_time:
            window_end = current_start + window_size
            window_errors = [
                log for log in error_logs
                if current_start <= log.timestamp < window_end
            ]
            windows.append((current_start, len(window_errors)))
            current_start = window_end

        if len(windows) < 3:
            return patterns

        # Calculate average and detect spikes
        error_counts = [count for _, count in windows]
        avg_errors = sum(error_counts) / len(error_counts)
        threshold = avg_errors * 3  # 3x average is considered a spike

        for start_time, error_count in windows:
            if error_count > threshold and error_count >= 5:
                patterns.append(DetectedPattern(
                    type=PatternType.ERROR_PATTERN,
                    pattern=f"Error spike: {error_count} errors in 5 minutes (avg: {avg_errors:.1f})",
                    confidence=0.85,
                    occurrence_count=error_count,
                    first_seen=start_time,
                    last_seen=start_time + window_size,
                    affected_pods=[],
                    sample_messages=[],
                    severity=SeverityLevel.HIGH,
                ))

        return patterns

    def _detect_periodic_patterns(self, sorted_logs: List[LogEntry]) -> List[DetectedPattern]:
        """Detect periodic patterns in logs."""
        # This is a simplified implementation
        # In a real system, you'd use FFT or other signal processing techniques
        patterns = []

        # For now, just detect if errors happen at regular intervals
        error_logs = [log for log in sorted_logs if log.level == LogLevel.ERROR]

        if len(error_logs) < 5:
            return patterns

        # Calculate intervals between errors
        intervals = []
        for i in range(1, len(error_logs)):
            interval = (error_logs[i].timestamp - error_logs[i - 1].timestamp).total_seconds()
            intervals.append(interval)

        # Check if intervals are similar (indicating periodicity)
        if len(intervals) >= 4:
            avg_interval = sum(intervals) / len(intervals)
            # Check if most intervals are within 20% of average
            similar_count = sum(1 for interval in intervals if abs(interval - avg_interval) / avg_interval < 0.2)

            if similar_count / len(intervals) > 0.7:  # 70% similarity threshold
                patterns.append(DetectedPattern(
                    type=PatternType.PERFORMANCE_ISSUE,
                    pattern=f"Periodic errors: every ~{avg_interval:.0f} seconds",
                    confidence=0.75,
                    occurrence_count=len(error_logs),
                    first_seen=error_logs[0].timestamp,
                    last_seen=error_logs[-1].timestamp,
                    affected_pods=list(set(log.pod_name for log in error_logs)),
                    sample_messages=[log.message for log in error_logs[:3]],
                    severity=SeverityLevel.MEDIUM,
                ))

        return patterns

    def _detect_cascade_sequences(self, pod_logs: List[LogEntry]) -> List[DetectedPattern]:
        """Detect cascading failure sequences within a pod."""
        patterns = []

        if len(pod_logs) < 2:  # Need at least 2 logs to detect escalation
            return patterns

        # Look for sequences where errors escalate in severity
        sequence_start = None
        current_severity = SeverityLevel.LOW
        sequence_logs = []

        for log in pod_logs:
            log_severity = SeverityDetectionAlgorithm.detect_combined_severity(log)
            severity_values = {
                SeverityLevel.LOW: 1,
                SeverityLevel.MEDIUM: 2,
                SeverityLevel.HIGH: 3,
                SeverityLevel.CRITICAL: 4,
            }

            if severity_values[log_severity] > severity_values[current_severity]:
                if sequence_start is None:
                    sequence_start = log.timestamp
                    sequence_logs = [log]
                else:
                    sequence_logs.append(log)
                current_severity = log_severity
            else:
                # Check if we have a cascade pattern before resetting
                if (len(sequence_logs) >= 2 and  # Reduced threshold to 2 for cascade detection
                    sequence_start and
                        current_severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]):

                    patterns.append(DetectedPattern(
                        type=PatternType.PERFORMANCE_ISSUE,
                        pattern=f"Cascading failure in pod {pod_logs[0].pod_name}",
                        confidence=0.8,
                        occurrence_count=len(sequence_logs),
                        first_seen=sequence_start,
                        last_seen=sequence_logs[-1].timestamp,
                        affected_pods=[pod_logs[0].pod_name],
                        sample_messages=[log.message for log in sequence_logs[:3]],
                        severity=current_severity,
                    ))

                # Reset sequence tracking
                sequence_start = None
                current_severity = SeverityLevel.LOW
                sequence_logs = []

        # Check for cascade at end of logs too
        if (len(sequence_logs) >= 2 and  # Reduced threshold
            sequence_start and
                current_severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]):

            patterns.append(DetectedPattern(
                type=PatternType.PERFORMANCE_ISSUE,
                pattern=f"Cascading failure in pod {pod_logs[0].pod_name}",
                confidence=0.8,
                occurrence_count=len(sequence_logs),
                first_seen=sequence_start,
                last_seen=sequence_logs[-1].timestamp,
                affected_pods=[pod_logs[0].pod_name],
                sample_messages=[log.message for log in sequence_logs[:3]],
                severity=current_severity,
            ))

        return patterns

    def _detect_volume_anomalies(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect anomalous log volume patterns."""
        patterns = []

        if len(log_entries) < 20:
            return patterns

        # Group logs by time windows
        window_size = timedelta(minutes=10)
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

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

        if len(windows) < 5:
            return patterns

        # Detect volume anomalies
        counts = [count for _, count in windows]
        avg_count = sum(counts) / len(counts)
        threshold_high = avg_count * 5  # 5x average
        threshold_low = avg_count * 0.2  # 20% of average

        for start_time, count in windows:
            if count > threshold_high:
                patterns.append(DetectedPattern(
                    type=PatternType.PERFORMANCE_ISSUE,
                    pattern=f"High log volume: {count} logs in 10 minutes (avg: {avg_count:.1f})",
                    confidence=0.7,
                    occurrence_count=count,
                    first_seen=start_time,
                    last_seen=start_time + window_size,
                    affected_pods=[],
                    sample_messages=[],
                    severity=SeverityLevel.MEDIUM,
                ))
            elif count < threshold_low and avg_count > 5:
                patterns.append(DetectedPattern(
                    type=PatternType.PERFORMANCE_ISSUE,
                    pattern=f"Low log volume: {count} logs in 10 minutes (avg: {avg_count:.1f})",
                    confidence=0.6,
                    occurrence_count=count,
                    first_seen=start_time,
                    last_seen=start_time + window_size,
                    affected_pods=[],
                    sample_messages=[],
                    severity=SeverityLevel.LOW,
                ))

        return patterns

    def _detect_pod_anomalies(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect anomalous behavior from specific pods."""
        patterns = []

        # Group by pod
        pod_logs = defaultdict(list)
        for log in log_entries:
            pod_logs[log.pod_name].append(log)

        if len(pod_logs) < 2:
            return patterns

        # Calculate average log volume per pod
        pod_counts = {pod: len(logs) for pod, logs in pod_logs.items()}
        avg_logs_per_pod = sum(pod_counts.values()) / len(pod_counts)

        # Detect pods with anomalous log volumes
        for pod_name, log_count in pod_counts.items():
            if log_count > avg_logs_per_pod * 3:  # 3x average
                severity = SeverityLevel.MEDIUM
                # Check if the high volume is due to errors
                error_count = sum(1 for log in pod_logs[pod_name]
                                  if log.level in [LogLevel.ERROR, LogLevel.CRITICAL])
                if error_count / log_count > 0.3:  # More than 30% errors
                    severity = SeverityLevel.HIGH

                patterns.append(DetectedPattern(
                    type=PatternType.PERFORMANCE_ISSUE,
                    pattern=f"Pod {pod_name} excessive logs: {log_count} (avg: {avg_logs_per_pod:.1f})",
                    confidence=0.75,
                    occurrence_count=log_count,
                    first_seen=min(log.timestamp for log in pod_logs[pod_name]),
                    last_seen=max(log.timestamp for log in pod_logs[pod_name]),
                    affected_pods=[pod_name],
                    sample_messages=[log.message for log in pod_logs[pod_name][:3]],
                    severity=severity,
                ))

        return patterns

    def detect_volume_anomalies(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect volume anomalies in log entries."""
        patterns = []

        if not log_entries:
            return patterns

        # Sort logs by timestamp
        sorted_logs = sorted(log_entries, key=lambda x: x.timestamp)

        # Calculate log volume per minute
        log_counts = defaultdict(int)
        for log in sorted_logs:
            minute_key = log.timestamp.replace(second=0, microsecond=0)
            log_counts[minute_key] += 1

        if len(log_counts) < 3:  # Need at least 3 minutes of data
            return patterns

        volumes = list(log_counts.values())
        mean_volume = statistics.mean(volumes)
        std_volume = statistics.stdev(volumes) if len(volumes) > 1 else 0

        # Detect volume spikes (> mean + 2 * std)
        threshold = mean_volume + (2 * std_volume)

        for timestamp, count in log_counts.items():
            if count > threshold and count >= 20:  # At least 20 logs and above threshold
                patterns.append(DetectedPattern(
                    type=PatternType.PERFORMANCE_ISSUE,
                    pattern=f"Volume spike: {count} logs/min (avg: {mean_volume:.1f})",
                    confidence=0.8,
                    occurrence_count=count,
                    first_seen=timestamp,
                    last_seen=timestamp + timedelta(minutes=1),
                    affected_pods=[],
                    sample_messages=[],
                    severity=SeverityLevel.MEDIUM if count < threshold * 2 else SeverityLevel.HIGH,
                ))

        return patterns


class BatchProcessor:
    """Efficient batch processing for large log volumes."""

    def __init__(self, batch_size: int = 1000, max_concurrent_batches: int = 5):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_logs_in_batches(
        self,
        log_entries: List[LogEntry],
        processor_func,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process logs in efficient batches."""
        if not log_entries:
            return []

        logger.info(f"Processing {len(log_entries)} logs in batches of {self.batch_size}")

        batches = [
            log_entries[i:i + self.batch_size]
            for i in range(0, len(log_entries), self.batch_size)
        ]

        # Process batches concurrently
        tasks = [
            self._process_batch(batch, processor_func, *args, **kwargs)
            for batch in batches
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            all_results.extend(result)

        logger.info(f"Batch processing complete: {len(all_results)} results")
        return all_results

    async def _process_batch(self, batch: List[LogEntry], processor_func, *args, **kwargs):
        """Process a single batch with concurrency control."""
        async with self.semaphore:
            try:
                if inspect.iscoroutinefunction(processor_func):
                    return await processor_func(batch, *args, **kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, processor_func, batch, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error processing batch of {len(batch)} logs: {e}")
                return []


class LogAnalysisEngine:
    """Main log analysis engine combining all analysis capabilities."""

    def __init__(self,
                 gemini_config: Optional[GeminiConfig] = None,
                 highlighter_config: Optional[HighlighterConfig] = None,
                 summarizer_config: Optional[SummarizerConfig] = None):
        self.severity_detector = SeverityDetectionAlgorithm()
        self.pattern_engine = PatternRecognitionEngine()
        self.batch_processor = BatchProcessor()
        self.gemini_client = GeminiClient(gemini_config) if gemini_config else None
        self.highlighter = SeverityHighlighter(highlighter_config)
        self.summarizer = LogSummarizer(self.gemini_client, summarizer_config)
        self.logger = get_logger(__name__)

    async def analyze_logs_comprehensive(
        self,
        log_entries: List[LogEntry],
        use_ai: bool = True,
        analysis_type: str = "comprehensive"
    ) -> AIAnalysisResult:
        """Perform comprehensive log analysis combining multiple techniques."""
        if not log_entries:
            raise LogProcessingError("No log entries provided for analysis")

        start_time = utc_now()
        self.logger.info(f"Starting comprehensive analysis of {len(log_entries)} log entries")

        # Basic statistics
        time_window_start = min(log.timestamp for log in log_entries)
        time_window_end = max(log.timestamp for log in log_entries)

        # Severity analysis
        severity_scores = await self._analyze_severity_distribution(log_entries)
        overall_severity = self._calculate_overall_severity(severity_scores)

        # Pattern detection
        patterns = await self._detect_all_patterns(log_entries)

        # Error analysis
        error_rate, warning_rate = self._calculate_error_rates(log_entries)
        top_errors = self._extract_top_error_messages(log_entries)

        # AI analysis (if available and requested)
        ai_insights = []
        recommendations = []
        confidence_score = 0.7  # Base confidence from rule-based analysis

        if use_ai and self.gemini_client:
            try:
                ai_result = await self.gemini_client.analyze_logs(log_entries, analysis_type)
                ai_insights = [ai_result.summary] if hasattr(ai_result, 'summary') else []
                recommendations = ai_result.recommendations
                confidence_score = max(confidence_score, ai_result.confidence_score)
                # Use AI severity if higher confidence
                if ai_result.confidence_score > 0.8:
                    overall_severity = ai_result.overall_severity
            except Exception as e:
                self.logger.warning(f"AI analysis failed, using rule-based analysis only: {e}")
                # Propagate error details to result metadata
                ai_insights = [f"AI Analysis Failed: {str(e)}"]
                recommendations.append("AI analysis unavailable - check configuration and logs")


        # Generate recommendations from patterns
        pattern_recommendations = self._generate_pattern_recommendations(patterns)
        recommendations.extend(pattern_recommendations)

        # Calculate analysis duration
        analysis_duration = (utc_now() - start_time).total_seconds()

        result = AIAnalysisResult(
            analysis_timestamp=start_time,
            log_entries_analyzed=len(log_entries),
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            overall_severity=overall_severity,
            confidence_score=confidence_score,
            detected_patterns=patterns,
            severity_distribution=severity_scores,
            error_rate=error_rate,
            warning_rate=warning_rate,
            top_error_messages=top_errors,
            anomalies=ai_insights,
            recommendations=list(set(recommendations)),  # Remove duplicates
            tags=[analysis_type, "comprehensive-analysis"],
            metadata={
                "analysis_method": "hybrid_rule_ai" if use_ai and self.gemini_client else "rule_based",
                "patterns_detected": len(patterns),
                "ai_enabled": use_ai and self.gemini_client is not None,
            },
            analysis_duration_seconds=analysis_duration,
        )

        self.logger.info(
            f"Analysis complete: {overall_severity.value} severity, "
            f"{len(patterns)} patterns, {len(recommendations)} recommendations"
        )

        return result

    async def analyze_severity_only(self, log_entries: List[LogEntry]) -> Dict[str, Union[SeverityLevel, float]]:
        """Fast severity-only analysis for real-time processing."""
        if not log_entries:
            return {"overall_severity": SeverityLevel.LOW, "confidence": 0.0}

        severity_scores = await self._analyze_severity_distribution(log_entries)
        overall_severity = self._calculate_overall_severity(severity_scores)
        error_rate, _ = self._calculate_error_rates(log_entries)

        # Adjust confidence based on data quality
        confidence = 0.8 if len(log_entries) > 10 else 0.5

        return {
            "overall_severity": overall_severity,
            "confidence": confidence,
            "error_rate": error_rate,
            "severity_distribution": severity_scores,
        }

    async def detect_patterns_only(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Fast pattern detection for real-time monitoring."""
        return await self._detect_all_patterns(log_entries)

    async def _analyze_severity_distribution(self, log_entries: List[LogEntry]) -> Dict[str, int]:
        """Analyze severity distribution across log entries."""
        severity_counts = defaultdict(int)

        # Use batch processing for large datasets
        if len(log_entries) > 1000:
            batch_results = await self.batch_processor.process_logs_in_batches(
                log_entries, self._analyze_batch_severity
            )
            # Aggregate batch results
            for batch_result in batch_results:
                for severity, count in batch_result.items():
                    severity_counts[severity] += count
        else:
            # Process directly for smaller datasets
            for log in log_entries:
                severity = self.severity_detector.detect_combined_severity(log)
                severity_counts[severity.value] += 1

        return dict(severity_counts)

    def _analyze_batch_severity(self, batch: List[LogEntry]) -> Dict[str, int]:
        """Analyze severity for a batch of logs (sync function for thread pool)."""
        severity_counts = defaultdict(int)
        for log in batch:
            severity = self.severity_detector.detect_combined_severity(log)
            severity_counts[severity.value] += 1
        return dict(severity_counts)

    async def _detect_all_patterns(self, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Detect all types of patterns in log entries."""
        all_patterns = []

        # Run different pattern detection algorithms concurrently
        pattern_tasks = [
            self._detect_patterns_async(self.pattern_engine.detect_error_patterns, log_entries),
            self._detect_patterns_async(self.pattern_engine.detect_temporal_patterns, log_entries),
            self._detect_patterns_async(self.pattern_engine.detect_cascade_patterns, log_entries),
            self._detect_patterns_async(self.pattern_engine.detect_anomaly_patterns, log_entries),
        ]

        pattern_results = await asyncio.gather(*pattern_tasks, return_exceptions=True)

        for result in pattern_results:
            if isinstance(result, Exception):
                self.logger.error(f"Pattern detection error: {result}")
                continue
            all_patterns.extend(result)

        # Remove duplicate patterns
        unique_patterns = self._deduplicate_patterns(all_patterns)

        return unique_patterns

    async def _detect_patterns_async(self, detection_func, log_entries: List[LogEntry]) -> List[DetectedPattern]:
        """Run pattern detection in thread pool for CPU-intensive work."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, detection_func, log_entries)

    def _calculate_overall_severity(self, severity_distribution: Dict[str, int]) -> SeverityLevel:
        """Calculate overall severity from distribution."""
        if not severity_distribution:
            return SeverityLevel.LOW

        # If any critical logs exist, return critical
        if severity_distribution.get("critical", 0) > 0:
            return SeverityLevel.CRITICAL

        # If any high severity logs exist, return high
        if severity_distribution.get("high", 0) > 0:
            return SeverityLevel.HIGH

        # If any medium severity logs exist, return medium
        if severity_distribution.get("medium", 0) > 0:
            return SeverityLevel.MEDIUM

        # Otherwise return low
        return SeverityLevel.LOW

    def _calculate_error_rates(self, log_entries: List[LogEntry]) -> Tuple[float, float]:
        """Calculate error and warning rates."""
        total_logs = len(log_entries)
        if total_logs == 0:
            return 0.0, 0.0

        error_count = sum(1 for log in log_entries
                          if log.level in [LogLevel.ERROR, LogLevel.CRITICAL])
        warning_count = sum(1 for log in log_entries
                            if log.level == LogLevel.WARNING)

        error_rate = error_count / total_logs
        warning_rate = warning_count / total_logs

        return error_rate, warning_rate

    def _extract_top_error_messages(self, log_entries: List[LogEntry], limit: int = 5) -> List[str]:
        """Extract most common error messages."""
        error_logs = [log for log in log_entries
                      if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]

        if not error_logs:
            return []

        # Normalize and count error messages
        error_counter = Counter()
        for log in error_logs:
            normalized = self.pattern_engine._normalize_error_message(log.message)
            error_counter[normalized] += 1

        return [message for message, _ in error_counter.most_common(limit)]

    def _generate_pattern_recommendations(self, patterns: List[DetectedPattern]) -> List[str]:
        """Generate recommendations based on detected patterns."""
        recommendations = []

        for pattern in patterns:
            if pattern.type == PatternType.ERROR_PATTERN:
                if pattern.severity == SeverityLevel.CRITICAL:
                    recommendations.append("URGENT: Critical error spike detected - investigate immediately")
                else:
                    recommendations.append("Investigate error spike pattern and check system resources")

            elif pattern.type == PatternType.PERFORMANCE_ISSUE:
                if "cascade" in pattern.pattern:
                    recommendations.append("Cascading failure detected - check dependency chains and circuit breakers")
                elif "periodic" in pattern.pattern:
                    recommendations.append("Periodic errors detected - investigate scheduled jobs or resource limitations")
                elif "volume" in pattern.pattern:
                    recommendations.append("Unusual log volume detected - check for resource issues or configuration changes")
                else:
                    recommendations.append("Performance issue detected - investigate system resources and bottlenecks")

        return recommendations

    def _deduplicate_patterns(self, patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        """Remove duplicate patterns based on type and similarity."""
        if not patterns:
            return patterns

        unique_patterns = []
        seen_patterns = set()

        for pattern in patterns:
            # Create a signature for deduplication
            signature = (pattern.type, pattern.pattern, pattern.severity)

            if signature not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(signature)
            else:
                # If similar pattern exists, merge information
                existing = next(p for p in unique_patterns
                                if (p.type, p.pattern, p.severity) == signature)
                existing.occurrences += pattern.occurrences
                existing.confidence = max(existing.confidence, pattern.confidence)

        return unique_patterns

    async def analyze_with_highlighting(
        self,
        log_entries: List[LogEntry],
        theme: Optional[HighlightTheme] = None
    ) -> Tuple[AIAnalysisResult, List[Text]]:
        """Perform comprehensive analysis and return highlighted log entries.

        Args:
            log_entries: List of log entries to analyze
            theme: Optional highlighting theme to use

        Returns:
            Tuple of (analysis result, highlighted log entries)
        """
        # Update highlighter theme if specified
        if theme and theme != self.highlighter.config.theme:
            new_config = HighlighterConfig(theme=theme)
            self.highlighter.update_config(new_config)

        # Perform analysis
        analysis_result = await self.analyze_logs_comprehensive(log_entries)

        # Generate highlighted logs
        highlighted_logs = self.highlighter.highlight_multiple_entries(log_entries)

        return analysis_result, highlighted_logs

    def get_highlighted_logs(
        self,
        log_entries: List[LogEntry],
        theme: Optional[HighlightTheme] = None
    ) -> List[Text]:
        """Get highlighted version of log entries.

        Args:
            log_entries: List of log entries to highlight
            theme: Optional highlighting theme to use

        Returns:
            List of Rich Text objects with applied highlighting
        """
        # Update highlighter theme if specified
        if theme and theme != self.highlighter.config.theme:
            new_config = HighlighterConfig(theme=theme)
            self.highlighter.update_config(new_config)

        return self.highlighter.highlight_multiple_entries(log_entries)

    def get_severity_stats_with_colors(
        self,
        log_entries: List[LogEntry]
    ) -> Dict[str, Union[int, Text]]:
        """Get severity statistics with color-coded severity labels.

        Args:
            log_entries: List of log entries to analyze

        Returns:
            Dictionary with severity counts and colored labels
        """
        stats = self.highlighter.get_severity_stats(log_entries)
        colored_stats = {}

        for severity, count in stats.items():
            # Get the style for this severity level
            style = self.highlighter._get_style_for_severity(severity)
            rich_style = self.highlighter._convert_to_rich_style(style)

            # Create colored text for the severity label
            colored_label = Text(severity.value.title(), style=rich_style)

            colored_stats[f"{severity.value}_count"] = count
            colored_stats[f"{severity.value}_label"] = colored_label

        return colored_stats

    def highlight_single_log(
        self,
        log_entry: LogEntry,
        theme: Optional[HighlightTheme] = None
    ) -> Text:
        """Highlight a single log entry.

        Args:
            log_entry: Log entry to highlight
            theme: Optional highlighting theme to use

        Returns:
            Rich Text object with applied highlighting
        """
        # Update highlighter theme if specified
        if theme and theme != self.highlighter.config.theme:
            new_config = HighlighterConfig(theme=theme)
            self.highlighter.update_config(new_config)

        return self.highlighter.highlight_log_entry(log_entry)

    def update_highlighting_config(self, config: HighlighterConfig):
        """Update the highlighting configuration.

        Args:
            config: New highlighter configuration
        """
        self.highlighter.update_config(config)

    def get_available_themes(self) -> List[HighlightTheme]:
        """Get list of available highlighting themes.

        Returns:
            List of available highlight themes
        """
        return list(HighlightTheme)

    def set_highlighting_theme(self, theme: HighlightTheme):
        """Set the highlighting theme.

        Args:
            theme: Theme to set
        """
        new_config = HighlighterConfig(theme=theme)
        self.highlighter.update_config(new_config)

    # Advanced Pattern Detection Methods

    def detect_recurring_issue_patterns(self, log_entries: List[LogEntry]) -> 'PatternDetectionResult':
        """Detect recurring issue patterns using advanced pattern detection.

        Args:
            log_entries: List of log entries to analyze for patterns

        Returns:
            Comprehensive pattern detection results
        """
        from .patterns import AdvancedPatternDetector

        detector = AdvancedPatternDetector()
        return detector.detect_all_patterns(log_entries)

    def analyze_pattern_trends(self, log_entries: List[LogEntry]) -> Dict[str, Any]:
        """Analyze pattern trends and evolution over time.

        Args:
            log_entries: List of log entries to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        from .patterns import AdvancedPatternDetector

        detector = AdvancedPatternDetector()
        patterns = detector.detect_all_patterns(log_entries)

        # Analyze trends in patterns
        trend_analysis = {
            "total_patterns_detected": (
                len(patterns.recurring_issues) +
                len(patterns.temporal_patterns) +
                len(patterns.cascade_patterns) +
                len(patterns.anomaly_patterns)
            ),
            "pattern_complexity_score": patterns.overall_pattern_score,
            "health_trend": patterns.health_trends.get("overall", "unknown"),
            "high_impact_issues": [
                p for p in patterns.recurring_issues
                if p.impact_score > 70
            ],
            "critical_anomalies": [
                a for a in patterns.anomaly_patterns
                if a.deviation_score > 3.0
            ],
            "cascade_risk": len(patterns.cascade_patterns) > 0,
            "prediction_confidence": self._calculate_prediction_confidence(patterns)
        }

        return trend_analysis

    def get_pattern_recommendations(self, log_entries: List[LogEntry]) -> List[str]:
        """Get actionable recommendations based on detected patterns.

        Args:
            log_entries: List of log entries to analyze

        Returns:
            List of actionable recommendations
        """
        from .patterns import AdvancedPatternDetector

        detector = AdvancedPatternDetector()
        patterns = detector.detect_all_patterns(log_entries)

        return patterns.recommendations

    def analyze_pattern_correlations(self, log_entries: List[LogEntry]) -> Dict[str, Any]:
        """Analyze correlations between different pattern types.

        Args:
            log_entries: List of log entries to analyze

        Returns:
            Pattern correlation analysis
        """
        from .patterns import AdvancedPatternDetector

        detector = AdvancedPatternDetector()
        patterns = detector.detect_all_patterns(log_entries)

        correlations = {
            "recurring_temporal_correlation": self._analyze_recurring_temporal_correlation(
                patterns.recurring_issues, patterns.temporal_patterns
            ),
            "cascade_triggers": self._identify_cascade_triggers(patterns.cascade_patterns),
            "anomaly_pattern_overlap": self._analyze_anomaly_overlap(
                patterns.anomaly_patterns, patterns.recurring_issues
            ),
            "severity_pattern_mapping": self._map_severity_to_patterns(patterns)
        }

        return correlations

    def predict_future_issues(self, log_entries: List[LogEntry]) -> Dict[str, Any]:
        """Predict potential future issues based on current patterns.

        Args:
            log_entries: List of log entries to analyze for predictions

        Returns:
            Predictions about potential future issues
        """
        from .patterns import AdvancedPatternDetector

        detector = AdvancedPatternDetector()
        patterns = detector.detect_all_patterns(log_entries)

        predictions = {
            "likely_recurring_issues": [
                {
                    "pattern": p.normalized_error,
                    "predicted_frequency": p.frequency_per_hour,
                    "confidence": 0.8 if p.trend == "increasing" else 0.6,
                    "risk_level": "high" if p.impact_score > 70 else "medium"
                }
                for p in patterns.recurring_issues
                if p.trend in ["increasing", "stable"]
            ],
            "cascade_risk_assessment": {
                "risk_level": "high" if len(patterns.cascade_patterns) > 2 else "low",
                "potential_triggers": [p.trigger_event for p in patterns.cascade_patterns],
                "containment_success_rate": sum(
                    1 for p in patterns.cascade_patterns if p.containment_success
                ) / max(len(patterns.cascade_patterns), 1)
            },
            "volume_spike_predictions": [
                {
                    "type": p.pattern_type,
                    "confidence": p.prediction_confidence,
                    "next_occurrence": p.peak_times
                }
                for p in patterns.temporal_patterns
                if p.prediction_confidence > 0.7
            ]
        }

        return predictions

    def _calculate_prediction_confidence(self, patterns: 'PatternDetectionResult') -> float:
        """Calculate overall prediction confidence based on pattern quality."""
        if not any([patterns.recurring_issues, patterns.temporal_patterns, patterns.cascade_patterns]):
            return 0.0

        total_confidence = 0.0
        confidence_count = 0

        # Recurring issue confidence
        for pattern in patterns.recurring_issues:
            if pattern.trend in ["increasing", "decreasing"]:
                total_confidence += 0.8
            else:
                total_confidence += 0.6
            confidence_count += 1

        # Temporal pattern confidence
        for pattern in patterns.temporal_patterns:
            total_confidence += pattern.prediction_confidence
            confidence_count += 1

        # Cascade pattern confidence (lower due to complexity)
        for pattern in patterns.cascade_patterns:
            total_confidence += 0.5 if pattern.containment_success else 0.3
            confidence_count += 1

        return total_confidence / max(confidence_count, 1)

    def _analyze_recurring_temporal_correlation(self, recurring: List, temporal: List) -> Dict[str, Any]:
        """Analyze correlation between recurring issues and temporal patterns."""
        if not recurring or not temporal:
            return {"correlation": "none", "details": "Insufficient data"}

        # Simple correlation analysis
        periodic_temporal = [p for p in temporal if p.pattern_type == "periodic_errors"]

        if periodic_temporal and recurring:
            return {
                "correlation": "possible",
                "periodic_patterns": len(periodic_temporal),
                "recurring_issues": len(recurring),
                "details": "Both periodic and recurring patterns detected"
            }

        return {"correlation": "weak", "details": "No clear correlation detected"}

    def _identify_cascade_triggers(self, cascades: List) -> List[str]:
        """Identify common cascade trigger events."""
        if not cascades:
            return []

        triggers = [cascade.trigger_event for cascade in cascades]
        return list(set(triggers))

    def _analyze_anomaly_overlap(self, anomalies: List, recurring: List) -> Dict[str, Any]:
        """Analyze overlap between anomalies and recurring patterns."""
        if not anomalies or not recurring:
            return {"overlap": "none"}

        # Check for timing overlaps or similar components
        overlap_count = 0
        for anomaly in anomalies:
            for recur in recurring:
                if any(pod in recur.affected_pods for pod in anomaly.affected_components):
                    overlap_count += 1
                    break

        return {
            "overlap": "significant" if overlap_count > len(anomalies) / 2 else "minimal",
            "overlap_count": overlap_count,
            "total_anomalies": len(anomalies)
        }

    def _map_severity_to_patterns(self, patterns: 'PatternDetectionResult') -> Dict[str, int]:
        """Map severity levels to pattern types."""
        severity_mapping = {
            "high_severity_recurring": len([
                p for p in patterns.recurring_issues
                if p.impact_score > 70
            ]),
            "critical_anomalies": len([
                a for a in patterns.anomaly_patterns
                if a.deviation_score > 3.0
            ]),
            "cascade_failures": len(patterns.cascade_patterns),
            "temporal_issues": len([
                p for p in patterns.temporal_patterns
                if p.pattern_type in ["volume_spike", "burst_pattern"]
            ])
        }

        return severity_mapping

    # Smart Summary Integration Methods

    async def generate_smart_summary(
        self,
        log_entries: List[LogEntry],
        config: Optional[SummarizerConfig] = None
    ) -> LogSummaryReport:
        """Generate comprehensive smart summary with configurable time windows.

        Args:
            log_entries: List of log entries to summarize
            config: Optional summarizer configuration override

        Returns:
            Complete log summary report with time-window analysis
        """
        effective_config = config or self.summarizer.config
        self.logger.info(
            f"Generating smart summary for {len(log_entries)} log entries "
            f"with {effective_config.window_size.value} windows"
        )

        return await self.summarizer.summarize_logs(log_entries, effective_config)

    async def generate_executive_summary(
        self,
        log_entries: List[LogEntry],
        window_size: TimeWindowSize = TimeWindowSize.ONE_HOUR
    ) -> str:
        """Generate executive-level summary for management reporting.

        Args:
            log_entries: List of log entries to summarize
            window_size: Time window size for analysis

        Returns:
            Executive summary text suitable for management
        """
        config = SummarizerConfig(
            window_size=window_size,
            summary_type=SummaryType.EXECUTIVE,
            max_summary_length=800,
            enable_trend_analysis=True
        )

        summary_report = await self.summarizer.summarize_logs(log_entries, config)
        return summary_report.executive_summary

    async def generate_technical_summary(
        self,
        log_entries: List[LogEntry],
        window_size: TimeWindowSize = TimeWindowSize.FIFTEEN_MINUTES
    ) -> Dict[str, Any]:
        """Generate detailed technical summary for engineers.

        Args:
            log_entries: List of log entries to summarize
            window_size: Time window size for analysis

        Returns:
            Dictionary containing technical analysis details
        """
        config = SummarizerConfig(
            window_size=window_size,
            summary_type=SummaryType.TECHNICAL,
            max_insights=15,
            min_confidence=0.5,
            enable_trend_analysis=True
        )

        summary_report = await self.summarizer.summarize_logs(log_entries, config)

        return {
            "time_range": {
                "start": summary_report.time_range_start,
                "end": summary_report.time_range_end,
                "duration_minutes": (
                    summary_report.time_range_end - summary_report.time_range_start
                ).total_seconds() / 60
            },
            "overview": {
                "total_logs": summary_report.total_log_entries,
                "time_windows": len(summary_report.window_summaries),
                "window_size": window_size.value,
                "key_insights_count": len(summary_report.key_insights),
                "trends_detected": len(summary_report.trend_analyses)
            },
            "window_summaries": [
                {
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                    "log_count": window.log_count,
                    "error_count": window.error_count,
                    "warning_count": window.warning_count,
                    "severity": window.overall_severity.value,
                    "summary": window.summary_text,
                    "key_events": window.key_events,
                    "top_errors": window.top_errors
                }
                for window in summary_report.window_summaries
            ],
            "key_insights": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "severity": insight.severity.value,
                    "confidence": insight.confidence,
                    "recommendation": insight.recommendation,
                    "affected_windows_count": len(insight.affected_windows)
                }
                for insight in summary_report.key_insights
            ],
            "trend_analyses": [
                {
                    "metric": trend.metric_name,
                    "direction": trend.direction.value,
                    "confidence": trend.confidence,
                    "change_percentage": trend.change_percentage,
                    "recommendation": trend.recommendation
                }
                for trend in summary_report.trend_analyses
            ],
            "executive_summary": summary_report.executive_summary,
            "recommendations": summary_report.recommendations,
            "generated_at": summary_report.generated_at
        }

    async def generate_operational_summary(
        self,
        log_entries: List[LogEntry],
        window_size: TimeWindowSize = TimeWindowSize.THIRTY_MINUTES
    ) -> Dict[str, Any]:
        """Generate operational summary focusing on system health.

        Args:
            log_entries: List of log entries to summarize
            window_size: Time window size for analysis

        Returns:
            Dictionary containing operational health metrics
        """
        config = SummarizerConfig(
            window_size=window_size,
            summary_type=SummaryType.OPERATIONAL,
            max_insights=10,
            min_confidence=0.6,
            enable_trend_analysis=True
        )

        summary_report = await self.summarizer.summarize_logs(log_entries, config)

        # Calculate operational health metrics
        total_windows = len(summary_report.window_summaries)
        error_windows = sum(1 for w in summary_report.window_summaries if w.error_count > 0)
        warning_windows = sum(1 for w in summary_report.window_summaries if w.warning_count > 0)
        healthy_windows = total_windows - error_windows - warning_windows

        health_score = (healthy_windows / max(total_windows, 1)) * 100

        critical_insights = [
            insight for insight in summary_report.key_insights
            if insight.severity == SeverityLevel.CRITICAL
        ]
        high_insights = [
            insight for insight in summary_report.key_insights
            if insight.severity == SeverityLevel.HIGH
        ]

        return {
            "health_overview": {
                "overall_health_score": round(health_score, 1),
                "status": (
                    "healthy" if health_score > 80 else
                    "warning" if health_score > 60 else
                    "critical"
                ),
                "total_time_windows": total_windows,
                "healthy_windows": healthy_windows,
                "warning_windows": warning_windows,
                "error_windows": error_windows
            },
            "critical_issues": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "recommendation": insight.recommendation
                }
                for insight in critical_insights
            ],
            "high_priority_issues": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence
                }
                for insight in high_insights
            ],
            "trending_issues": [
                {
                    "metric": trend.metric_name,
                    "direction": trend.direction.value,
                    "change": trend.change_percentage,
                    "recommendation": trend.recommendation
                }
                for trend in summary_report.trend_analyses
                if trend.direction.value in ["increasing", "decreasing"]
            ],
            "immediate_actions": [
                rec for rec in summary_report.recommendations
                if any(keyword in rec.lower() for keyword in ["immediate", "urgent", "critical"])
            ],
            "summary": summary_report.executive_summary,
            "analysis_period": {
                "start": summary_report.time_range_start,
                "end": summary_report.time_range_end,
                "window_size": window_size.value
            }
        }

    async def generate_custom_summary(
        self,
        log_entries: List[LogEntry],
        window_size: TimeWindowSize,
        summary_type: SummaryType,
        max_insights: int = 10,
        min_confidence: float = 0.6,
        enable_ai: bool = True
    ) -> LogSummaryReport:
        """Generate custom summary with specific configuration.

        Args:
            log_entries: List of log entries to summarize
            window_size: Size of time windows for analysis
            summary_type: Type of summary to generate
            max_insights: Maximum number of insights to include
            min_confidence: Minimum confidence threshold for insights
            enable_ai: Whether to enable AI-powered summarization

        Returns:
            Complete summary report with custom configuration
        """
        config = SummarizerConfig(
            window_size=window_size,
            summary_type=summary_type,
            max_insights=max_insights,
            min_confidence=min_confidence,
            enable_ai_summarization=enable_ai,
            enable_trend_analysis=True
        )

        self.logger.info(
            f"Generating custom summary: {summary_type.value} style, "
            f"{window_size.value} windows, max {max_insights} insights"
        )

        return await self.summarizer.summarize_logs(log_entries, config)

    def get_available_window_sizes(self) -> List[TimeWindowSize]:
        """Get list of available time window sizes for summarization.

        Returns:
            List of supported time window sizes
        """
        return list(TimeWindowSize)

    def get_available_summary_types(self) -> List[SummaryType]:
        """Get list of available summary types.

        Returns:
            List of supported summary types
        """
        return list(SummaryType)

    async def analyze_with_smart_summary(
        self,
        log_entries: List[LogEntry],
        window_size: TimeWindowSize = TimeWindowSize.FIFTEEN_MINUTES,
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis with smart summarization.

        Args:
            log_entries: List of log entries to analyze
            window_size: Time window size for summarization
            include_analysis: Whether to include full analysis results

        Returns:
            Dictionary containing both analysis and summary results
        """
        results = {"timestamp": utc_now()}

        # Generate smart summary
        summary_config = SummarizerConfig(
            window_size=window_size,
            summary_type=SummaryType.TECHNICAL,
            enable_trend_analysis=True
        )

        summary_report = await self.summarizer.summarize_logs(log_entries, summary_config)
        results["smart_summary"] = {
            "executive_summary": summary_report.executive_summary,
            "time_windows": len(summary_report.window_summaries),
            "key_insights_count": len(summary_report.key_insights),
            "recommendations": summary_report.recommendations,
            "trend_analyses": len(summary_report.trend_analyses)
        }

        # Include full analysis if requested
        if include_analysis:
            analysis_result = await self.analyze_logs_comprehensive(log_entries)
            results["analysis"] = {
                "overall_severity": analysis_result.overall_severity.value,
                "confidence_score": analysis_result.confidence_score,
                "patterns_detected": len(analysis_result.detected_patterns),
                "error_rate": analysis_result.error_rate,
                "warning_rate": analysis_result.warning_rate,
                "top_errors": analysis_result.top_error_messages,
                "analysis_duration": analysis_result.analysis_duration_seconds
            }

        return results

    def update_summarizer_config(self, config: SummarizerConfig):
        """Update the summarizer configuration.

        Args:
            config: New summarizer configuration
        """
        self.summarizer.config = config
        self.logger.info(
            f"Updated summarizer config: {config.window_size.value} windows, "
            f"{config.summary_type.value} style"
        )

    # Custom Query Methods

    async def query_logs_natural_language(
        self,
        log_entries: List[LogEntry],
        request: 'QueryRequest'
    ) -> 'QueryResponse':
        """Process a natural language query against log entries.

        Args:
            log_entries: List of log entries to query against
            request: QueryRequest containing the natural language question

        Returns:
            QueryResponse with AI-generated answer and metadata
        """
        from ..core.models import QueryResponse

        if not self.gemini_client:
            raise ValueError("Gemini client not available for natural language queries")

        start_time = utc_now()
        self.logger.info(
            f"Processing natural language query: '{request.question}' "
            f"against {len(log_entries)} log entries"
        )

        # Apply filters if specified
        filtered_logs = self._apply_query_filters(log_entries, request.context_filters)

        # Limit log entries to prevent token overflow
        analysis_logs = filtered_logs[:request.max_log_entries]

        try:
            # Get AI answer
            answer = await self.gemini_client.query_logs(analysis_logs, request.question)

            # Calculate confidence score based on various factors
            confidence_score = self._calculate_query_confidence(
                answer, analysis_logs, request
            )

            # Find related patterns if enabled
            related_patterns = []
            if request.enable_pattern_matching and len(analysis_logs) > 0:
                patterns = await self._detect_all_patterns(analysis_logs)
                related_patterns = [
                    f"{p.type.value}: {p.pattern}" for p in patterns[:5]
                ]

            # Generate follow-up suggestions
            followups = self._generate_followup_questions(
                request.question, answer, request.query_type
            ) if request.include_context else []

            # Calculate processing time
            duration = (utc_now() - start_time).total_seconds()

            response = QueryResponse(
                request_id=request.id,
                answer=answer,
                confidence_score=confidence_score,
                sources_analyzed=len(analysis_logs),
                query_duration_seconds=duration,
                related_patterns=related_patterns,
                suggested_followups=followups,
                metadata={
                    "query_type": request.query_type.value,
                    "total_available_logs": len(filtered_logs),
                    "filters_applied": bool(request.context_filters),
                    "pattern_matching_enabled": request.enable_pattern_matching,
                    "ai_model": "gemini",
                }
            )

            self.logger.info(
                f"Query processed: {len(analysis_logs)} logs, "
                f"confidence {confidence_score:.2f}, {duration:.2f}s"
            )

            return response

        except Exception as e:
            self.logger.error(f"Natural language query failed: {e}")
            # Return error response
            duration = (utc_now() - start_time).total_seconds()
            return QueryResponse(
                request_id=request.id,
                answer=f"Sorry, I couldn't process your query: {str(e)}",
                confidence_score=0.0,
                sources_analyzed=len(analysis_logs),
                query_duration_seconds=duration,
                metadata={"error": str(e), "query_type": request.query_type.value}
            )

    async def analyze_with_query(
        self,
        log_entries: List[LogEntry],
        question: str,
        include_analysis: bool = True,
        query_type: 'QueryType' = None
    ) -> Dict[str, Any]:
        """Perform analysis combined with a natural language query.

        Args:
            log_entries: List of log entries to analyze and query
            question: Natural language question
            include_analysis: Whether to include comprehensive analysis
            query_type: Type of query being performed

        Returns:
            Combined analysis and query results
        """
        from ..core.models import QueryRequest, QueryType

        if query_type is None:
            query_type = QueryType.ANALYSIS

        self.logger.info(
            f"Performing combined analysis and query on {len(log_entries)} logs"
        )

        results = {}

        # Perform comprehensive analysis if requested
        if include_analysis:
            analysis_result = await self.analyze_logs_comprehensive(log_entries)
            results["analysis"] = {
                "overall_severity": analysis_result.overall_severity.value,
                "confidence_score": analysis_result.confidence_score,
                "patterns_detected": len(analysis_result.detected_patterns),
                "error_rate": analysis_result.error_rate,
                "warning_rate": analysis_result.warning_rate,
                "recommendations": analysis_result.recommendations,
                "analysis_duration": analysis_result.analysis_duration_seconds
            }

        # Process natural language query
        query_request = QueryRequest(
            question=question,
            query_type=query_type,
            max_log_entries=min(len(log_entries), 100),
            enable_pattern_matching=True,
            include_context=True
        )

        query_response = await self.query_logs_natural_language(log_entries, query_request)
        results["query"] = {
            "question": question,
            "answer": query_response.answer,
            "confidence": query_response.confidence_score,
            "sources_analyzed": query_response.sources_analyzed,
            "duration": query_response.query_duration_seconds,
            "related_patterns": query_response.related_patterns,
            "followup_suggestions": query_response.suggested_followups
        }

        # Add correlation insights
        if include_analysis and "analysis" in results:
            results["insights"] = self._correlate_analysis_with_query(
                results["analysis"], results["query"]
            )

        return results

    async def batch_query_logs(
        self,
        log_entries: List[LogEntry],
        questions: List[str],
        query_config: 'QueryConfig' = None
    ) -> Dict[str, 'QueryResponse']:
        """Process multiple natural language queries in batch.

        Args:
            log_entries: List of log entries to query against
            questions: List of natural language questions
            query_config: Optional query configuration

        Returns:
            Dictionary mapping questions to their responses
        """
        from ..core.models import QueryConfig, QueryRequest, QueryType

        if query_config is None:
            from ..core.models import QueryConfig
            query_config = QueryConfig()

        self.logger.info(
            f"Processing {len(questions)} batch queries against {len(log_entries)} logs"
        )

        results = {}

        for i, question in enumerate(questions, 1):
            self.logger.debug(f"Processing batch query {i}/{len(questions)}: {question}")

            request = QueryRequest(
                question=question,
                query_type=QueryType.ANALYSIS,
                max_log_entries=query_config.default_max_logs,
                enable_pattern_matching=query_config.enable_pattern_integration,
                include_context=query_config.enable_followup_suggestions
            )

            try:
                response = await self.query_logs_natural_language(log_entries, request)
                results[question] = response
            except Exception as e:
                self.logger.error(f"Batch query failed for '{question}': {e}")
                results[question] = QueryResponse(
                    request_id=request.id,
                    answer=f"Query failed: {str(e)}",
                    confidence_score=0.0,
                    sources_analyzed=0,
                    query_duration_seconds=0.0,
                    metadata={"error": str(e), "batch_index": i}
                )

        return results

    def validate_query_request(self, request: 'QueryRequest') -> List[str]:
        """Validate a query request and return any issues.

        Args:
            request: QueryRequest to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        if len(request.question.strip()) < 3:
            issues.append("Question must be at least 3 characters long")

        if request.max_log_entries < 1:
            issues.append("max_log_entries must be at least 1")

        if request.max_log_entries > 1000:
            issues.append("max_log_entries cannot exceed 1000")

        # Check if Gemini client is available for AI queries
        if not self.gemini_client:
            issues.append("AI client not available for natural language queries")

        return issues

    def _apply_query_filters(
        self,
        log_entries: List[LogEntry],
        filters: Dict[str, Any]
    ) -> List[LogEntry]:
        """Apply context filters to log entries.

        Args:
            log_entries: Original log entries
            filters: Filter criteria

        Returns:
            Filtered log entries
        """
        if not filters:
            return log_entries

        filtered = log_entries

        # Filter by pod names
        if "pod_names" in filters:
            pod_names = set(filters["pod_names"])
            filtered = [log for log in filtered if log.pod_name in pod_names]

        # Filter by log levels
        if "log_levels" in filters:
            log_levels = set(filters["log_levels"])
            filtered = [log for log in filtered if log.level and log.level in log_levels]

        # Filter by time range
        if "start_time" in filters:
            start_time = filters["start_time"]
            filtered = [log for log in filtered if log.timestamp >= start_time]

        if "end_time" in filters:
            end_time = filters["end_time"]
            filtered = [log for log in filtered if log.timestamp <= end_time]

        # Filter by message content
        if "message_contains" in filters:
            keywords = filters["message_contains"]
            if isinstance(keywords, str):
                keywords = [keywords]
            filtered = [
                log for log in filtered
                if any(keyword.lower() in log.message.lower() for keyword in keywords)
            ]

        return filtered

    def _calculate_query_confidence(
        self,
        answer: str,
        log_entries: List[LogEntry],
        request: 'QueryRequest'
    ) -> float:
        """Calculate confidence score for query response.

        Args:
            answer: AI-generated answer
            log_entries: Log entries analyzed
            request: Original query request

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7

        # Increase confidence based on number of logs analyzed
        log_factor = min(len(log_entries) / 50, 1.0) * 0.1
        base_confidence += log_factor

        # Decrease confidence for very short answers
        if len(answer.split()) < 10:
            base_confidence -= 0.2

        # Increase confidence for structured answers
        if any(marker in answer.lower() for marker in ["1.", "2.", "•", "-", ":"]):
            base_confidence += 0.1

        # Check for uncertainty indicators
        uncertainty_markers = ["maybe", "possibly", "might", "unclear", "not sure", "hard to tell"]
        if any(marker in answer.lower() for marker in uncertainty_markers):
            base_confidence -= 0.2

        # Adjust for specific query types
        if request.query_type in ['QueryType.METRICS', 'QueryType.SEARCH']:
            base_confidence += 0.05
        elif request.query_type == 'QueryType.TROUBLESHOOTING':
            base_confidence -= 0.05

        return max(0.0, min(1.0, base_confidence))

    def _generate_followup_questions(
        self,
        original_question: str,
        answer: str,
        query_type: 'QueryType'
    ) -> List[str]:
        """Generate suggested follow-up questions.

        Args:
            original_question: The original question asked
            answer: The AI-generated answer
            query_type: Type of the original query

        Returns:
            List of suggested follow-up questions
        """
        followups = []

        # Generic follow-ups based on query type
        if 'QueryType.TROUBLESHOOTING' in str(query_type):
            followups.extend([
                "What are the root causes of these issues?",
                "How can I prevent these problems in the future?",
                "Are there any patterns in when these issues occur?"
            ])
        elif 'QueryType.METRICS' in str(query_type):
            followups.extend([
                "What are the performance trends over time?",
                "Which pods are performing best and worst?",
                "Are there any resource bottlenecks?"
            ])
        elif 'QueryType.ANALYSIS' in str(query_type):
            followups.extend([
                "What are the most critical issues to address first?",
                "How do these issues impact overall system health?",
                "What monitoring should be put in place?"
            ])

        # Context-specific follow-ups based on answer content
        if "error" in answer.lower():
            followups.append("What specific error patterns should I watch for?")
        if "performance" in answer.lower():
            followups.append("What's causing the performance degradation?")
        if "pod" in answer.lower():
            followups.append("Which pods are most affected?")

        return followups[:3]  # Limit to 3 suggestions

    def _correlate_analysis_with_query(
        self,
        analysis_results: Dict[str, Any],
        query_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate comprehensive analysis with query results.

        Args:
            analysis_results: Results from comprehensive analysis
            query_results: Results from natural language query

        Returns:
            Correlation insights
        """
        insights = {
            "correlation_score": 0.0,
            "analysis_query_alignment": "unknown",
            "confidence_correlation": 0.0,
            "insights": []
        }

        # Calculate correlation between analysis severity and query confidence
        analysis_confidence = analysis_results.get("confidence_score", 0.0)
        query_confidence = query_results.get("confidence", 0.0)
        insights["confidence_correlation"] = abs(analysis_confidence - query_confidence)

        # Check alignment between analysis findings and query answer
        analysis_severity = analysis_results.get("overall_severity", "unknown")
        query_answer = query_results.get("answer", "").lower()

        if analysis_severity in ["high", "critical"] and any(
            term in query_answer for term in ["serious", "critical", "urgent", "error", "problem"]
        ):
            insights["analysis_query_alignment"] = "strong"
            insights["correlation_score"] = 0.8
        elif analysis_severity in ["medium"] and any(
            term in query_answer for term in ["moderate", "concern", "warning"]
        ):
            insights["analysis_query_alignment"] = "good"
            insights["correlation_score"] = 0.6
        elif analysis_severity == "low" and not any(
            term in query_answer for term in ["error", "problem", "critical", "serious"]
        ):
            insights["analysis_query_alignment"] = "good"
            insights["correlation_score"] = 0.7
        else:
            insights["analysis_query_alignment"] = "weak"
            insights["correlation_score"] = 0.3

        # Add specific insights
        if insights["correlation_score"] > 0.7:
            insights["insights"].append(
                "Analysis and query results are well-aligned, increasing confidence in findings"
            )
        elif insights["correlation_score"] < 0.4:
            insights["insights"].append(
                "Analysis and query results show some discrepancy, consider additional investigation"
            )

        if analysis_results.get("patterns_detected", 0) > 0 and query_results.get("related_patterns"):
            insights["insights"].append(
                "Detected patterns correlate with query findings, supporting analysis reliability"
            )

        return insights
