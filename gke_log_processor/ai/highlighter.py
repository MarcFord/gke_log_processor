"""Severity highlighting and automatic error/warning detection for log entries."""

import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field
from rich.console import Console
from rich.style import Style
from rich.text import Text

from ..core.logging import get_logger
from ..core.models import LogEntry, LogLevel, SeverityLevel

logger = get_logger(__name__)


class HighlightTheme(str, Enum):
    """Available highlighting themes."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    COLORBLIND = "colorblind"


class HighlightLevel(str, Enum):
    """Highlighting intensity levels."""

    NONE = "none"
    SUBTLE = "subtle"
    NORMAL = "normal"
    BOLD = "bold"
    INTENSE = "intense"


class HighlightStyle(BaseModel):
    """Style configuration for a specific severity level."""

    color: str = Field(..., description="Text color")
    background: Optional[str] = Field(None, description="Background color")
    bold: bool = Field(False, description="Bold text")
    italic: bool = Field(False, description="Italic text")
    underline: bool = Field(False, description="Underlined text")
    blink: bool = Field(False, description="Blinking text")


class ColorScheme(BaseModel):
    """Color scheme for different severity levels."""

    critical: HighlightStyle = Field(..., description="Critical severity style")
    high: HighlightStyle = Field(..., description="High severity style")
    medium: HighlightStyle = Field(..., description="Medium severity style")
    low: HighlightStyle = Field(..., description="Low severity style")
    unknown: HighlightStyle = Field(..., description="Unknown severity style")


class HighlighterConfig(BaseModel):
    """Configuration for the severity highlighter."""

    theme: HighlightTheme = Field(
        HighlightTheme.DEFAULT,
        description="Highlighting theme"
    )
    level: HighlightLevel = Field(
        HighlightLevel.NORMAL,
        description="Highlighting intensity"
    )
    enable_keyword_highlighting: bool = Field(
        True,
        description="Enable keyword-based highlighting"
    )
    enable_pattern_highlighting: bool = Field(
        True,
        description="Enable pattern-based highlighting"
    )
    highlight_timestamps: bool = Field(
        True,
        description="Highlight timestamps"
    )
    highlight_levels: bool = Field(
        True,
        description="Highlight log levels"
    )
    highlight_pod_names: bool = Field(
        True,
        description="Highlight pod names"
    )
    custom_keywords: Dict[SeverityLevel, List[str]] = Field(
        default_factory=dict,
        description="Custom keywords for each severity level"
    )
    custom_patterns: Dict[SeverityLevel, List[str]] = Field(
        default_factory=dict,
        description="Custom regex patterns for each severity level"
    )


class HighlightedText(BaseModel):
    """A highlighted text segment with styling information."""

    text: str = Field(..., description="Text content")
    style: HighlightStyle = Field(..., description="Applied style")
    severity: SeverityLevel = Field(..., description="Detected severity")
    start_pos: int = Field(..., description="Start position in original text")
    end_pos: int = Field(..., description="End position in original text")
    match_type: str = Field(..., description="Type of match (keyword, pattern, level)")


class SeverityHighlighter:
    """Automatic severity highlighting for log entries."""

    # Severity keywords - duplicated from analyzer to avoid circular import
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

    def __init__(self, config: Optional[HighlighterConfig] = None):
        """Initialize the severity highlighter.

        Args:
            config: Highlighter configuration. Defaults to standard config.
        """
        self.config = config or HighlighterConfig()
        self.color_schemes = self._initialize_color_schemes()
        self.console = Console()
        self._compiled_patterns: Dict[SeverityLevel, List[re.Pattern]] = {}
        self._initialize_patterns()

    def detect_combined_severity(self, log_entry: LogEntry) -> SeverityLevel:
        """Detect severity using the same logic as SeverityDetectionAlgorithm."""
        keyword_severity = self._detect_severity_by_keywords(log_entry.message)
        pattern_severity = self._detect_severity_by_patterns(log_entry.message)
        level_severity = self._detect_severity_by_log_level(log_entry)

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

    def _detect_severity_by_keywords(self, message: str) -> SeverityLevel:
        """Detect severity based on keyword analysis."""
        message_lower = message.lower()

        # Check for critical keywords
        if any(keyword in message_lower for keyword in self.CRITICAL_KEYWORDS):
            return SeverityLevel.CRITICAL

        # Check for high severity keywords
        if any(keyword in message_lower for keyword in self.HIGH_KEYWORDS):
            return SeverityLevel.HIGH

        # Check for medium severity keywords
        if any(keyword in message_lower for keyword in self.MEDIUM_KEYWORDS):
            return SeverityLevel.MEDIUM

        # Check for low severity keywords
        if any(keyword in message_lower for keyword in self.LOW_KEYWORDS):
            return SeverityLevel.LOW

        return SeverityLevel.LOW

    def _detect_severity_by_patterns(self, message: str) -> SeverityLevel:
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

    def _detect_severity_by_log_level(self, log_entry: LogEntry) -> SeverityLevel:
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

    def _initialize_color_schemes(self) -> Dict[HighlightTheme, ColorScheme]:
        """Initialize color schemes for different themes."""
        schemes = {}

        # Default theme - standard colors
        schemes[HighlightTheme.DEFAULT] = ColorScheme(
            critical=HighlightStyle(
                color="bright_red",
                bold=True,
                background="red"
            ),
            high=HighlightStyle(
                color="red",
                bold=True
            ),
            medium=HighlightStyle(
                color="yellow",
                bold=False
            ),
            low=HighlightStyle(
                color="green",
                bold=False
            ),
            unknown=HighlightStyle(
                color="white",
                bold=False
            )
        )

        # Dark theme - optimized for dark backgrounds
        schemes[HighlightTheme.DARK] = ColorScheme(
            critical=HighlightStyle(
                color="bright_red",
                bold=True,
                background="dark_red"
            ),
            high=HighlightStyle(
                color="bright_magenta",
                bold=True
            ),
            medium=HighlightStyle(
                color="bright_yellow",
                bold=False
            ),
            low=HighlightStyle(
                color="bright_green",
                bold=False
            ),
            unknown=HighlightStyle(
                color="bright_white",
                bold=False
            )
        )

        # Light theme - optimized for light backgrounds
        schemes[HighlightTheme.LIGHT] = ColorScheme(
            critical=HighlightStyle(
                color="dark_red",
                bold=True,
                background="light_red"
            ),
            high=HighlightStyle(
                color="red",
                bold=True
            ),
            medium=HighlightStyle(
                color="dark_orange",
                bold=False
            ),
            low=HighlightStyle(
                color="dark_green",
                bold=False
            ),
            unknown=HighlightStyle(
                color="black",
                bold=False
            )
        )

        # Minimal theme - subtle highlighting
        schemes[HighlightTheme.MINIMAL] = ColorScheme(
            critical=HighlightStyle(
                color="red",
                bold=True
            ),
            high=HighlightStyle(
                color="red",
                bold=False
            ),
            medium=HighlightStyle(
                color="yellow",
                bold=False
            ),
            low=HighlightStyle(
                color="white",
                bold=False
            ),
            unknown=HighlightStyle(
                color="white",
                bold=False
            )
        )

        # Colorblind-friendly theme
        schemes[HighlightTheme.COLORBLIND] = ColorScheme(
            critical=HighlightStyle(
                color="white",
                bold=True,
                background="black",
                underline=True
            ),
            high=HighlightStyle(
                color="white",
                bold=True,
                underline=True
            ),
            medium=HighlightStyle(
                color="white",
                bold=False,
                italic=True
            ),
            low=HighlightStyle(
                color="white",
                bold=False
            ),
            unknown=HighlightStyle(
                color="white",
                bold=False
            )
        )

        return schemes

    def _initialize_patterns(self):
        """Initialize compiled regex patterns for severity detection."""
        # Base patterns from internal keyword sets
        base_patterns = {
            SeverityLevel.CRITICAL: list(self.CRITICAL_KEYWORDS),
            SeverityLevel.HIGH: list(self.HIGH_KEYWORDS),
            SeverityLevel.MEDIUM: list(self.MEDIUM_KEYWORDS),
            SeverityLevel.LOW: list(self.LOW_KEYWORDS)
        }

        # Add custom keywords from config
        for severity, keywords in self.config.custom_keywords.items():
            if severity in base_patterns:
                base_patterns[severity].extend(keywords)

        # Add custom patterns from config
        for severity, patterns in self.config.custom_patterns.items():
            if severity in base_patterns:
                base_patterns[severity].extend(patterns)

        # Compile patterns
        for severity, keywords in base_patterns.items():
            patterns = []
            for keyword in keywords:
                try:
                    # Create word boundary patterns for keywords
                    if keyword.isalpha():
                        pattern = rf'\b{re.escape(keyword)}\b'
                    else:
                        pattern = re.escape(keyword)
                    patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{keyword}': {e}")

            self._compiled_patterns[severity] = patterns

    def highlight_log_entry(self, log_entry: LogEntry) -> Text:
        """Highlight a complete log entry with severity-based styling.

        Args:
            log_entry: Log entry to highlight

        Returns:
            Rich Text object with applied highlighting
        """
        # Detect overall severity
        detected_severity = self.detect_combined_severity(log_entry)

        # Create Rich Text object
        text = Text()

        # Build formatted log line
        log_line = self._format_log_line(log_entry)

        # Apply highlighting based on configuration
        if self.config.enable_keyword_highlighting or self.config.enable_pattern_highlighting:
            highlighted_segments = self._highlight_message_content(
                log_line, detected_severity
            )
        else:
            # Basic highlighting based on overall severity
            style = self._get_style_for_severity(detected_severity)
            highlighted_segments = [
                HighlightedText(
                    text=log_line,
                    style=style,
                    severity=detected_severity,
                    start_pos=0,
                    end_pos=len(log_line),
                    match_type="overall"
                )
            ]

        # Apply segments to Rich Text
        last_pos = 0
        for segment in highlighted_segments:
            # Add any unhighlighted text before this segment
            if segment.start_pos > last_pos:
                text.append(log_line[last_pos:segment.start_pos])

            # Add highlighted segment
            rich_style = self._convert_to_rich_style(segment.style)
            text.append(segment.text, style=rich_style)

            last_pos = segment.end_pos

        # Add any remaining unhighlighted text
        if last_pos < len(log_line):
            text.append(log_line[last_pos:])

        return text

    def _format_log_line(self, log_entry: LogEntry) -> str:
        """Format a log entry into a display string.

        Args:
            log_entry: Log entry to format

        Returns:
            Formatted log line string
        """
        timestamp_str = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        level_str = log_entry.level.value if log_entry.level else "UNKNOWN"
        pod_str = f"{log_entry.namespace}/{log_entry.pod_name}"

        return f"[{timestamp_str}] [{level_str}] {pod_str}: {log_entry.message}"

    def _highlight_message_content(
        self,
        text: str,
        base_severity: SeverityLevel
    ) -> List[HighlightedText]:
        """Highlight specific keywords and patterns in text content.

        Args:
            text: Text to analyze and highlight
            base_severity: Base severity level for the text

        Returns:
            List of highlighted text segments
        """
        segments = []
        matches = []

        # Find all keyword/pattern matches
        for severity, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append({
                        'start': match.start(),
                        'end': match.end(),
                        'severity': severity,
                        'text': match.group(),
                        'type': 'keyword'
                    })

        # Sort matches by position
        matches.sort(key=lambda x: x['start'])

        # Handle overlapping matches - prefer higher severity
        final_matches = []
        for match in matches:
            # Check if this match overlaps with any previous match
            overlaps = False
            for existing in final_matches:
                if (match['start'] < existing['end'] and
                        match['end'] > existing['start']):
                    # Overlapping - keep the higher severity match
                    if self._severity_priority(match['severity']) > self._severity_priority(existing['severity']):
                        final_matches.remove(existing)
                    else:
                        overlaps = True
                    break

            if not overlaps:
                final_matches.append(match)

        # Convert matches to HighlightedText segments
        if not final_matches:
            # No specific matches, use base severity
            style = self._get_style_for_severity(base_severity)
            segments.append(
                HighlightedText(
                    text=text,
                    style=style,
                    severity=base_severity,
                    start_pos=0,
                    end_pos=len(text),
                    match_type="overall"
                )
            )
        else:
            # Create segments for matched and unmatched portions
            last_pos = 0
            for match in final_matches:
                # Add unmatched portion before this match
                if match['start'] > last_pos:
                    style = self._get_style_for_severity(base_severity)
                    segments.append(
                        HighlightedText(
                            text=text[last_pos:match['start']],
                            style=style,
                            severity=base_severity,
                            start_pos=last_pos,
                            end_pos=match['start'],
                            match_type="text"
                        )
                    )

                # Add matched portion with specific severity styling
                style = self._get_style_for_severity(match['severity'])
                segments.append(
                    HighlightedText(
                        text=match['text'],
                        style=style,
                        severity=match['severity'],
                        start_pos=match['start'],
                        end_pos=match['end'],
                        match_type=match['type']
                    )
                )

                last_pos = match['end']

            # Add any remaining text after the last match
            if last_pos < len(text):
                style = self._get_style_for_severity(base_severity)
                segments.append(
                    HighlightedText(
                        text=text[last_pos:],
                        style=style,
                        severity=base_severity,
                        start_pos=last_pos,
                        end_pos=len(text),
                        match_type="text"
                    )
                )

        return segments

    def _severity_priority(self, severity: SeverityLevel) -> int:
        """Get numeric priority for severity level (higher = more severe).

        Args:
            severity: Severity level

        Returns:
            Numeric priority value
        """
        priority_map = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1
        }
        return priority_map.get(severity, 0)

    def _get_style_for_severity(self, severity: SeverityLevel) -> HighlightStyle:
        """Get highlighting style for a severity level.

        Args:
            severity: Severity level

        Returns:
            Highlight style configuration
        """
        color_scheme = self.color_schemes[self.config.theme]

        style_map = {
            SeverityLevel.CRITICAL: color_scheme.critical,
            SeverityLevel.HIGH: color_scheme.high,
            SeverityLevel.MEDIUM: color_scheme.medium,
            SeverityLevel.LOW: color_scheme.low
        }

        base_style = style_map.get(severity, color_scheme.unknown)

        # Modify style based on highlighting level
        if self.config.level == HighlightLevel.NONE:
            return HighlightStyle(color="white", bold=False)
        elif self.config.level == HighlightLevel.SUBTLE:
            return HighlightStyle(
                color=base_style.color,
                bold=False,
                background=None
            )
        elif self.config.level == HighlightLevel.INTENSE:
            return HighlightStyle(
                color=base_style.color,
                bold=True,
                background=base_style.background or "black",
                underline=True
            )
        else:
            return base_style

    def _convert_to_rich_style(self, highlight_style: HighlightStyle) -> Style:
        """Convert HighlightStyle to Rich Style object.

        Args:
            highlight_style: Highlight style configuration

        Returns:
            Rich Style object
        """
        return Style(
            color=highlight_style.color,
            bgcolor=highlight_style.background,
            bold=highlight_style.bold,
            italic=highlight_style.italic,
            underline=highlight_style.underline,
            blink=highlight_style.blink
        )

    def highlight_multiple_entries(self, log_entries: List[LogEntry]) -> List[Text]:
        """Highlight multiple log entries efficiently.

        Args:
            log_entries: List of log entries to highlight

        Returns:
            List of highlighted Rich Text objects
        """
        return [self.highlight_log_entry(entry) for entry in log_entries]

    def get_severity_stats(self, log_entries: List[LogEntry]) -> Dict[SeverityLevel, int]:
        """Get statistics on severity levels in a list of log entries.

        Args:
            log_entries: List of log entries to analyze

        Returns:
            Dictionary mapping severity levels to counts
        """
        stats = {level: 0 for level in SeverityLevel}

        for entry in log_entries:
            severity = self.detect_combined_severity(entry)
            stats[severity] += 1

        return stats

    def update_config(self, config: HighlighterConfig):
        """Update highlighter configuration and reinitialize patterns.

        Args:
            config: New highlighter configuration
        """
        self.config = config
        self._compiled_patterns.clear()
        self._initialize_patterns()


def create_default_highlighter() -> SeverityHighlighter:
    """Create a default severity highlighter with standard configuration.

    Returns:
        Configured severity highlighter
    """
    return SeverityHighlighter(HighlighterConfig())


def create_dark_theme_highlighter() -> SeverityHighlighter:
    """Create a severity highlighter optimized for dark terminals.

    Returns:
        Configured severity highlighter with dark theme
    """
    config = HighlighterConfig(theme=HighlightTheme.DARK)
    return SeverityHighlighter(config)


def create_minimal_highlighter() -> SeverityHighlighter:
    """Create a minimal severity highlighter with subtle highlighting.

    Returns:
        Configured severity highlighter with minimal theme
    """
    config = HighlighterConfig(
        theme=HighlightTheme.MINIMAL,
        level=HighlightLevel.SUBTLE
    )
    return SeverityHighlighter(config)
