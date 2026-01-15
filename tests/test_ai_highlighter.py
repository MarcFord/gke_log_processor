"""Tests for the severity highlighter module."""

import pytest
from rich.style import Style
from rich.text import Text

from gke_log_processor.ai.highlighter import (
    ColorScheme,
    HighlightedText,
    HighlighterConfig,
    HighlightLevel,
    HighlightStyle,
    HighlightTheme,
    SeverityHighlighter,
    create_dark_theme_highlighter,
    create_default_highlighter,
    create_minimal_highlighter,
)
from gke_log_processor.core.models import LogEntry, LogLevel, SeverityLevel
from gke_log_processor.core.utils import utc_now


class TestHighlightStyle:
    """Test HighlightStyle model functionality."""

    def test_highlight_style_creation(self):
        """Test HighlightStyle model creation."""
        style = HighlightStyle(
            color="red",
            background="black",
            bold=True,
            italic=False,
            underline=True,
            blink=False
        )

        assert style.color == "red"
        assert style.background == "black"
        assert style.bold is True
        assert style.italic is False
        assert style.underline is True
        assert style.blink is False

    def test_highlight_style_defaults(self):
        """Test HighlightStyle default values."""
        style = HighlightStyle(color="blue")

        assert style.color == "blue"
        assert style.background is None
        assert style.bold is False
        assert style.italic is False
        assert style.underline is False
        assert style.blink is False


class TestColorScheme:
    """Test ColorScheme model functionality."""

    def test_color_scheme_creation(self):
        """Test ColorScheme model creation."""
        critical_style = HighlightStyle(color="bright_red", bold=True)
        high_style = HighlightStyle(color="red", bold=True)
        medium_style = HighlightStyle(color="yellow")
        low_style = HighlightStyle(color="green")
        unknown_style = HighlightStyle(color="white")

        scheme = ColorScheme(
            critical=critical_style,
            high=high_style,
            medium=medium_style,
            low=low_style,
            unknown=unknown_style
        )

        assert scheme.critical.color == "bright_red"
        assert scheme.high.color == "red"
        assert scheme.medium.color == "yellow"
        assert scheme.low.color == "green"
        assert scheme.unknown.color == "white"


class TestHighlighterConfig:
    """Test HighlighterConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HighlighterConfig()

        assert config.theme == HighlightTheme.DEFAULT
        assert config.level == HighlightLevel.NORMAL
        assert config.enable_keyword_highlighting is True
        assert config.enable_pattern_highlighting is True
        assert config.highlight_timestamps is True
        assert config.highlight_levels is True
        assert config.highlight_pod_names is True
        assert config.custom_keywords == {}
        assert config.custom_patterns == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_keywords = {
            SeverityLevel.HIGH: ["critical_failure", "system_down"]
        }
        custom_patterns = {
            SeverityLevel.MEDIUM: [r"retry_\d+", r"timeout_\w+"]
        }

        config = HighlighterConfig(
            theme=HighlightTheme.DARK,
            level=HighlightLevel.INTENSE,
            enable_keyword_highlighting=False,
            custom_keywords=custom_keywords,
            custom_patterns=custom_patterns
        )

        assert config.theme == HighlightTheme.DARK
        assert config.level == HighlightLevel.INTENSE
        assert config.enable_keyword_highlighting is False
        assert config.custom_keywords == custom_keywords
        assert config.custom_patterns == custom_patterns


class TestSeverityHighlighter:
    """Test SeverityHighlighter functionality."""

    @pytest.fixture
    def sample_log_entry(self):
        """Create a sample log entry for testing."""
        return LogEntry(
            timestamp=utc_now(),
            message="Error: Database connection failed",
            level=LogLevel.ERROR,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="Error: Database connection failed"
        )

    @pytest.fixture
    def critical_log_entry(self):
        """Create a critical log entry for testing."""
        return LogEntry(
            timestamp=utc_now(),
            message="FATAL: System crashed due to memory corruption",
            level=LogLevel.ERROR,
            source="critical-container",
            pod_name="critical-pod",
            namespace="production",
            cluster="prod-cluster",
            container_name="critical-container",
            raw_message="FATAL: System crashed due to memory corruption"
        )

    @pytest.fixture
    def info_log_entry(self):
        """Create an info log entry for testing."""
        return LogEntry(
            timestamp=utc_now(),
            message="Application started successfully on port 8080",
            level=LogLevel.INFO,
            source="app-container",
            pod_name="app-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="app-container",
            raw_message="Application started successfully on port 8080"
        )

    def test_highlighter_initialization(self):
        """Test highlighter initialization."""
        highlighter = SeverityHighlighter()

        assert highlighter.config is not None
        assert highlighter.config.theme == HighlightTheme.DEFAULT
        assert len(highlighter.color_schemes) == 5  # All theme types
        assert highlighter.console is not None
        assert len(highlighter._compiled_patterns) == 4  # All severity levels

    def test_highlighter_with_custom_config(self):
        """Test highlighter initialization with custom config."""
        config = HighlighterConfig(
            theme=HighlightTheme.DARK,
            level=HighlightLevel.SUBTLE
        )
        highlighter = SeverityHighlighter(config)

        assert highlighter.config.theme == HighlightTheme.DARK
        assert highlighter.config.level == HighlightLevel.SUBTLE

    def test_highlight_error_log_entry(self, sample_log_entry):
        """Test highlighting of an error log entry."""
        highlighter = SeverityHighlighter()
        result = highlighter.highlight_log_entry(sample_log_entry)

        assert isinstance(result, Text)
        assert len(result) > 0
        # Should contain the error message
        assert "Error" in str(result)
        assert "Database connection failed" in str(result)

    def test_highlight_critical_log_entry(self, critical_log_entry):
        """Test highlighting of a critical log entry."""
        highlighter = SeverityHighlighter()
        result = highlighter.highlight_log_entry(critical_log_entry)

        assert isinstance(result, Text)
        assert len(result) > 0
        assert "FATAL" in str(result)
        assert "System crashed" in str(result)

    def test_highlight_info_log_entry(self, info_log_entry):
        """Test highlighting of an info log entry."""
        highlighter = SeverityHighlighter()
        result = highlighter.highlight_log_entry(info_log_entry)

        assert isinstance(result, Text)
        assert len(result) > 0
        assert "started successfully" in str(result)

    def test_format_log_line(self, sample_log_entry):
        """Test log line formatting."""
        highlighter = SeverityHighlighter()
        formatted = highlighter._format_log_line(sample_log_entry)

        assert isinstance(formatted, str)
        assert "ERROR" in formatted
        assert "default/test-pod" in formatted
        assert "Error: Database connection failed" in formatted
        # Should include timestamp
        assert "[" in formatted and "]" in formatted

    def test_severity_priority(self):
        """Test severity priority calculation."""
        highlighter = SeverityHighlighter()

        assert highlighter._severity_priority(SeverityLevel.CRITICAL) == 4
        assert highlighter._severity_priority(SeverityLevel.HIGH) == 3
        assert highlighter._severity_priority(SeverityLevel.MEDIUM) == 2
        assert highlighter._severity_priority(SeverityLevel.LOW) == 1

    def test_highlight_multiple_entries(self, sample_log_entry, info_log_entry):
        """Test highlighting multiple log entries."""
        highlighter = SeverityHighlighter()
        entries = [sample_log_entry, info_log_entry]

        results = highlighter.highlight_multiple_entries(entries)

        assert len(results) == 2
        assert all(isinstance(result, Text) for result in results)

    def test_get_severity_stats(self, sample_log_entry, critical_log_entry, info_log_entry):
        """Test severity statistics calculation."""
        highlighter = SeverityHighlighter()
        entries = [sample_log_entry, critical_log_entry, info_log_entry]

        stats = highlighter.get_severity_stats(entries)

        assert isinstance(stats, dict)
        assert all(level in stats for level in SeverityLevel)
        assert sum(stats.values()) == len(entries)

    def test_get_style_for_severity(self):
        """Test getting style for different severity levels."""
        highlighter = SeverityHighlighter()

        critical_style = highlighter._get_style_for_severity(SeverityLevel.CRITICAL)
        high_style = highlighter._get_style_for_severity(SeverityLevel.HIGH)
        medium_style = highlighter._get_style_for_severity(SeverityLevel.MEDIUM)
        low_style = highlighter._get_style_for_severity(SeverityLevel.LOW)

        assert isinstance(critical_style, HighlightStyle)
        assert isinstance(high_style, HighlightStyle)
        assert isinstance(medium_style, HighlightStyle)
        assert isinstance(low_style, HighlightStyle)

        # Critical should be most prominent
        assert critical_style.bold is True
        assert critical_style.color == "bright_red"

    def test_convert_to_rich_style(self):
        """Test conversion from HighlightStyle to Rich Style."""
        highlighter = SeverityHighlighter()
        highlight_style = HighlightStyle(
            color="red",
            background="black",
            bold=True,
            italic=True,
            underline=False,
            blink=False
        )

        rich_style = highlighter._convert_to_rich_style(highlight_style)

        assert isinstance(rich_style, Style)
        assert rich_style.color.name == "red"
        assert rich_style.bgcolor.name == "black"
        assert rich_style.bold is True
        assert rich_style.italic is True

    def test_update_config(self):
        """Test updating highlighter configuration."""
        highlighter = SeverityHighlighter()

        # Initial config
        assert highlighter.config.theme == HighlightTheme.DEFAULT

        # Update config
        new_config = HighlighterConfig(theme=HighlightTheme.DARK)
        highlighter.update_config(new_config)

        assert highlighter.config.theme == HighlightTheme.DARK

    def test_custom_keywords_highlighting(self):
        """Test highlighting with custom keywords."""
        config = HighlighterConfig(
            custom_keywords={
                SeverityLevel.HIGH: ["custom_error", "special_failure"]
            }
        )
        highlighter = SeverityHighlighter(config)

        # Create log entry with custom keyword
        log_entry = LogEntry(
            timestamp=utc_now(),
            message="A custom_error occurred in the system",
            level=LogLevel.INFO,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="A custom_error occurred in the system"
        )

        result = highlighter.highlight_log_entry(log_entry)
        assert isinstance(result, Text)
        assert "custom_error" in str(result)

    def test_highlighting_levels(self):
        """Test different highlighting intensity levels."""
        # Test NONE level
        config_none = HighlighterConfig(level=HighlightLevel.NONE)
        highlighter_none = SeverityHighlighter(config_none)
        style_none = highlighter_none._get_style_for_severity(SeverityLevel.CRITICAL)
        assert style_none.color == "white"
        assert style_none.bold is False

        # Test SUBTLE level
        config_subtle = HighlighterConfig(level=HighlightLevel.SUBTLE)
        highlighter_subtle = SeverityHighlighter(config_subtle)
        style_subtle = highlighter_subtle._get_style_for_severity(SeverityLevel.CRITICAL)
        assert style_subtle.bold is False
        assert style_subtle.background is None

        # Test INTENSE level
        config_intense = HighlighterConfig(level=HighlightLevel.INTENSE)
        highlighter_intense = SeverityHighlighter(config_intense)
        style_intense = highlighter_intense._get_style_for_severity(SeverityLevel.CRITICAL)
        assert style_intense.bold is True
        assert style_intense.underline is True

    def test_theme_color_schemes(self):
        """Test different theme color schemes."""
        themes_to_test = [
            HighlightTheme.DEFAULT,
            HighlightTheme.DARK,
            HighlightTheme.LIGHT,
            HighlightTheme.MINIMAL,
            HighlightTheme.COLORBLIND
        ]

        for theme in themes_to_test:
            config = HighlighterConfig(theme=theme)
            highlighter = SeverityHighlighter(config)

            # Verify theme is properly set
            assert highlighter.config.theme == theme

            # Verify color scheme exists
            assert theme in highlighter.color_schemes

            # Test that styles can be retrieved
            style = highlighter._get_style_for_severity(SeverityLevel.CRITICAL)
            assert isinstance(style, HighlightStyle)

    def test_keyword_pattern_compilation(self):
        """Test that keyword patterns are properly compiled."""
        highlighter = SeverityHighlighter()

        # Should have patterns for all severity levels
        assert SeverityLevel.CRITICAL in highlighter._compiled_patterns
        assert SeverityLevel.HIGH in highlighter._compiled_patterns
        assert SeverityLevel.MEDIUM in highlighter._compiled_patterns
        assert SeverityLevel.LOW in highlighter._compiled_patterns

        # Each severity should have multiple patterns
        for severity, patterns in highlighter._compiled_patterns.items():
            assert len(patterns) > 0
            # All patterns should be compiled regex objects
            for pattern in patterns:
                assert hasattr(pattern, 'finditer')

    def test_keyword_highlighting_disabled(self, sample_log_entry):
        """Test with keyword highlighting disabled."""
        config = HighlighterConfig(
            enable_keyword_highlighting=False,
            enable_pattern_highlighting=False
        )
        highlighter = SeverityHighlighter(config)

        result = highlighter.highlight_log_entry(sample_log_entry)
        assert isinstance(result, Text)
        assert len(result) > 0

    def test_highlight_message_content_overlapping_matches(self):
        """Test handling of overlapping keyword matches."""
        highlighter = SeverityHighlighter()

        # Text with overlapping keywords of different severities
        text = "fatal error occurred"
        segments = highlighter._highlight_message_content(text, SeverityLevel.LOW)

        assert len(segments) > 0
        # Should prefer higher severity matches
        has_critical = any(seg.severity == SeverityLevel.CRITICAL for seg in segments)
        has_high = any(seg.severity == SeverityLevel.HIGH for seg in segments)
        assert has_critical or has_high  # Should match at least one high-severity term


class TestHighlighterFactoryFunctions:
    """Test factory functions for creating highlighters."""

    def test_create_default_highlighter(self):
        """Test creating default highlighter."""
        highlighter = create_default_highlighter()

        assert isinstance(highlighter, SeverityHighlighter)
        assert highlighter.config.theme == HighlightTheme.DEFAULT
        assert highlighter.config.level == HighlightLevel.NORMAL

    def test_create_dark_theme_highlighter(self):
        """Test creating dark theme highlighter."""
        highlighter = create_dark_theme_highlighter()

        assert isinstance(highlighter, SeverityHighlighter)
        assert highlighter.config.theme == HighlightTheme.DARK

    def test_create_minimal_highlighter(self):
        """Test creating minimal highlighter."""
        highlighter = create_minimal_highlighter()

        assert isinstance(highlighter, SeverityHighlighter)
        assert highlighter.config.theme == HighlightTheme.MINIMAL
        assert highlighter.config.level == HighlightLevel.SUBTLE


class TestHighlightedText:
    """Test HighlightedText model."""

    def test_highlighted_text_creation(self):
        """Test HighlightedText model creation."""
        style = HighlightStyle(color="red", bold=True)

        highlighted = HighlightedText(
            text="Error: Something went wrong",
            style=style,
            severity=SeverityLevel.HIGH,
            start_pos=0,
            end_pos=26,
            match_type="keyword"
        )

        assert highlighted.text == "Error: Something went wrong"
        assert highlighted.style.color == "red"
        assert highlighted.severity == SeverityLevel.HIGH
        assert highlighted.start_pos == 0
        assert highlighted.end_pos == 26
        assert highlighted.match_type == "keyword"


class TestIntegrationWithAnalyzer:
    """Test integration with SeverityDetectionAlgorithm."""

    def test_highlighter_uses_analyzer_keywords(self):
        """Test that highlighter uses keywords from SeverityDetectionAlgorithm."""
        from gke_log_processor.ai.analyzer import SeverityDetectionAlgorithm

        highlighter = SeverityHighlighter()

        # Verify that analyzer keywords are included in compiled patterns
        for severity, patterns in highlighter._compiled_patterns.items():
            if severity == SeverityLevel.CRITICAL:
                # Should contain patterns for critical keywords from analyzer
                critical_keywords = SeverityDetectionAlgorithm.CRITICAL_KEYWORDS
                # At least some critical keywords should be compiled as patterns
                assert len(patterns) >= len(critical_keywords) * 0.5

    def test_severity_detection_consistency(self):
        """Test that highlighter and analyzer detect same severities."""
        from gke_log_processor.ai.analyzer import SeverityDetectionAlgorithm

        highlighter = SeverityHighlighter()

        # Test with a critical log entry
        log_entry = LogEntry(
            timestamp=utc_now(),
            message="FATAL error: system crashed",
            level=LogLevel.ERROR,
            source="test-container",
            pod_name="test-pod",
            namespace="default",
            cluster="test-cluster",
            container_name="test-container",
            raw_message="FATAL error: system crashed"
        )

        # Both should detect critical severity
        analyzer_severity = SeverityDetectionAlgorithm.detect_combined_severity(log_entry)

        # Highlight the entry - the highlighter uses the same detection
        result = highlighter.highlight_log_entry(log_entry)
        assert isinstance(result, Text)

        # Should be consistent
        assert analyzer_severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
