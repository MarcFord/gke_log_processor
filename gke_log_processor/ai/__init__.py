"""AI-powered log analysis and highlighting for GKE logs."""

from .analyzer import (
    LogAnalysisEngine,
    PatternRecognitionEngine,
    SeverityDetectionAlgorithm,
)
from .client import GeminiClient, GeminiConfig
from .highlighter import (
    ColorScheme,
    HighlighterConfig,
    HighlightLevel,
    HighlightStyle,
    HighlightTheme,
    SeverityHighlighter,
    create_dark_theme_highlighter,
    create_default_highlighter,
    create_minimal_highlighter,
)
from .summarizer import LogSummarizer, SummarizerConfig

__all__ = [
    # Analysis Engine
    "LogAnalysisEngine",
    "SeverityDetectionAlgorithm",
    "PatternRecognitionEngine",

    # AI Client
    "GeminiClient",
    "GeminiConfig",

    # Highlighting
    "SeverityHighlighter",
    "HighlighterConfig",
    "HighlightTheme",
    "HighlightLevel",
    "HighlightStyle",
    "ColorScheme",
    "create_default_highlighter",
    "create_dark_theme_highlighter",
    "create_minimal_highlighter",

    # Summarization
    "LogSummarizer",
    "SummarizerConfig",
]
