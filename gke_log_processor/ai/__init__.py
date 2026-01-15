"""AI-powered log analysis and highlighting for GKE logs."""

# Import custom query models from core.models
from ..core.models import (
    QueryConfig,
    QueryRequest,
    QueryResponse,
    QueryType,
)
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
from .patterns import (
    AdvancedPatternDetector,
    AnomalyPattern,
    CascadePattern,
    PatternDetectionConfig,
    PatternDetectionResult,
    PatternSimilarity,
    RecurringIssuePattern,
    TemporalPattern,
)
from .summarizer import (
    KeyInsight,
    LogSummarizer,
    LogSummaryReport,
    SummarizerConfig,
    SummaryType,
    TimeWindowSize,
    TimeWindowSummary,
    TrendAnalysis,
)

__all__ = [
    # Analysis Engine
    "LogAnalysisEngine",
    "SeverityDetectionAlgorithm",
    "PatternRecognitionEngine",

    # AI Client
    "GeminiClient",
    "GeminiConfig",

    # Pattern Detection
    "AdvancedPatternDetector",
    "PatternDetectionConfig",
    "PatternDetectionResult",
    "RecurringIssuePattern",
    "TemporalPattern",
    "CascadePattern",
    "AnomalyPattern",
    "PatternSimilarity",

    # Severity Highlighting
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
    "LogSummaryReport",
    "SummarizerConfig",
    "TimeWindowSize",
    "SummaryType",
    "TimeWindowSummary",
    "TrendAnalysis",
    "KeyInsight",

    # Custom Queries (Phase 2.2 AI Feature)
    "QueryConfig",
    "QueryRequest",
    "QueryResponse",
    "QueryType",
]
