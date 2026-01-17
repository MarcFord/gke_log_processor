"""Gemini AI client for log analysis and processing."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from ..core.exceptions import LogProcessingError
from ..core.logging import get_logger
from ..core.models import AIAnalysisResult, DetectedPattern, LogEntry, SeverityLevel
from ..core.utils import utc_now


class RateLimitConfig(BaseModel):
    """Configuration for API rate limiting."""

    requests_per_minute: int = Field(default=60, ge=1, le=1000)
    requests_per_hour: int = Field(default=1000, ge=1, le=10000)
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    max_retries: int = Field(default=3, ge=0, le=10)


class GeminiConfig(BaseModel):
    """Configuration for Gemini AI client."""

    api_key: str = Field(..., description="Gemini API key")
    model: str = Field(default="gemini-1.5-pro", description="Model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192, ge=1, le=32768)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=64, ge=1, le=100)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    def configure_genai(self) -> None:
        """Configure the Google Generative AI library."""
        # Configuration is now handled by the Client object
        pass


class RequestTracker:
    """Track API requests for rate limiting."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests_per_minute: List[datetime] = []
        self.requests_per_hour: List[datetime] = []
        self.active_requests = 0
        self._lock = asyncio.Lock()

    async def can_make_request(self) -> bool:
        """Check if we can make a request within rate limits."""
        async with self._lock:
            now = utc_now()

            # Clean old requests
            self._clean_old_requests(now)

            # Check rate limits
            if len(self.requests_per_minute) >= self.config.requests_per_minute:
                return False
            if len(self.requests_per_hour) >= self.config.requests_per_hour:
                return False
            if self.active_requests >= self.config.max_concurrent_requests:
                return False

            return True

    async def record_request_start(self) -> None:
        """Record the start of a request."""
        async with self._lock:
            now = utc_now()
            self.requests_per_minute.append(now)
            self.requests_per_hour.append(now)
            self.active_requests += 1

    async def record_request_end(self) -> None:
        """Record the end of a request."""
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    def _clean_old_requests(self, now: datetime) -> None:
        """Remove old request timestamps."""
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        self.requests_per_minute = [
            req_time for req_time in self.requests_per_minute
            if req_time > minute_ago
        ]
        self.requests_per_hour = [
            req_time for req_time in self.requests_per_hour
            if req_time > hour_ago
        ]

    async def wait_for_rate_limit(self) -> float:
        """Wait until we can make a request. Returns wait time in seconds."""
        wait_time = 0.1
        total_wait = 0.0

        while not await self.can_make_request():
            await asyncio.sleep(wait_time)
            total_wait += wait_time
            wait_time = min(wait_time * self.config.backoff_factor, 60.0)

        return total_wait


class GeminiError(LogProcessingError):
    """Base exception for Gemini AI errors."""

    pass


class GeminiRateLimitError(GeminiError):
    """Rate limit exceeded error."""

    pass


class GeminiAuthenticationError(GeminiError):
    """Authentication failed error."""

    pass


class GeminiClient:
    """Client for interacting with Google's Gemini AI API."""

    def __init__(self, config: GeminiConfig):
        """Initialize the Gemini client.

        Args:
            config: Gemini configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.request_tracker = RequestTracker(config.rate_limit)
        self._client: Optional[genai.Client] = None

        # Configure the API
        config.configure_genai()

        # Safety settings - be more permissive for log analysis
        self.safety_settings = [
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            ),
        ]

    @property
    def client(self) -> genai.Client:
        """Get the configured client instance."""
        if self._client is None:
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    async def test_connection(self) -> bool:
        """Test the connection to Gemini API.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            response = await self._make_request_with_retry(
                "Test connection. Respond with 'OK'."
            )
            return response is not None and "ok" in response.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def analyze_logs(
        self,
        log_entries: List[LogEntry],
        analysis_type: str = "comprehensive",
        context: Optional[Dict[str, Any]] = None
    ) -> AIAnalysisResult:
        """Analyze log entries for patterns, severity, and insights.

        Args:
            log_entries: List of log entries to analyze
            analysis_type: Type of analysis ('quick', 'comprehensive', 'security')
            context: Additional context for analysis

        Returns:
            AIAnalysisResult: Analysis results
        """
        if not log_entries:
            raise GeminiError("No log entries provided for analysis")

        self.logger.info(f"Analyzing {len(log_entries)} log entries with {analysis_type} analysis")

        # Prepare the prompt based on analysis type
        prompt = self._build_analysis_prompt(log_entries, analysis_type, context)

        try:
            # Make the API request
            response_text = await self._make_request_with_retry(prompt)

            # Parse the response into structured results
            analysis_result = self._parse_analysis_response(
                response_text, log_entries, analysis_type
            )

            self.logger.info(f"Analysis complete: {analysis_result.overall_severity} severity")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Log analysis failed: {e}")
            raise GeminiError(f"Failed to analyze logs: {e}") from e

    async def detect_patterns(
        self,
        log_entries: List[LogEntry],
        pattern_types: Optional[List[str]] = None
    ) -> List[DetectedPattern]:
        """Detect specific patterns in log entries.

        Args:
            log_entries: Log entries to analyze
            pattern_types: Specific pattern types to look for

        Returns:
            List[DetectedPattern]: Detected patterns
        """
        if not log_entries:
            return []

        prompt = self._build_pattern_detection_prompt(log_entries, pattern_types)

        try:
            response_text = await self._make_request_with_retry(prompt)
            patterns = self._parse_pattern_response(response_text, log_entries)

            self.logger.info(f"Detected {len(patterns)} patterns in {len(log_entries)} log entries")
            return patterns

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            raise GeminiError(f"Failed to detect patterns: {e}") from e

    async def summarize_logs(
        self,
        log_entries: List[LogEntry],
        summary_style: str = "executive",
        max_length: int = 500
    ) -> str:
        """Generate a summary of log entries.

        Args:
            log_entries: Log entries to summarize
            summary_style: Style of summary ('executive', 'technical', 'brief')
            max_length: Maximum length of summary in words

        Returns:
            str: Generated summary
        """
        if not log_entries:
            return "No log entries to summarize."

        prompt = self._build_summary_prompt(log_entries, summary_style, max_length)

        try:
            summary = await self._make_request_with_retry(prompt)
            self.logger.info(f"Generated {summary_style} summary of {len(log_entries)} log entries")
            return summary

        except Exception as e:
            self.logger.error(f"Log summarization failed: {e}")
            raise GeminiError(f"Failed to summarize logs: {e}") from e

    async def query_logs(self, log_entries: List[LogEntry], question: str) -> str:
        """Answer a natural language question about the logs.

        Args:
            log_entries: Log entries to query against
            question: Natural language question

        Returns:
            str: Answer to the question
        """
        if not log_entries:
            return "No log entries available to answer the question."

        prompt = self._build_query_prompt(log_entries, question)

        try:
            answer = await self._make_request_with_retry(prompt)
            self.logger.info(f"Answered question about {len(log_entries)} log entries")
            return answer

        except Exception as e:
            self.logger.error(f"Log query failed: {e}")
            raise GeminiError(f"Failed to query logs: {e}") from e

    async def _make_request_with_retry(self, prompt: str) -> str:
        """Make a request to Gemini with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            str: The response text
        """
        for attempt in range(self.config.rate_limit.max_retries + 1):
            try:
                # Wait for rate limit
                wait_time = await self.request_tracker.wait_for_rate_limit()
                if wait_time > 0:
                    self.logger.debug(f"Waited {wait_time:.2f}s for rate limit")

                # Record request start
                await self.request_tracker.record_request_start()

                try:
                    # Make the API call
                    response = await self._make_api_call(prompt)
                    return response

                finally:
                    # Always record request end
                    await self.request_tracker.record_request_end()

            except Exception as e:
                if attempt == self.config.rate_limit.max_retries:
                    raise

                # Check if it's a rate limit error
                error_str = str(e).lower()
                if "quota" in error_str or "rate limit" in error_str:
                    backoff_time = (self.config.rate_limit.backoff_factor ** attempt) * 2
                    self.logger.warning(f"Rate limit hit, backing off {backoff_time}s")
                    await asyncio.sleep(backoff_time)
                    continue

                # Check if it's an authentication error
                if "authentication" in error_str or "api key" in error_str:
                    raise GeminiAuthenticationError(f"Authentication failed: {e}") from e

                # For other errors, exponential backoff
                backoff_time = (self.config.rate_limit.backoff_factor ** attempt)
                self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {backoff_time}s: {e}")
                await asyncio.sleep(backoff_time)

    async def _make_api_call(self, prompt: str) -> str:
        """Make the actual API call.

        Args:
            prompt: The prompt to send

        Returns:
            str: The response text
        """
        try:
            # Generate content with new API
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    max_output_tokens=self.config.max_output_tokens,
                    safety_settings=self.safety_settings,
                )
            )

            if not response.text:
                raise GeminiError("Empty response from Gemini API")

            return response.text.strip()

        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise

    def _build_analysis_prompt(
        self,
        log_entries: List[LogEntry],
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a prompt for log analysis.

        Args:
            log_entries: Log entries to analyze
            analysis_type: Type of analysis
            context: Additional context

        Returns:
            str: The constructed prompt
        """
        # Limit the number of logs for analysis to avoid token limits
        max_logs = 50 if analysis_type == "comprehensive" else 30
        sample_logs = log_entries[:max_logs]

        # Build context information
        total_logs = len(log_entries)
        time_range = self._get_time_range(log_entries)
        log_levels = self._get_log_level_distribution(log_entries)

        context_str = ""
        if context:
            context_str = f"\n\nAdditional context: {context}"

        # Create the analysis prompt
        prompt = f"""You are an expert log analyst. Analyze the following Kubernetes pod logs and provide insights.

Analysis Type: {analysis_type}
Total Log Entries: {total_logs} (showing first {len(sample_logs)})
Time Range: {time_range}
Log Level Distribution: {log_levels}
{context_str}

Please analyze these logs and provide:

1. Overall Severity Assessment (low/medium/high/critical)
2. Key Issues and Patterns Detected
3. Error Analysis and Recommendations
4. Performance Insights
5. Security Concerns (if any)
6. Recommended Actions

Log Entries:
"""

        for i, entry in enumerate(sample_logs, 1):
            prompt += f"\n{i}. [{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            prompt += f"[{entry.level or 'INFO'}] "
            prompt += f"[{entry.pod_name}/{entry.container_name}] "
            prompt += f"{entry.message[:200]}"  # Truncate long messages

        prompt += "\n\nProvide your analysis in a structured format with clear recommendations."

        return prompt

    def _build_pattern_detection_prompt(
        self,
        log_entries: List[LogEntry],
        pattern_types: Optional[List[str]] = None
    ) -> str:
        """Build a prompt for pattern detection."""
        max_logs = 100
        sample_logs = log_entries[:max_logs]

        pattern_focus = ""
        if pattern_types:
            pattern_focus = f"\nFocus on these pattern types: {', '.join(pattern_types)}"

        prompt = f"""Analyze the following log entries and identify recurring patterns, anomalies, and trends.
{pattern_focus}

Look for:
- Error patterns and recurring failures
- Performance degradation patterns
- Security-related patterns
- Resource exhaustion patterns
- Network connectivity issues
- Application-specific patterns

For each pattern found, provide:
1. Pattern type and description
2. Confidence level (0.0-1.0)
3. Affected pods/containers
4. Severity level
5. Sample log messages
6. Recommended action

Log Entries ({len(sample_logs)} of {len(log_entries)} total):
"""

        for i, entry in enumerate(sample_logs, 1):
            prompt += f"\n{i}. [{entry.timestamp.strftime('%H:%M:%S')}] "
            prompt += f"[{entry.level or 'INFO'}] "
            prompt += f"[{entry.pod_name}] {entry.message[:150]}"

        return prompt

    def _build_summary_prompt(
        self,
        log_entries: List[LogEntry],
        summary_style: str,
        max_length: int
    ) -> str:
        """Build a prompt for log summarization."""
        max_logs = 200
        sample_logs = log_entries[:max_logs]

        style_instruction = {
            "executive": "Create an executive summary suitable for management",
            "technical": "Create a technical summary for engineers and DevOps teams",
            "brief": "Create a brief, bullet-point summary for a developer to quickly understand if there are any issues"
        }.get(summary_style, "Create a general summary")

        prompt = f"""Summarize the following Kubernetes pod logs in {max_length} words or less.

Style: {style_instruction}

Include:
- Key events and issues
- System health status
- Notable errors or warnings
- Performance observations
- Actionable recommendations

Log Entries ({len(sample_logs)} of {len(log_entries)} total):
"""

        for i, entry in enumerate(sample_logs, 1):
            prompt += f"\n{i}. [{entry.timestamp.strftime('%H:%M:%S')}] "
            prompt += f"[{entry.level or 'INFO'}] "
            prompt += f"[{entry.pod_name}] {entry.message[:100]}"

        return prompt

    def _build_query_prompt(self, log_entries: List[LogEntry], question: str) -> str:
        """Build a prompt for natural language queries."""
        max_logs = 100
        sample_logs = log_entries[:max_logs]

        prompt = f"""Answer the following question about these Kubernetes pod logs:

Question: {question}

Base your answer only on the information present in the logs. If the logs don't contain enough information to answer the question, say so.

Log Entries ({len(sample_logs)} of {len(log_entries)} total):
"""

        for i, entry in enumerate(sample_logs, 1):
            prompt += f"\n{i}. [{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            prompt += f"[{entry.level or 'INFO'}] "
            prompt += f"[{entry.pod_name}/{entry.container_name}] {entry.message}"

        return prompt

    def _parse_analysis_response(
        self,
        response_text: str,
        log_entries: List[LogEntry],
        analysis_type: str
    ) -> AIAnalysisResult:
        """Parse the analysis response into structured results."""
        now = utc_now()
        time_start = min(entry.timestamp for entry in log_entries) if log_entries else now
        time_end = max(entry.timestamp for entry in log_entries) if log_entries else now

        # Extract severity from response (simple heuristic)
        response_lower = response_text.lower()
        if "critical" in response_lower:
            severity = SeverityLevel.CRITICAL
        elif "high" in response_lower:
            severity = SeverityLevel.HIGH
        elif "medium" in response_lower:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW

        # Calculate error rate
        error_count = sum(1 for entry in log_entries if entry.is_error)
        error_rate = error_count / len(log_entries) if log_entries else 0.0

        # Build severity distribution
        severity_dist = {}
        for entry in log_entries:
            level = entry.level.value if entry.level else "UNKNOWN"
            severity_dist[level] = severity_dist.get(level, 0) + 1

        # Extract recommendations (simple parsing)
        recommendations = []
        lines = response_text.split('\n')
        in_recommendations = False
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['recommend', 'action', 'suggestion']):
                in_recommendations = True
            elif in_recommendations and line and not line.startswith('#'):
                recommendations.append(line)
                if len(recommendations) >= 5:  # Limit recommendations
                    break

        return AIAnalysisResult(
            log_entries_analyzed=len(log_entries),
            time_window_start=time_start,
            time_window_end=time_end,
            overall_severity=severity,
            confidence_score=0.8,  # Default confidence
            error_rate=error_rate,
            recommendations=recommendations,
            severity_distribution=severity_dist,
            tags=[analysis_type, "gemini-analysis"]
        )

    def _parse_pattern_response(
        self,
        response_text: str,
        log_entries: List[LogEntry]
    ) -> List[DetectedPattern]:
        """Parse pattern detection response."""
        # This is a simplified parser - in production you might want more sophisticated parsing
        patterns = []

        # Simple heuristic to extract patterns from the response
        lines = response_text.split('\n')
        current_pattern = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for pattern indicators
            if any(word in line.lower() for word in ['pattern', 'error', 'issue', 'problem']):
                if current_pattern:
                    patterns.append(current_pattern)

                from ..core.models import PatternType
                # Create a new pattern
                current_pattern = DetectedPattern(
                    type=PatternType.ERROR_PATTERN,  # Default type
                    pattern=line[:100],  # First 100 chars as pattern description
                    confidence=0.7,  # Default confidence
                    first_seen=log_entries[0].timestamp if log_entries else utc_now(),
                    last_seen=log_entries[-1].timestamp if log_entries else utc_now(),
                    severity=SeverityLevel.MEDIUM,  # Default severity
                )

        if current_pattern:
            patterns.append(current_pattern)

        return patterns

    def _get_time_range(self, log_entries: List[LogEntry]) -> str:
        """Get the time range of log entries."""
        if not log_entries:
            return "No logs"

        start = min(entry.timestamp for entry in log_entries)
        end = max(entry.timestamp for entry in log_entries)
        duration = end - start

        return f"{start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%H:%M:%S')} ({duration})"

    def _get_log_level_distribution(self, log_entries: List[LogEntry]) -> str:
        """Get the distribution of log levels."""
        if not log_entries:
            return "No logs"

        levels = {}
        for entry in log_entries:
            level = entry.level.value if entry.level else "UNKNOWN"
            levels[level] = levels.get(level, 0) + 1

        total = len(log_entries)
        dist_parts = []
        for level, count in sorted(levels.items()):
            percentage = (count / total) * 100
            dist_parts.append(f"{level}: {count} ({percentage:.1f}%)")

        return ", ".join(dist_parts)
