"""Tests for Gemini AI client."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gke_log_processor.ai.client import (
    GeminiAuthenticationError,
    GeminiClient,
    GeminiConfig,
    GeminiError,
    GeminiRateLimitError,
    RateLimitConfig,
    RequestTracker,
)
from gke_log_processor.core.exceptions import LogProcessingError
from gke_log_processor.core.models import LogEntry, LogLevel, SeverityLevel
from gke_log_processor.core.utils import utc_now


class TestRateLimitConfig:
    """Test RateLimitConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.max_concurrent_requests == 5
        assert config.backoff_factor == 2.0
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            max_concurrent_requests=3,
            backoff_factor=1.5,
            max_retries=2,
        )
        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.max_concurrent_requests == 3
        assert config.backoff_factor == 1.5
        assert config.max_retries == 2

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Test minimum values
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_minute=0)

        with pytest.raises(ValueError):
            RateLimitConfig(max_concurrent_requests=0)

        # Test maximum values
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_minute=1001)

        with pytest.raises(ValueError):
            RateLimitConfig(backoff_factor=11.0)


class TestGeminiConfig:
    """Test GeminiConfig model."""

    def test_required_api_key(self):
        """Test that API key is required."""
        with pytest.raises(ValueError):
            GeminiConfig()

    def test_default_values(self):
        """Test default configuration values."""
        config = GeminiConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.model == "gemini-1.5-pro"
        assert config.temperature == 0.1
        assert config.max_output_tokens == 8192
        assert config.top_p == 0.95
        assert config.top_k == 64
        assert isinstance(config.rate_limit, RateLimitConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        rate_limit = RateLimitConfig(requests_per_minute=30)
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-pro",
            temperature=0.5,
            max_output_tokens=4096,
            top_p=0.8,
            top_k=32,
            rate_limit=rate_limit,
        )
        assert config.model == "gemini-pro"
        assert config.temperature == 0.5
        assert config.max_output_tokens == 4096
        assert config.top_p == 0.8
        assert config.top_k == 32
        assert config.rate_limit.requests_per_minute == 30

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Test temperature bounds
        with pytest.raises(ValueError):
            GeminiConfig(api_key="test", temperature=-0.1)
        with pytest.raises(ValueError):
            GeminiConfig(api_key="test", temperature=2.1)

        # Test max_output_tokens bounds
        with pytest.raises(ValueError):
            GeminiConfig(api_key="test", max_output_tokens=0)
        with pytest.raises(ValueError):
            GeminiConfig(api_key="test", max_output_tokens=32769)

    @patch("google.generativeai.configure")
    def test_configure_genai(self, mock_configure):
        """Test GenAI configuration."""
        config = GeminiConfig(api_key="test-api-key")
        config.configure_genai()
        mock_configure.assert_called_once_with(api_key="test-api-key")


class TestRequestTracker:
    """Test RequestTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a request tracker for testing."""
        config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=20,
            max_concurrent_requests=2
        )
        return RequestTracker(config)

    @pytest.mark.asyncio
    async def test_initial_state(self, tracker):
        """Test initial tracker state."""
        assert await tracker.can_make_request() is True
        assert tracker.active_requests == 0
        assert len(tracker.requests_per_minute) == 0
        assert len(tracker.requests_per_hour) == 0

    @pytest.mark.asyncio
    async def test_request_lifecycle(self, tracker):
        """Test request start and end tracking."""
        # Start a request
        await tracker.record_request_start()
        assert tracker.active_requests == 1
        assert len(tracker.requests_per_minute) == 1
        assert len(tracker.requests_per_hour) == 1

        # End the request
        await tracker.record_request_end()
        assert tracker.active_requests == 0
        assert len(tracker.requests_per_minute) == 1  # History preserved
        assert len(tracker.requests_per_hour) == 1

    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self, tracker):
        """Test concurrent request limiting."""
        # Start maximum concurrent requests
        await tracker.record_request_start()
        await tracker.record_request_start()
        assert tracker.active_requests == 2

        # Should not be able to make another request
        assert await tracker.can_make_request() is False

        # End one request
        await tracker.record_request_end()
        assert await tracker.can_make_request() is True

    @pytest.mark.asyncio
    async def test_minute_rate_limit(self, tracker):
        """Test per-minute rate limiting."""
        # Fill up the minute quota
        for _ in range(5):
            await tracker.record_request_start()
            await tracker.record_request_end()

        # Should not be able to make another request
        assert await tracker.can_make_request() is False

    @pytest.mark.asyncio
    async def test_old_request_cleanup(self, tracker):
        """Test cleanup of old request timestamps."""
        # Add old requests manually
        old_time = utc_now() - timedelta(minutes=2)
        tracker.requests_per_minute.append(old_time)
        tracker.requests_per_hour.append(old_time)

        # Check can make request (should clean up old ones)
        can_make = await tracker.can_make_request()
        assert can_make is True
        assert len(tracker.requests_per_minute) == 0

    @pytest.mark.asyncio
    async def test_wait_for_rate_limit(self, tracker):
        """Test waiting for rate limit availability."""
        # Fill up concurrent requests
        await tracker.record_request_start()
        await tracker.record_request_start()
        assert tracker.active_requests == 2

        # Start waiting task
        wait_task = asyncio.create_task(tracker.wait_for_rate_limit())

        # Let it start waiting
        await asyncio.sleep(0.01)

        # Free up a slot
        await tracker.record_request_end()

        # Should complete waiting
        wait_time = await wait_task
        assert wait_time >= 0


class TestGeminiClient:
    """Test GeminiClient functionality."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return GeminiConfig(
            api_key="test-api-key",
            rate_limit=RateLimitConfig(max_retries=1)  # Faster testing
        )

    @pytest.fixture
    def sample_log_entries(self):
        """Create sample log entries for testing."""
        now = utc_now()
        return [
            LogEntry(
                timestamp=now - timedelta(seconds=30),
                message="Application started successfully",
                level=LogLevel.INFO,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Application started successfully",
            ),
            LogEntry(
                timestamp=now - timedelta(seconds=20),
                message="Connection refused to database",
                level=LogLevel.ERROR,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Connection refused to database",
            ),
            LogEntry(
                timestamp=now - timedelta(seconds=10),
                message="Retrying database connection",
                level=LogLevel.WARNING,
                source="app-container",
                pod_name="app-pod-1",
                namespace="default",
                cluster="test-cluster",
                container_name="app-container",
                raw_message="Retrying database connection",
            ),
        ]

    @patch("google.generativeai.configure")
    def test_client_initialization(self, mock_configure, config):
        """Test client initialization."""
        client = GeminiClient(config)
        assert client.config == config
        assert client.request_tracker is not None
        assert client._model is None
        mock_configure.assert_called_once_with(api_key="test-api-key")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_model_property(self, mock_model_class, mock_configure, config):
        """Test model property lazy loading."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        client = GeminiClient(config)

        # First access creates model
        model = client.model
        assert model == mock_model
        mock_model_class.assert_called_once()

        # Second access returns same model
        model2 = client.model
        assert model2 == mock_model
        assert mock_model_class.call_count == 1

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_configure, config):
        """Test successful connection test."""
        with patch.object(GeminiClient, "_make_request_with_retry", return_value="OK"):
            client = GeminiClient(config)
            result = await client.test_connection()
            assert result is True

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, mock_configure, config):
        """Test failed connection test."""
        with patch.object(GeminiClient, "_make_request_with_retry", side_effect=Exception("Connection failed")):
            client = GeminiClient(config)
            result = await client.test_connection()
            assert result is False

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_analyze_logs_empty_list(self, mock_configure, config):
        """Test analyzing empty log list."""
        client = GeminiClient(config)
        with pytest.raises(GeminiError, match="No log entries provided"):
            await client.analyze_logs([])

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_analyze_logs_success(self, mock_configure, config, sample_log_entries):
        """Test successful log analysis."""
        mock_response = "Analysis: CRITICAL severity issues detected. Critical database connectivity problems found."

        with patch.object(GeminiClient, "_make_request_with_retry", return_value=mock_response):
            client = GeminiClient(config)
            result = await client.analyze_logs(sample_log_entries)

            assert result.log_entries_analyzed == len(sample_log_entries)
            assert result.overall_severity == SeverityLevel.CRITICAL
            assert result.confidence_score == 0.8

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_detect_patterns(self, mock_configure, config, sample_log_entries):
        """Test pattern detection."""
        mock_response = "Pattern detected: Database connection errors occurring repeatedly."

        with patch.object(GeminiClient, "_make_request_with_retry", return_value=mock_response):
            client = GeminiClient(config)
            patterns = await client.detect_patterns(sample_log_entries)

            assert isinstance(patterns, list)

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_summarize_logs(self, mock_configure, config, sample_log_entries):
        """Test log summarization."""
        mock_response = "Summary: Application experiencing database connectivity issues. Requires immediate attention."

        with patch.object(GeminiClient, "_make_request_with_retry", return_value=mock_response):
            client = GeminiClient(config)
            summary = await client.summarize_logs(sample_log_entries, "executive", 100)

            assert isinstance(summary, str)
            assert len(summary) > 0

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_query_logs(self, mock_configure, config, sample_log_entries):
        """Test natural language queries."""
        mock_response = "Based on the logs, there are database connection issues affecting the application."

        with patch.object(GeminiClient, "_make_request_with_retry", return_value=mock_response):
            client = GeminiClient(config)
            answer = await client.query_logs(sample_log_entries, "What issues are affecting the application?")

            assert isinstance(answer, str)
            assert len(answer) > 0

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_make_request_with_retry_success(self, mock_configure, config):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.text = "Test response"

        with patch.object(GeminiClient, "_make_api_call", return_value="Test response"):
            client = GeminiClient(config)
            result = await client._make_request_with_retry("test prompt")
            assert result == "Test response"

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_make_request_with_retry_rate_limit(self, mock_configure, config):
        """Test rate limit handling."""
        with patch.object(GeminiClient, "_make_api_call", side_effect=Exception("Rate limit exceeded")):
            client = GeminiClient(config)
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await client._make_request_with_retry("test prompt")

    @patch("google.generativeai.configure")
    @pytest.mark.asyncio
    async def test_make_request_with_retry_auth_error(self, mock_configure, config):
        """Test authentication error handling."""
        with patch.object(GeminiClient, "_make_api_call", side_effect=Exception("Authentication failed")):
            client = GeminiClient(config)
            with pytest.raises(GeminiAuthenticationError):
                await client._make_request_with_retry("test prompt")

    @patch("google.generativeai.configure")
    def test_build_analysis_prompt(self, mock_configure, config, sample_log_entries):
        """Test analysis prompt building."""
        client = GeminiClient(config)
        prompt = client._build_analysis_prompt(sample_log_entries, "comprehensive")

        assert "Analysis Type: comprehensive" in prompt
        assert "Total Log Entries: 3" in prompt
        assert "Application started successfully" in prompt
        assert "Connection refused to database" in prompt

    @patch("google.generativeai.configure")
    def test_parse_analysis_response(self, mock_configure, config, sample_log_entries):
        """Test parsing of analysis response."""
        client = GeminiClient(config)
        response = "The analysis shows CRITICAL issues with database connectivity. Immediate action recommended: Check database connection settings."

        result = client._parse_analysis_response(response, sample_log_entries, "comprehensive")

        assert result.overall_severity == SeverityLevel.CRITICAL
        assert result.log_entries_analyzed == len(sample_log_entries)
        # Note: The simple parsing logic generates basic recommendations
        assert result.needs_immediate_attention == True

    @patch("google.generativeai.configure")
    def test_get_time_range(self, mock_configure, config, sample_log_entries):
        """Test time range calculation."""
        client = GeminiClient(config)
        time_range = client._get_time_range(sample_log_entries)

        assert isinstance(time_range, str)
        assert "to" in time_range

    @patch("google.generativeai.configure")
    def test_get_log_level_distribution(self, mock_configure, config, sample_log_entries):
        """Test log level distribution calculation."""
        client = GeminiClient(config)
        distribution = client._get_log_level_distribution(sample_log_entries)

        assert isinstance(distribution, str)
        assert "INFO:" in distribution
        assert "ERROR:" in distribution
        assert "WARNING:" in distribution


class TestGeminiErrors:
    """Test custom Gemini exceptions."""

    def test_gemini_error(self):
        """Test base GeminiError."""
        error = GeminiError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, LogProcessingError)

    def test_gemini_authentication_error(self):
        """Test GeminiAuthenticationError."""
        error = GeminiAuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, GeminiError)

    def test_gemini_rate_limit_error(self):
        """Test GeminiRateLimitError."""
        error = GeminiRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, GeminiError)
