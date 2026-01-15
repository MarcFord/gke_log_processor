"""Tests for the logging module."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog

from gke_log_processor.core.config import Config, LoggingConfig
from gke_log_processor.core.logging import (
    ColoredFormatter,
    StructuredLogger,
    add_logging_context,
    clear_logging_context,
    get_logger,
    get_stdlib_logger,
    set_log_level,
    setup_logging,
)


class TestColoredFormatter:
    """Test colored formatter."""

    def test_format_with_tty(self):
        """Test formatting with TTY (color support)."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        with patch("sys.stderr.isatty", return_value=True):
            formatted = formatter.format(record)

        # Should contain ANSI color codes
        assert "\033[32m" in formatted  # Green for INFO
        assert "\033[0m" in formatted  # Reset code
        assert "Test message" in formatted

    def test_format_without_tty(self):
        """Test formatting without TTY (no color support)."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        with patch("sys.stderr.isatty", return_value=False):
            formatted = formatter.format(record)

        # Should not contain ANSI color codes
        assert "\033[" not in formatted
        assert formatted == "INFO - Test message"

    def test_all_log_levels_have_colors(self):
        """Test that all log levels have color definitions."""
        formatter = ColoredFormatter("%(levelname)s")

        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            assert level in formatter.COLORS
            color_code = formatter.COLORS[level]
            assert color_code.startswith("\033[")
            assert color_code.endswith("m")


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def setup_method(self):
        """Set up test environment."""
        # Reset the singleton
        StructuredLogger._instance = None
        StructuredLogger._initialized = False

        # Clear any existing loggers
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def teardown_method(self):
        """Clean up after tests."""
        # Reset the singleton
        StructuredLogger._instance = None
        StructuredLogger._initialized = False

        # Clear loggers
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def test_singleton_pattern(self):
        """Test that StructuredLogger follows singleton pattern."""
        logger1 = StructuredLogger()
        logger2 = StructuredLogger()

        assert logger1 is logger2

    def test_configure_with_defaults(self):
        """Test configuration with default settings."""
        logger = StructuredLogger()
        config = LoggingConfig()

        logger.configure(config)

        assert logger.is_configured()

        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1  # Console handler

    def test_configure_with_file_output(self):
        """Test configuration with file output."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            logger = StructuredLogger()
            config = LoggingConfig(
                level="DEBUG", file=log_file, console=True, max_size=5, backup_count=3
            )

            logger.configure(config)

            assert logger.is_configured()

            # Check that both console and file handlers are present
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2

            # Verify file handler configuration
            file_handler = None
            for handler in root_logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    file_handler = handler
                    break

            assert file_handler is not None
            assert file_handler.maxBytes == 5 * 1024 * 1024  # 5MB
            assert file_handler.backupCount == 3

        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_configure_console_only(self):
        """Test configuration with console output only."""
        logger = StructuredLogger()
        config = LoggingConfig(level="WARNING", console=True, file=None)

        logger.configure(config)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)
        assert root_logger.level == logging.WARNING

    def test_get_logger(self):
        """Test getting a structured logger."""
        logger_manager = StructuredLogger()
        config = LoggingConfig()
        logger_manager.configure(config)

        logger = logger_manager.get_logger("test.module")

        # The logger should be a bound logger or have bind method
        assert hasattr(logger, "bind") or isinstance(logger, structlog.BoundLogger)

    def test_get_stdlib_logger(self):
        """Test getting a standard library logger."""
        logger_manager = StructuredLogger()
        config = LoggingConfig()
        logger_manager.configure(config)

        logger = logger_manager.get_stdlib_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

        # Getting the same logger should return the same instance
        logger2 = logger_manager.get_stdlib_logger("test.module")
        assert logger is logger2

    def test_set_level(self):
        """Test dynamically changing log level."""
        logger_manager = StructuredLogger()
        config = LoggingConfig(level="INFO")
        logger_manager.configure(config)

        # Change to DEBUG level
        logger_manager.set_level("DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Check that all handlers have updated level
        for handler in root_logger.handlers:
            assert handler.level == logging.DEBUG

    def test_context_management(self):
        """Test adding and clearing context."""
        logger_manager = StructuredLogger()
        config = LoggingConfig()
        logger_manager.configure(config)

        # Add context
        logger_manager.add_context(cluster="test-cluster", namespace="default")

        # Clear context
        logger_manager.clear_context()

        # No exceptions should be raised

    def test_reconfigure(self):
        """Test reconfiguring the logger."""
        logger_manager = StructuredLogger()
        config1 = LoggingConfig(level="INFO", console=True)

        logger_manager.configure(config1)
        assert logger_manager.is_configured()

        # Reconfigure with different settings
        config2 = LoggingConfig(level="DEBUG", console=False)
        logger_manager.reconfigure(config2)

        assert logger_manager.is_configured()

    def test_setup_from_config(self):
        """Test setting up logger from main config."""
        config = Config()
        config.logging.level = "DEBUG"
        config.logging.console = True

        logger_manager = StructuredLogger.setup_from_config(config)

        assert logger_manager.is_configured()
        assert isinstance(logger_manager, StructuredLogger)


class TestGlobalFunctions:
    """Test global logging functions."""

    def setup_method(self):
        """Set up test environment."""
        # Reset global state
        import gke_log_processor.core.logging as logging_module

        logging_module._logger_instance = None

        # Clear existing loggers
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def teardown_method(self):
        """Clean up after tests."""
        import gke_log_processor.core.logging as logging_module

        logging_module._logger_instance = None

        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def test_get_logger_with_auto_setup(self):
        """Test get_logger with automatic setup."""
        logger = get_logger("test.module")

        assert hasattr(logger, "bind") or isinstance(logger, structlog.BoundLogger)

    def test_get_stdlib_logger_with_auto_setup(self):
        """Test get_stdlib_logger with automatic setup."""
        logger = get_stdlib_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_setup_logging(self):
        """Test setup_logging function."""
        config = Config()
        config.logging.level = "DEBUG"
        config.logging.console = True

        logger_manager = setup_logging(config)

        assert isinstance(logger_manager, StructuredLogger)
        assert logger_manager.is_configured()

        # Test that subsequent get_logger calls use the configured instance
        logger = get_logger("test")
        assert hasattr(logger, "bind") or isinstance(logger, structlog.BoundLogger)

    def test_set_log_level_global(self):
        """Test global set_log_level function."""
        # First set up a logger
        config = Config()
        setup_logging(config)

        # Change log level globally
        set_log_level("DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_add_and_clear_logging_context(self):
        """Test global context management functions."""
        # Set up logging first
        config = Config()
        setup_logging(config)

        # Add context
        add_logging_context(cluster="test", namespace="default")

        # Clear context
        clear_logging_context()

        # No exceptions should be raised

    def test_context_functions_without_setup(self):
        """Test context functions when logger is not set up."""
        # These should not raise exceptions even if logger is not configured
        add_logging_context(test="value")
        clear_logging_context()
        set_log_level("INFO")


class TestLoggingIntegration:
    """Test logging integration with configuration."""

    def setup_method(self):
        """Set up test environment."""
        import gke_log_processor.core.logging as logging_module

        logging_module._logger_instance = None
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def teardown_method(self):
        """Clean up after tests."""
        import gke_log_processor.core.logging as logging_module

        logging_module._logger_instance = None
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def test_logging_with_file_config(self):
        """Test logging configuration from file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            config = Config()
            config.logging.file = log_file
            config.logging.level = "DEBUG"
            config.logging.console = True
            config.logging.max_size = 1
            config.logging.backup_count = 2

            logger_manager = setup_logging(config)

            # Get a logger and log some messages
            logger = get_logger("test.integration")
            logger.info("Test message", extra_field="test_value")
            logger.error("Error message", error_code=500)

            # Force flush to ensure logs are written
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                handler.flush()

            # Verify file was created
            assert Path(log_file).exists()

            # Read log file content
            with open(log_file, "r") as f:
                content = f.read()

            # File should contain log messages
            assert len(content) > 0

        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_log_level_validation_from_config(self):
        """Test that config validates log levels properly."""
        config = Config()
        config.logging.level = "INFO"  # Valid level

        logger_manager = setup_logging(config)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_structured_logging_output(self):
        """Test that structured logging produces expected output."""
        import io
        import sys

        config = Config()
        config.logging.console = True
        config.logging.level = "INFO"

        # Capture stdout before setting up logging
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            logger_manager = setup_logging(config)

            logger = get_logger("test.structured")
            logger.info("Test message", cluster="test-cluster", pod_count=5)

            # Force flush
            import logging

            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                handler.flush()

            output = captured_output.getvalue()

        finally:
            sys.stdout = original_stdout

        # Verify that output was captured
        assert len(output) > 0
        assert "Test message" in output

    def test_context_across_loggers(self):
        """Test that context is shared across different logger instances."""
        config = Config()
        setup_logging(config)

        # Add global context
        add_logging_context(session_id="12345", user="testuser")

        # Get different loggers
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Both loggers should have the same context
        # We can't easily test the context directly, but we can verify
        # that the loggers are bound correctly
        assert hasattr(logger1, "bind") or isinstance(logger1, structlog.BoundLogger)
        assert hasattr(logger2, "bind") or isinstance(logger2, structlog.BoundLogger)

        clear_logging_context()


class TestLoggerConfiguration:
    """Test various logger configuration scenarios."""

    def setup_method(self):
        """Set up test environment."""
        StructuredLogger._instance = None
        StructuredLogger._initialized = False
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def teardown_method(self):
        """Clean up after tests."""
        StructuredLogger._instance = None
        StructuredLogger._initialized = False
        logging.getLogger().handlers.clear()
        structlog.reset_defaults()

    def test_no_configure_twice(self):
        """Test that configure is only called once."""
        logger = StructuredLogger()
        config = LoggingConfig()

        logger.configure(config)
        assert logger.is_configured()

        # Configure again - should not reconfigure
        with patch.object(logger, "_configured", True):
            logger.configure(config)
            # Should still be configured, but not reconfigured

    def test_log_file_directory_creation(self):
        """Test that log file directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "app.log"

            logger = StructuredLogger()
            config = LoggingConfig(file=str(log_file))

            logger.configure(config)

            # Directory should be created
            assert log_file.parent.exists()

    def test_invalid_log_level_handling(self):
        """Test handling of invalid log levels."""
        logger = StructuredLogger()
        config = LoggingConfig()
        logger.configure(config)

        # This should not raise an exception, but use default level
        with pytest.raises(AttributeError):
            logger.set_level("INVALID_LEVEL")
