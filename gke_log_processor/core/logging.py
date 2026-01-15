"""Application logging setup and configuration for GKE Log Processor."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from .config import Config, LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors if terminal supports it."""
        # Check if output is a TTY (supports colors)
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, "")
            reset = self.RESET

            # Apply color to the level name
            original_levelname = record.levelname
            record.levelname = f"{color}{record.levelname}{reset}"

            formatted = super().format(record)

            # Restore original level name
            record.levelname = original_levelname
            return formatted
        else:
            return super().format(record)


class StructuredLogger:
    """Centralized logging configuration with structured logging support."""

    _instance: Optional["StructuredLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "StructuredLogger":
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger (only once)."""
        if not self._initialized:
            self._loggers: Dict[str, logging.Logger] = {}
            self._configured: bool = False
            StructuredLogger._initialized = True

    def configure(self, config: Optional[LoggingConfig] = None) -> None:
        """Configure the logging system with the provided configuration.

        Args:
            config: Logging configuration. If None, uses default configuration.
        """
        if self._configured:
            return

        if config is None:
            config = LoggingConfig()

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="ISO"),
                # Use stdlib log processor to route to standard logging
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        log_level = getattr(logging, config.level)
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        if config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)

            # Use colored formatter for console with structlog processor
            console_formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=config.console),
                foreign_pre_chain=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="ISO"),
                ],
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if config.file:
            log_file = Path(config.file).expanduser().resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler to manage log file size
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=config.max_size * 1024 * 1024,  # Convert MB to bytes
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)

            # Use structured formatter for file (JSON format)
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="ISO"),
                ],
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Force immediate flush for testing
            file_handler.flush()

        self._configured = True

    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger instance for the given name.

        Args:
            name: Logger name, typically the module name.

        Returns:
            Configured structured logger instance.
        """
        # Ensure we're configured
        if not self._configured:
            self.configure()

        logger = structlog.get_logger(name)
        # Force the logger to be bound properly
        if hasattr(logger, "bind"):
            return logger
        else:
            # If we get a proxy, bind it to make it a proper BoundLogger
            return structlog.get_logger(name).bind()

    def get_stdlib_logger(self, name: str) -> logging.Logger:
        """Get a standard library logger instance for the given name.

        Args:
            name: Logger name, typically the module name.

        Returns:
            Standard library logger instance.
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

    def set_level(self, level: str) -> None:
        """Dynamically change the logging level.

        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        log_level = getattr(logging, level.upper())

        # Update root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

    def add_context(self, **kwargs) -> None:
        """Add context to all subsequent log messages.

        Args:
            **kwargs: Context key-value pairs to add.
        """
        for key, value in kwargs.items():
            structlog.contextvars.bind_contextvars(**{key: value})

    def clear_context(self) -> None:
        """Clear all bound context variables."""
        structlog.contextvars.clear_contextvars()

    @classmethod
    def setup_from_config(cls, config: Config) -> "StructuredLogger":
        """Set up logging from the main configuration.

        Args:
            config: Main application configuration.

        Returns:
            Configured logger instance.
        """
        logger = cls()
        logger.configure(config.logging)

        # Add some application context
        logger.add_context(
            application="gke-log-processor",
            version="0.1.0",  # TODO: Get from package metadata
        )

        return logger

    def reconfigure(self, config: LoggingConfig) -> None:
        """Reconfigure the logging system with new settings.

        Args:
            config: New logging configuration.
        """
        self._configured = False
        self.configure(config)

    def is_configured(self) -> bool:
        """Check if the logger has been configured.

        Returns:
            True if logger is configured, False otherwise.
        """
        return self._configured


# Global logger instance
_logger_instance: Optional[StructuredLogger] = None


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for the given name.

    This is the primary interface for getting loggers throughout the application.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured structured logger.

    Example:
        >>> from gke_log_processor.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started", version="1.0.0")
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = StructuredLogger()
        # Configure with defaults if not already configured
        if not _logger_instance.is_configured():
            _logger_instance.configure()

    return _logger_instance.get_logger(name)


def get_stdlib_logger(name: str) -> logging.Logger:
    """Get a standard library logger for the given name.

    Use this when you need a standard library logger instead of structlog.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Standard library logger.
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = StructuredLogger()
        if not _logger_instance.is_configured():
            _logger_instance.configure()

    return _logger_instance.get_stdlib_logger(name)


def setup_logging(config: Config) -> StructuredLogger:
    """Set up application logging from configuration.

    This should be called early in application startup.

    Args:
        config: Main application configuration.

    Returns:
        Configured logger instance.

    Example:
        >>> from gke_log_processor.core.config import Config
        >>> from gke_log_processor.core.logging import setup_logging
        >>>
        >>> config = Config.load_from_file("config.yaml")
        >>> logger_manager = setup_logging(config)
        >>>
        >>> # Now you can use loggers throughout the application
        >>> from gke_log_processor.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application configured")
    """
    global _logger_instance
    _logger_instance = StructuredLogger()
    # Force reconfiguration even if already configured
    _logger_instance._configured = False
    _logger_instance.configure(config.logging)

    # Add some application context
    _logger_instance.add_context(
        application="gke-log-processor",
        version="0.1.0",  # TODO: Get from package metadata
    )

    return _logger_instance


def set_log_level(level: str) -> None:
    """Dynamically change the global logging level.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    if _logger_instance is not None:
        _logger_instance.set_level(level)


def add_logging_context(**kwargs) -> None:
    """Add context to all subsequent log messages.

    Args:
        **kwargs: Context key-value pairs to add.

    Example:
        >>> add_logging_context(cluster="my-cluster", namespace="default")
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing logs")  # Will include cluster and namespace
    """
    if _logger_instance is not None:
        _logger_instance.add_context(**kwargs)


def clear_logging_context() -> None:
    """Clear all bound context variables."""
    if _logger_instance is not None:
        _logger_instance.clear_context()


# Convenience logger for this module
logger = get_logger(__name__)
