"""Core configuration and settings for GKE Log Processor."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClusterConfig(BaseModel):
    """Configuration for a single GKE cluster."""

    name: str = Field(..., description="GKE cluster name")
    project_id: str = Field(..., description="GCP project ID")
    zone: Optional[str] = Field(None, description="GKE cluster zone")
    region: Optional[str] = Field(None, description="GKE cluster region")
    namespace: str = Field("default", description="Kubernetes namespace")

    @model_validator(mode="after")
    def validate_location(self):
        """Validate that either zone or region is provided, but not both."""
        if not self.zone and not self.region:
            raise ValueError("Either zone or region must be specified")
        if self.zone and self.region:
            raise ValueError("Cannot specify both zone and region")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def location(self) -> str:
        """Get the location (zone or region) for the cluster."""
        return self.zone or self.region or ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_regional(self) -> bool:
        """Check if this is a regional cluster."""
        return bool(self.region)


class GeminiConfig(BaseModel):
    """Configuration for Gemini AI integration."""

    api_key: Optional[str] = Field(None, description="Gemini AI API key")
    model: str = Field("gemini-pro", description="Gemini model to use")
    temperature: float = Field(
        0.1, ge=0.0, le=2.0, description="Temperature for AI responses"
    )
    max_tokens: int = Field(1000, gt=0, description="Maximum tokens for AI responses")


class LoggingConfig(BaseModel):
    """Configuration for application logging."""

    level: str = Field(
        "INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file: Optional[str] = Field(
        None, description="Log file path (if None, logs to console only)"
    )
    max_size: int = Field(10, description="Maximum log file size in MB")
    backup_count: int = Field(5, description="Number of backup log files to keep")
    console: bool = Field(True, description="Enable console logging")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()


class StreamingConfig(BaseModel):
    """Configuration for log streaming."""

    max_buffer_size: int = Field(
        1000, gt=0, description="Maximum buffer size for log entries"
    )
    buffer_flush_interval: float = Field(
        1.0, gt=0, description="Buffer flush interval in seconds"
    )
    max_logs_per_second: float = Field(
        100.0, gt=0, description="Maximum logs per second rate limit"
    )
    follow_logs: bool = Field(True, description="Follow new logs in real-time")
    tail_lines: int = Field(
        100, ge=0, description="Number of recent log lines to fetch initially"
    )
    timestamps: bool = Field(True, description="Include timestamps in log parsing")


class UIConfig(BaseModel):
    """Configuration for the user interface."""

    theme: str = Field("dark", description="UI theme (dark/light)")
    refresh_rate: int = Field(1000, gt=0, description="Refresh rate in milliseconds")
    max_log_lines: int = Field(1000, gt=0, description="Maximum log lines to display")
    show_timestamps: bool = Field(True, description="Show timestamps in UI")
    auto_scroll: bool = Field(True, description="Auto-scroll to new log entries")

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v):
        """Validate UI theme."""
        if v.lower() not in ["dark", "light"]:
            raise ValueError("Theme must be 'dark' or 'light'")
        return v.lower()


class Config(BaseSettings):
    """Main configuration for GKE Log Processor."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # CLI arguments (these will be set programmatically)
    cluster_name: Optional[str] = Field(None, description="GKE cluster name")
    project_id: Optional[str] = Field(None, description="GCP project ID")
    zone: Optional[str] = Field(None, description="GKE cluster zone")
    region: Optional[str] = Field(None, description="GKE cluster region")
    namespace: str = Field("default", description="Kubernetes namespace")
    config_file: Optional[str] = Field(None, description="Path to configuration file")
    verbose: bool = Field(False, description="Enable verbose logging")

    # Configuration from file/environment
    clusters: List[ClusterConfig] = Field(
        default_factory=list, description="Predefined clusters"
    )
    gemini: GeminiConfig = Field(
        default_factory=lambda: GeminiConfig(),  # type: ignore[arg-type, call-arg]
        description="Gemini AI configuration",
    )
    ui: UIConfig = Field(
        default_factory=lambda: UIConfig(),  # type: ignore[arg-type, call-arg]
        description="UI configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(),  # type: ignore[arg-type, call-arg]
        description="Logging configuration",
    )
    streaming: StreamingConfig = Field(
        default_factory=lambda: StreamingConfig(),  # type: ignore[arg-type, call-arg]
        description="Log streaming configuration",
    )

    # Environment variables (using pydantic-settings automatic env detection)
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini AI API key")
    google_application_credentials: Optional[str] = Field(
        default=None, description="GCP service account key path"
    )

    # Configuration file paths to search
    config_search_paths: List[str] = Field(
        default_factory=lambda: [
            "./gke-logs.yaml",
            "./gke-logs.yml",
            "~/.config/gke-logs/config.yaml",
            "~/.config/gke-logs/config.yml",
            "/etc/gke-logs/config.yaml",
            "/etc/gke-logs/config.yml",
        ],
        description="Paths to search for configuration files",
    )

    @field_validator("cluster_name", "project_id")
    @classmethod
    def validate_required_if_set(cls, v):
        """Validate required fields if they are set."""
        if v is not None and not v.strip():
            raise ValueError("Cannot be empty if provided")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_cluster(self) -> Optional[ClusterConfig]:
        """Get the current cluster configuration."""
        if not self.cluster_name or not self.project_id:
            return None

        return ClusterConfig(
            name=self.cluster_name,
            project_id=self.project_id,
            zone=self.zone,
            region=self.region,
            namespace=self.namespace,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Get the effective Gemini API key from various sources."""
        return self.gemini.api_key or self.gemini_api_key

    @staticmethod
    def _expand_environment_variables(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively expand environment variables in configuration data."""
        if isinstance(config_data, dict):
            return {
                key: Config._expand_environment_variables(value)
                for key, value in config_data.items()
            }
        elif isinstance(config_data, list):
            return [Config._expand_environment_variables(item) for item in config_data]
        elif isinstance(config_data, str):
            return os.path.expandvars(config_data)
        else:
            return config_data

    @classmethod
    def _load_yaml_file(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse YAML configuration with environment variable expansion."""
        config_file = Path(config_path).expanduser().resolve()

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        if not config_file.is_file():
            raise ValueError(f"Configuration path is not a file: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in configuration file {config_file}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error reading configuration file {config_file}: {e}"
            ) from e

        # Expand environment variables
        config_data = cls._expand_environment_variables(config_data)

        return config_data

    @classmethod
    def find_config_file(
        cls, search_paths: Optional[List[str]] = None
    ) -> Optional[Path]:
        """Find the first existing configuration file in the search paths."""
        if search_paths is None:
            # Use default search paths
            search_paths = [
                "./gke-logs.yaml",
                "./gke-logs.yml",
                "~/.config/gke-logs/config.yaml",
                "~/.config/gke-logs/config.yml",
                "/etc/gke-logs/config.yaml",
                "/etc/gke-logs/config.yml",
            ]

        for path_str in search_paths:
            config_path = Path(path_str).expanduser().resolve()
            if config_path.exists() and config_path.is_file():
                return config_path

        return None

    @classmethod
    def load_from_file(
        cls,
        config_path: Optional[str] = None,
        search_paths: Optional[List[str]] = None,
    ) -> "Config":
        """Load configuration from a YAML file.

        Args:
            config_path: Explicit path to config file. If None, will search.
            search_paths: Custom search paths if config_path is None.

        Returns:
            Config instance with loaded configuration.

        Raises:
            FileNotFoundError: If config file not found.
            ValueError: If config file contains invalid YAML or data.
            ValidationError: If config data doesn't match schema.
        """
        if config_path:
            config_data = cls._load_yaml_file(config_path)
        else:
            config_file = cls.find_config_file(search_paths)
            if not config_file:
                # Return default config if no file found
                return cls()  # type: ignore[call-arg]
            config_data = cls._load_yaml_file(config_file)

        try:
            return cls(**config_data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    @classmethod
    def load_with_overrides(
        cls,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        """Load configuration with optional overrides.

        Args:
            config_path: Path to configuration file.
            overrides: Dictionary of values to override in the config.

        Returns:
            Config instance with loaded and overridden configuration.
        """
        config = cls.load_from_file(config_path)

        if overrides:
            # Apply overrides
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def save_to_file(self, config_path: str, include_cli_args: bool = False) -> None:
        """Save current configuration to a YAML file.

        Args:
            config_path: Path where to save the configuration file.
            include_cli_args: Whether to include CLI arguments in the saved config.
        """
        config_file = Path(config_path).expanduser().resolve()
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Export configuration sections
        export_data = {
            "clusters": [cluster.model_dump() for cluster in self.clusters],
            "gemini": self.gemini.model_dump(),
            "ui": self.ui.model_dump(),
            "logging": self.logging.model_dump(),
            "streaming": self.streaming.model_dump(),
        }

        # Optionally include CLI arguments
        if include_cli_args:
            cli_data = {
                "cluster_name": self.cluster_name,
                "project_id": self.project_id,
                "zone": self.zone,
                "region": self.region,
                "namespace": self.namespace,
                "verbose": self.verbose,
            }
            # Only include non-None values
            cli_data = {k: v for k, v in cli_data.items() if v is not None}
            if cli_data:
                export_data["cli_defaults"] = cli_data

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    export_data, f, default_flow_style=False, sort_keys=False, indent=2
                )
        except Exception as e:
            raise RuntimeError(
                f"Error writing configuration file {config_file}: {e}"
            ) from e

    @classmethod
    def create_template(cls, config_path: str) -> None:
        """Create a template configuration file with documentation.

        Args:
            config_path: Path where to create the template file.
        """
        config_file = Path(config_path).expanduser().resolve()
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a custom YAML representation that includes comments
        yaml_lines = []
        yaml_lines.append("# GKE Log Processor Configuration")
        yaml_lines.append(
            "# This file supports environment variable expansion using "
            "$VAR or ${VAR} syntax"
        )
        yaml_lines.append("")

        # Add clusters section with comments
        yaml_lines.append("# Predefined GKE clusters")
        yaml_lines.append("clusters:")
        yaml_lines.append("  - name: my-cluster")
        yaml_lines.append("    project_id: ${GCP_PROJECT_ID}")
        yaml_lines.append("    # Environment variable expansion example above")
        yaml_lines.append("    zone: us-central1-a")
        yaml_lines.append("    # region: us-central1  # Use region instead of zone")
        yaml_lines.append("    # for regional clusters")
        yaml_lines.append("    namespace: default")
        yaml_lines.append("")

        # Add other sections
        sections = {
            "gemini": "# Gemini AI configuration",
            "ui": "# User interface configuration",
            "logging": "# Application logging configuration",
            "streaming": "# Log streaming configuration",
        }

        config_obj = cls()  # type: ignore[call-arg]
        for section_name, comment in sections.items():
            yaml_lines.append(comment)
            section_data = getattr(config_obj, section_name).model_dump()

            # Add environment variable examples for specific fields
            if section_name == "gemini" and "api_key" in section_data:
                section_data["api_key"] = "${GEMINI_API_KEY}"

            section_yaml = yaml.safe_dump(
                {section_name: section_data}, default_flow_style=False, indent=2
            )
            yaml_lines.extend(section_yaml.strip().split("\n"))
            yaml_lines.append("")

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                f.write("\n".join(yaml_lines))
        except Exception as e:
            raise RuntimeError(
                f"Error creating template file {config_file}: {e}"
            ) from e

    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any warnings.

        Returns:
            List of validation warning messages.
        """
        warnings = []

        # Check if we have any clusters defined
        if not self.clusters and not self.current_cluster:
            warnings.append("No clusters configured")

        # Check Gemini API key
        if not self.effective_gemini_api_key:
            warnings.append(
                "Gemini API key not configured - AI features will be disabled"
            )

        # Check Google Cloud credentials
        if not self.google_application_credentials and not os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            warnings.append("Google Cloud credentials not configured")

        # Check log file path if specified
        if self.logging.file:
            log_path = Path(self.logging.file).expanduser().resolve()
            if not log_path.parent.exists():
                warnings.append(f"Log directory does not exist: {log_path.parent}")

        return warnings
