"""Core configuration and settings for GKE Log Processor."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClusterConfig(BaseModel):
    """Configuration for a single GKE cluster."""

    name: str = Field(..., description="GKE cluster name")
    project_id: str = Field(..., description="GCP project ID")
    zone: Optional[str] = Field(None, description="GKE cluster zone")
    region: Optional[str] = Field(None, description="GKE cluster region")
    namespace: str = Field("default", description="Kubernetes namespace")

    @field_validator('zone', 'region')
    @classmethod
    def validate_location(cls, v, info):
        """Validate that either zone or region is provided, but not both."""
        data = info.data if hasattr(info, 'data') else {}
        zone = data.get(
            'zone') if 'zone' in data else v if info.field_name == 'zone' else None
        region = data.get(
            'region') if 'region' in data else v if info.field_name == 'region' else None

        if not zone and not region:
            raise ValueError("Either zone or region must be specified")
        if zone and region:
            raise ValueError("Cannot specify both zone and region")
        return v

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
        0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for AI responses")
    max_tokens: int = Field(
        1000, gt=0, description="Maximum tokens for AI responses")


class UIConfig(BaseModel):
    """Configuration for the user interface."""

    theme: str = Field("dark", description="UI theme (dark/light)")
    refresh_rate: int = Field(
        1000, gt=0, description="Refresh rate in milliseconds")
    max_log_lines: int = Field(
        1000, gt=0, description="Maximum log lines to display")


class Config(BaseSettings):
    """Main configuration for GKE Log Processor."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    # CLI arguments (these will be set programmatically)
    cluster_name: Optional[str] = Field(None, description="GKE cluster name")
    project_id: Optional[str] = Field(None, description="GCP project ID")
    zone: Optional[str] = Field(None, description="GKE cluster zone")
    region: Optional[str] = Field(None, description="GKE cluster region")
    namespace: str = Field("default", description="Kubernetes namespace")
    config_file: Optional[str] = Field(
        None, description="Path to configuration file")
    verbose: bool = Field(False, description="Enable verbose logging")

    # Configuration from file/environment
    clusters: List[ClusterConfig] = Field(
        default_factory=list,
        description="Predefined clusters")
    gemini: GeminiConfig = Field(
        default_factory=lambda: GeminiConfig(),  # type: ignore[arg-type, call-arg]
        description="Gemini AI configuration")
    ui: UIConfig = Field(
        default_factory=lambda: UIConfig(),  # type: ignore[arg-type, call-arg]
        description="UI configuration")

    # Environment variables (using pydantic-settings automatic env detection)
    gemini_api_key: Optional[str] = Field(
        default=None, description="Gemini AI API key")
    google_application_credentials: Optional[str] = Field(
        default=None, description="GCP service account key path"
    )

    @field_validator('cluster_name', 'project_id')
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
            namespace=self.namespace
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Get the effective Gemini API key from various sources."""
        return self.gemini.api_key or self.gemini_api_key

    @classmethod
    def load_from_file(cls, config_path: str) -> "Config":
        """Load configuration from a YAML file."""
        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        return cls(**config_data)

    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to a YAML file."""
        import yaml

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Export only the configuration sections, not CLI arguments
        export_data = {
            "clusters": [cluster.model_dump() for cluster in self.clusters],
            "gemini": self.gemini.model_dump(),
            "ui": self.ui.model_dump()
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(export_data, f, default_flow_style=False)
