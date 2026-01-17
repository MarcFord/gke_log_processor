"""Core configuration and settings for GKE Log Processor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

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
    namespace: str = Field("default", description="Default namespace for the cluster")
    description: Optional[str] = Field(None, description="Human friendly description")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metadata labels")

    @model_validator(mode="after")
    def validate_location(self) -> "ClusterConfig":
        """Ensure exactly one of zone or region is provided."""
        if not self.zone and not self.region:
            raise ValueError("Either zone or region must be specified")
        if self.zone and self.region:
            raise ValueError("Cannot specify both zone and region")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def location(self) -> str:
        """Return the zone or region for the cluster."""
        return self.zone or self.region or ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_regional(self) -> bool:
        """Determine whether this cluster is regional."""
        return bool(self.region)


class GKEConfig(BaseModel):
    """Configuration for the active GKE connection."""

    cluster_name: Optional[str] = Field(None, description="Active cluster name")
    project_id: Optional[str] = Field(None, description="Active GCP project")
    zone: Optional[str] = Field(None, description="Active cluster zone")
    region: Optional[str] = Field(None, description="Active cluster region")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def location(self) -> Optional[str]:
        """Return the active location if available."""
        return self.zone or self.region

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_regional(self) -> bool:
        """Check if the active configuration is regional."""
        return bool(self.region)

    def apply_cluster(self, cluster: ClusterConfig) -> None:
        """Copy values from a saved cluster definition."""
        self.cluster_name = cluster.name
        self.project_id = cluster.project_id
        self.zone = cluster.zone
        self.region = cluster.region


class KubernetesConfig(BaseModel):
    """Kubernetes API configuration."""

    default_namespace: str = Field("default", description="Default namespace")
    context: Optional[str] = Field(None, description="Local kubeconfig context override")
    request_timeout_seconds: int = Field(
        30, ge=1, description="Timeout for Kubernetes API calls"
    )


class AIConfig(BaseModel):
    """Gemini AI integration configuration."""

    gemini_api_key: Optional[str] = Field(None, description="Gemini AI API key")
    model_name: str = Field("gemini-3-flash-preview", description="Gemini model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2048, gt=0, description="Maximum response tokens")
    analysis_enabled: bool = Field(True, description="Enable AI analysis features")
    query_timeout_seconds: int = Field(30, ge=1, description="AI query timeout (seconds)")

    @property
    def api_key(self) -> Optional[str]:
        """Alias for backwards compatibility."""
        return self.gemini_api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self.gemini_api_key = value


class LoggingConfig(BaseModel):
    """Application logging configuration."""

    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log formatting string",
    )
    file: Optional[str] = Field(None, description="Log file path if file logging is enabled")
    max_size: int = Field(10, ge=1, description="Maximum log file size in MB")
    backup_count: int = Field(5, ge=0, description="Number of rotated log files")
    console: bool = Field(True, description="Enable console logging output")
    structured: bool = Field(True, description="Emit structured JSON logs for files")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure provided log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(sorted(valid_levels))}")
        return upper

    @property
    def console_enabled(self) -> bool:
        """Mirror property for UI compatibility."""
        return self.console

    @console_enabled.setter
    def console_enabled(self, value: bool) -> None:
        self.console = value

    @property
    def file_enabled(self) -> bool:
        """Whether file logging is enabled."""
        return bool(self.file)

    @file_enabled.setter
    def file_enabled(self, value: bool) -> None:
        if not value:
            self.file = None
        elif value and not self.file:
            # Default file path when toggled on via UI
            self.file = "./gke-log-processor.log"

    @property
    def file_path(self) -> str:
        """Expose file path for UI editing."""
        return self.file or ""

    @file_path.setter
    def file_path(self, value: str) -> None:
        self.file = value.strip() or None

    @property
    def max_file_size_mb(self) -> int:
        """Expose size field with UI-friendly naming."""
        return self.max_size

    @max_file_size_mb.setter
    def max_file_size_mb(self, value: int) -> None:
        self.max_size = value


class StreamingConfig(BaseModel):
    """Configuration for log streaming."""

    max_buffer_size: int = Field(1000, ge=1, description="Max buffered log entries")
    buffer_flush_interval: float = Field(
        1.0, gt=0, description="How frequently to flush buffers"
    )
    max_logs_per_second: float = Field(
        100.0, gt=0, description="Rate limit for incoming logs"
    )
    follow_logs: bool = Field(True, description="Follow logs in real-time")
    tail_lines: int = Field(100, ge=0, description="Initial tail length when streaming")
    timestamps: bool = Field(True, description="Include timestamps in parsed output")


class UIConfig(BaseModel):
    """Configuration for the Textual UI."""

    theme: str = Field("dark", description="UI theme (dark or light)")
    refresh_rate: int = Field(1000, ge=100, description="Refresh rate in milliseconds")
    max_log_lines: int = Field(1000, ge=100, description="Maximum lines to keep in viewer")
    show_timestamps: bool = Field(True, description="Display timestamps in UI")
    auto_scroll: bool = Field(True, description="Auto-scroll to new log entries")

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, value: str) -> str:
        """Ensure theme string is supported."""
        lowered = value.lower()
        if lowered not in {"dark", "light"}:
            raise ValueError("Theme must be 'dark' or 'light'")
        return lowered


class ProfileConfig(BaseModel):
    """Saved configuration profile referencing a cluster."""

    name: str = Field(..., description="Profile name")
    cluster: str = Field(..., description="Name of cluster this profile uses")
    namespace: Optional[str] = Field(None, description="Namespace override for profile")
    description: Optional[str] = Field(None, description="Profile description")

    def apply_to(self, config: "Config") -> None:
        """Apply this profile to a configuration instance."""
        cluster = config.get_cluster(self.cluster)
        if cluster is None:
            raise ValueError(
                f"Cluster '{self.cluster}' referenced by profile '{self.name}' was not found"
            )

        config.set_active_cluster(cluster.name)
        if self.namespace:
            config.kubernetes.default_namespace = self.namespace
        else:
            config.kubernetes.default_namespace = cluster.namespace


class ConfigTemplate(BaseModel):
    """Reusable template describing clusters, profiles, and defaults."""

    name: str
    description: str
    clusters: List[ClusterConfig] = Field(default_factory=list)
    profiles: List[ProfileConfig] = Field(default_factory=list)
    default_profile: Optional[str] = Field(
        None, description="Name of profile to activate by default"
    )
    logging: Optional[LoggingConfig] = None
    streaming: Optional[StreamingConfig] = None
    ai: Optional[AIConfig] = None
    kubernetes: Optional[KubernetesConfig] = None
    ui: Optional[UIConfig] = None

    def summary_lines(self) -> List[str]:
        """Render a short textual summary for UI display."""
        lines = [self.description.strip()]
        if self.clusters:
            lines.append("Clusters: " + ", ".join(c.name for c in self.clusters))
        if self.profiles:
            lines.append("Profiles: " + ", ".join(p.name for p in self.profiles))
        if self.default_profile:
            lines.append(f"Default profile: {self.default_profile}")
        return lines


BUILTIN_TEMPLATES: Dict[str, ConfigTemplate] = {
    "development": ConfigTemplate(
        name="development",
        description="Single-cluster development environment with verbose logging and rapid refresh settings.",
        clusters=[
            ClusterConfig(
                name="dev-cluster",
                project_id="${GCP_PROJECT_ID}",
                zone="us-central1-a",
                namespace="development",
                description="Default development cluster",
            )
        ],
        profiles=[
            ProfileConfig(
                name="dev-default",
                cluster="dev-cluster",
                namespace="development",
                description="Default development profile",
            )
        ],
        default_profile="dev-default",
        logging=LoggingConfig(level="DEBUG", console=True, structured=False),
        streaming=StreamingConfig(max_logs_per_second=200.0, tail_lines=200),
        kubernetes=KubernetesConfig(default_namespace="development"),
    ),
    "staging": ConfigTemplate(
        name="staging",
        description="Two-cluster staging environment with separate namespaces for canary testing.",
        clusters=[
            ClusterConfig(
                name="staging-canary",
                project_id="${GCP_PROJECT_ID}",
                zone="us-west1-a",
                namespace="canary",
                description="Canary testing cluster",
            ),
            ClusterConfig(
                name="staging-stable",
                project_id="${GCP_PROJECT_ID}",
                region="us-west1",
                namespace="stable",
                description="Stable staging cluster",
            ),
        ],
        profiles=[
            ProfileConfig(
                name="staging-canary",
                cluster="staging-canary",
                namespace="canary",
                description="Route to canary namespace for smoke tests",
            ),
            ProfileConfig(
                name="staging-stable",
                cluster="staging-stable",
                namespace="stable",
                description="Stable staging environment",
            ),
        ],
        default_profile="staging-stable",
        logging=LoggingConfig(level="INFO", console=True, structured=True),
        streaming=StreamingConfig(max_logs_per_second=120.0, tail_lines=150),
    ),
    "production": ConfigTemplate(
        name="production",
        description="Highly controlled production template with locked-down logging and streaming defaults.",
        clusters=[
            ClusterConfig(
                name="prod-primary",
                project_id="${GCP_PROJECT_ID}",
                region="us-east1",
                namespace="prod",
                description="Primary production cluster",
            )
        ],
        profiles=[
            ProfileConfig(
                name="production",
                cluster="prod-primary",
                namespace="prod",
                description="Primary production operations profile",
            )
        ],
        default_profile="production",
        logging=LoggingConfig(level="WARNING", console=True, structured=True, file="/var/log/gke-log-processor.log"),
        streaming=StreamingConfig(max_logs_per_second=80.0, tail_lines=100, follow_logs=True),
        kubernetes=KubernetesConfig(default_namespace="prod"),
    ),
}


def get_builtin_templates() -> Dict[str, ConfigTemplate]:
    """Return deep copies of the built-in templates."""
    return {name: template.model_copy(deep=True) for name, template in BUILTIN_TEMPLATES.items()}


class Config(BaseSettings):
    """Main configuration for GKE Log Processor."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    gke: GKEConfig = Field(default_factory=GKEConfig, description="Active connection settings")
    kubernetes: KubernetesConfig = Field(
        default_factory=KubernetesConfig, description="Kubernetes defaults"
    )
    ai: AIConfig = Field(
        default_factory=AIConfig,
        alias="gemini",
        description="Gemini AI configuration",
    )
    ui: UIConfig = Field(default_factory=UIConfig, description="UI configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    streaming: StreamingConfig = Field(
        default_factory=StreamingConfig, description="Log streaming configuration"
    )

    clusters: List[ClusterConfig] = Field(default_factory=list, description="Saved clusters")
    active_cluster: Optional[str] = Field(
        None, description="Name of the currently active saved cluster"
    )
    profiles: Dict[str, ProfileConfig] = Field(
        default_factory=dict, description="Saved configuration profiles"
    )
    active_profile: Optional[str] = Field(
        None, description="Name of the active profile, if any"
    )

    config_file: Optional[str] = Field(
        None, description="Path to the configuration file used to load settings"
    )
    verbose: bool = Field(False, description="Enable verbose logging output")

    gemini_api_key: Optional[str] = Field(
        default=None, description="Environment override for Gemini API key"
    )
    google_application_credentials: Optional[str] = Field(
        default=None, description="Path to Google Cloud service account key"
    )

    config_search_paths: List[str] = Field(
        default_factory=lambda: [
            "./gke-logs.yaml",
            "./gke-logs.yml",
            "~/.config/gke-logs/config.yaml",
            "~/.config/gke-logs/config.yml",
            "/etc/gke-logs/config.yaml",
            "/etc/gke-logs/config.yml",
        ],
        description="Paths searched when a config file is not explicitly provided",
    )

    @model_validator(mode="after")
    def _apply_active_context(self) -> "Config":
        """Reapply active profile or cluster so derived fields stay in sync."""
        if self.active_profile:
            try:
                self.apply_profile(self.active_profile)
                return self
            except ValueError:
                # Leave inconsistent state for downstream validation warnings
                return self

        if self.active_cluster:
            try:
                self.set_active_cluster(self.active_cluster)
            except ValueError:
                pass

        return self

    @property
    def gemini(self) -> AIConfig:
        """Provide backwards compatible access to AI configuration."""
        return self.ai

    @gemini.setter
    def gemini(self, value: AIConfig) -> None:
        self.ai = value

    @property
    def cluster_name(self) -> Optional[str]:
        """Compatibility accessor for legacy code."""
        return self.gke.cluster_name

    @cluster_name.setter
    def cluster_name(self, value: Optional[str]) -> None:
        self.gke.cluster_name = value

    @property
    def project_id(self) -> Optional[str]:
        """Compatibility accessor for legacy code."""
        return self.gke.project_id

    @project_id.setter
    def project_id(self, value: Optional[str]) -> None:
        self.gke.project_id = value

    @property
    def zone(self) -> Optional[str]:
        """Compatibility accessor for legacy code."""
        return self.gke.zone

    @zone.setter
    def zone(self, value: Optional[str]) -> None:
        self.gke.zone = value
        if value:
            self.gke.region = None

    @property
    def region(self) -> Optional[str]:
        """Compatibility accessor for legacy code."""
        return self.gke.region

    @region.setter
    def region(self, value: Optional[str]) -> None:
        self.gke.region = value
        if value:
            self.gke.zone = None

    @property
    def namespace(self) -> str:
        """Compatibility accessor for legacy code."""
        return self.kubernetes.default_namespace

    @namespace.setter
    def namespace(self, value: str) -> None:
        self.kubernetes.default_namespace = value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_cluster(self) -> Optional[ClusterConfig]:
        """Determine the current effective cluster settings."""
        if self.active_cluster:
            cluster = self.get_cluster(self.active_cluster)
            if cluster:
                return cluster

        if self.gke.cluster_name and self.gke.project_id:
            return ClusterConfig(
                name=self.gke.cluster_name,
                project_id=self.gke.project_id,
                zone=self.gke.zone,
                region=self.gke.region,
                namespace=self.kubernetes.default_namespace,
            )

        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Resolve the best available Gemini API key."""
        return self.ai.gemini_api_key or self.gemini_api_key

    def get_cluster(self, name: str) -> Optional[ClusterConfig]:
        """Retrieve a saved cluster by name."""
        return next((cluster for cluster in self.clusters if cluster.name == name), None)

    def iter_clusters(self) -> Iterable[ClusterConfig]:
        """Iterate over saved clusters."""
        return iter(self.clusters)

    def upsert_cluster(self, cluster: ClusterConfig, *, set_active: bool = False) -> None:
        """Insert or update a cluster definition."""
        for idx, existing in enumerate(self.clusters):
            if existing.name == cluster.name:
                self.clusters[idx] = cluster
                break
        else:
            self.clusters.append(cluster)

        if set_active:
            self.set_active_cluster(cluster.name)

    def remove_cluster(self, name: str) -> bool:
        """Remove a cluster by name, returning True if removed."""
        for idx, cluster in enumerate(self.clusters):
            if cluster.name == name:
                del self.clusters[idx]
                if self.active_cluster == name:
                    self.active_cluster = None
                return True
        return False

    def set_active_cluster(self, name: Optional[str]) -> None:
        """Activate a cluster by name and update active connection settings."""
        if name is None:
            self.active_cluster = None
            return

        cluster = self.get_cluster(name)
        if cluster is None:
            raise ValueError(f"Cluster '{name}' not found")

        self.active_cluster = cluster.name
        self.gke.apply_cluster(cluster)
        self.kubernetes.default_namespace = cluster.namespace

    def list_profiles(self) -> List[ProfileConfig]:
        """Return profiles in insertion order."""
        return list(self.profiles.values())

    def get_profile(self, name: str) -> Optional[ProfileConfig]:
        """Retrieve a profile by name."""
        return self.profiles.get(name)

    def upsert_profile(self, profile: ProfileConfig, *, set_active: bool = False) -> None:
        """Insert or update a profile."""
        self.profiles[profile.name] = profile
        if set_active:
            self.apply_profile(profile.name)

    def remove_profile(self, name: str) -> bool:
        """Remove a profile by name."""
        removed = self.profiles.pop(name, None)
        if removed and self.active_profile == name:
            self.active_profile = None
        return removed is not None

    def apply_profile(self, name: str) -> None:
        """Apply the specified profile and mark it active."""
        profile = self.get_profile(name)
        if profile is None:
            raise ValueError(f"Profile '{name}' not found")

        profile.apply_to(self)
        self.active_profile = profile.name

    @property
    def available_templates(self) -> Dict[str, ConfigTemplate]:
        """Expose available templates for UI consumption."""
        return get_builtin_templates()

    def apply_template(
        self, template: Union[str, ConfigTemplate], *, set_active: bool = True
    ) -> ConfigTemplate:
        """Apply a configuration template.

        Args:
            template: Template name or instance to apply.
            set_active: Whether to activate default cluster/profile after applying.

        Returns:
            The applied template instance (deep copy).
        """

        if isinstance(template, str):
            templates = self.available_templates
            if template not in templates:
                raise ValueError(f"Unknown template '{template}'")
            template_obj = templates[template]
        else:
            template_obj = template.model_copy(deep=True)

        for cluster in template_obj.clusters:
            self.upsert_cluster(cluster, set_active=False)

        if template_obj.kubernetes:
            self.kubernetes = template_obj.kubernetes.model_copy(deep=True)
        if template_obj.logging:
            self.logging = template_obj.logging.model_copy(deep=True)
        if template_obj.streaming:
            self.streaming = template_obj.streaming.model_copy(deep=True)
        if template_obj.ai:
            self.ai = template_obj.ai.model_copy(deep=True)
        if template_obj.ui:
            self.ui = template_obj.ui.model_copy(deep=True)

        for profile in template_obj.profiles:
            self.upsert_profile(profile, set_active=False)

        if set_active:
            if template_obj.default_profile and template_obj.default_profile in self.profiles:
                self.apply_profile(template_obj.default_profile)
            elif template_obj.clusters:
                self.set_active_cluster(template_obj.clusters[0].name)

        return template_obj

    @staticmethod
    def _expand_environment_variables(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively expand environment variables in configuration data."""
        if isinstance(config_data, dict):
            return {
                key: Config._expand_environment_variables(value)
                for key, value in config_data.items()
            }
        if isinstance(config_data, list):
            return [Config._expand_environment_variables(item) for item in config_data]
        if isinstance(config_data, str):
            return os.path.expandvars(config_data)
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
            with open(config_file, "r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in configuration file {config_file}: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - safeguards unexpected failures
            raise RuntimeError(
                f"Error reading configuration file {config_file}: {exc}"
            ) from exc

        return cls._expand_environment_variables(config_data)

    @classmethod
    def find_config_file(
        cls, search_paths: Optional[List[str]] = None
    ) -> Optional[Path]:
        """Find the first existing configuration file in the search paths."""
        paths = search_paths or cls().config_search_paths  # type: ignore[call-arg]
        for path_str in paths:
            candidate = Path(path_str).expanduser().resolve()
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @classmethod
    def load_from_file(
        cls,
        config_path: Optional[str] = None,
        search_paths: Optional[List[str]] = None,
    ) -> "Config":
        """Load configuration from YAML."""
        if config_path:
            config_data = cls._load_yaml_file(config_path)
        else:
            config_file = cls.find_config_file(search_paths)
            if not config_file:
                return cls()  # type: ignore[call-arg]
            config_data = cls._load_yaml_file(config_file)

        # Normalise profiles which may be specified as a list in older configs
        profiles_data = config_data.get("profiles")
        if isinstance(profiles_data, list):
            config_data["profiles"] = {p["name"]: p for p in profiles_data if "name" in p}

        try:
            return cls(**config_data)
        except ValidationError as exc:
            raise ValueError(f"Configuration validation failed: {exc}") from exc

    @classmethod
    def load_with_overrides(
        cls,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        """Load configuration with optional overrides."""
        config = cls.load_from_file(config_path)

        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def save_to_file(self, config_path: str, include_cli_args: bool = False) -> None:
        """Persist the current configuration to disk."""
        config_file = Path(config_path).expanduser().resolve()
        config_file.parent.mkdir(parents=True, exist_ok=True)

        export_data: Dict[str, Any] = {
            "gke": self.gke.model_dump(exclude_none=True),
            "kubernetes": self.kubernetes.model_dump(exclude_none=True),
            "ai": self.ai.model_dump(exclude_none=True),
            "ui": self.ui.model_dump(exclude_none=True),
            "logging": self.logging.model_dump(exclude_none=True),
            "streaming": self.streaming.model_dump(exclude_none=True),
            "clusters": [cluster.model_dump(exclude_none=True) for cluster in self.clusters],
            "profiles": {
                name: profile.model_dump(exclude_none=True)
                for name, profile in self.profiles.items()
            },
            "active_cluster": self.active_cluster,
            "active_profile": self.active_profile,
        }

        if include_cli_args:
            export_data["cli_defaults"] = {
                "cluster_name": self.gke.cluster_name,
                "project_id": self.gke.project_id,
                "zone": self.gke.zone,
                "region": self.gke.region,
                "namespace": self.kubernetes.default_namespace,
                "verbose": self.verbose,
            }

        try:
            with open(config_file, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    export_data,
                    handle,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )
        except Exception as exc:  # pragma: no cover - safeguard
            raise RuntimeError(
                f"Error writing configuration file {config_file}: {exc}"
            ) from exc

    @classmethod
    def create_template(cls, config_path: str) -> None:
        """Generate a documented configuration template."""
        config_file = Path(config_path).expanduser().resolve()
        config_file.parent.mkdir(parents=True, exist_ok=True)

        base_config = cls()  # type: ignore[call-arg]

        yaml_lines: List[str] = [
            "# GKE Log Processor Configuration",
            "# This template supports environment variable expansion via $VAR / ${VAR} syntax",
            "",
            "# Saved clusters allow you to switch between environments quickly",
            "clusters:",
            "  - name: my-cluster",
            "    project_id: ${GCP_PROJECT_ID}",
            "    zone: us-central1-a",
            "    namespace: default",
            "",
            "# Optional saved profiles for quick switching",
            "profiles:",
            "  my-profile:",
            "    cluster: my-cluster",
            "    namespace: default",
            "",
        ]

        sections = {
            "gke": "# Active connection defaults (overridden when applying a profile)",
            "kubernetes": "# Kubernetes API defaults",
            "ai": "# Gemini AI configuration (set GEMINI_API_KEY env var when possible)",
            "logging": "# Logging configuration",
            "streaming": "# Log streaming configuration",
            "ui": "# UI preferences",
        }

        for section, comment in sections.items():
            yaml_lines.append(comment)
            section_data = getattr(base_config, section).model_dump()
            if section == "ai" and "gemini_api_key" in section_data:
                section_data["gemini_api_key"] = "${GEMINI_API_KEY}"

            section_yaml = yaml.safe_dump(
                {section: section_data}, default_flow_style=False, sort_keys=False, indent=2
            )
            yaml_lines.extend(section_yaml.strip().split("\n"))
            yaml_lines.append("")

        try:
            with open(config_file, "w", encoding="utf-8") as handle:
                handle.write("\n".join(yaml_lines))
        except Exception as exc:  # pragma: no cover - safeguard
            raise RuntimeError(
                f"Error creating template file {config_file}: {exc}"
            ) from exc

    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any warnings."""
        warnings: List[str] = []

        if self.active_cluster and not self.get_cluster(self.active_cluster):
            warnings.append(
                f"Active cluster '{self.active_cluster}' is not present in saved clusters"
            )

        if not self.active_cluster and not self.current_cluster:
            warnings.append("No active cluster configured")

        for profile in self.profiles.values():
            if not self.get_cluster(profile.cluster):
                warnings.append(
                    f"Profile '{profile.name}' references unknown cluster '{profile.cluster}'"
                )

        if not self.effective_gemini_api_key:
            warnings.append("Gemini API key not configured - AI features will be disabled")

        if not self.google_application_credentials and not os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            warnings.append("Google Cloud credentials not configured")

        if self.logging.file:
            log_path = Path(self.logging.file).expanduser().resolve()
            if not log_path.parent.exists():
                warnings.append(f"Log directory does not exist: {log_path.parent}")

        return warnings
