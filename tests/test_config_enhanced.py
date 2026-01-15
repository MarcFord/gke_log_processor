"""
Tests for enhanced configuration functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from gke_log_processor.core.config import (
    ClusterConfig,
    Config,
    GeminiConfig,
    LoggingConfig,
    StreamingConfig,
    UIConfig,
)


class TestClusterConfig:
    """Test ClusterConfig model."""

    def test_valid_zonal_cluster(self):
        """Test creating a valid zonal cluster configuration."""
        config = ClusterConfig(
            name="test-cluster", project_id="test-project", zone="us-central1-a"
        )

        assert config.name == "test-cluster"
        assert config.project_id == "test-project"
        assert config.zone == "us-central1-a"
        assert config.region is None
        assert config.location == "us-central1-a"
        assert config.is_regional is False

    def test_valid_regional_cluster(self):
        """Test creating a valid regional cluster configuration."""
        config = ClusterConfig(
            name="test-cluster", project_id="test-project", region="us-central1"
        )

        assert config.region == "us-central1"
        assert config.zone is None
        assert config.location == "us-central1"
        assert config.is_regional is True

    def test_invalid_no_location(self):
        """Test validation error when neither zone nor region is specified."""
        with pytest.raises(ValidationError):
            ClusterConfig(name="test-cluster", project_id="test-project")

    def test_invalid_both_locations(self):
        """Test validation error when both zone and region are specified."""
        with pytest.raises(ValidationError):
            ClusterConfig(
                name="test-cluster",
                project_id="test-project",
                zone="us-central1-a",
                region="us-central1",
            )


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file is None
        assert config.max_size == 10
        assert config.backup_count == 5
        assert config.console is True

    def test_valid_log_levels(self):
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_case_insensitive_log_levels(self):
        """Test log levels are case-insensitive."""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"

    def test_invalid_log_level(self):
        """Test invalid log level raises validation error."""
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")


class TestStreamingConfig:
    """Test StreamingConfig model."""

    def test_default_config(self):
        """Test default streaming configuration."""
        config = StreamingConfig()

        assert config.max_buffer_size == 1000
        assert config.buffer_flush_interval == 1.0
        assert config.max_logs_per_second == 100.0
        assert config.follow_logs is True
        assert config.tail_lines == 100
        assert config.timestamps is True

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Test positive constraints
        with pytest.raises(ValidationError):
            StreamingConfig(max_buffer_size=0)

        with pytest.raises(ValidationError):
            StreamingConfig(buffer_flush_interval=0)

        with pytest.raises(ValidationError):
            StreamingConfig(max_logs_per_second=0)


class TestUIConfig:
    """Test UIConfig model."""

    def test_default_config(self):
        """Test default UI configuration."""
        config = UIConfig()

        assert config.theme == "dark"
        assert config.refresh_rate == 1000
        assert config.max_log_lines == 1000
        assert config.show_timestamps is True
        assert config.auto_scroll is True

    def test_theme_validation(self):
        """Test theme validation."""
        # Valid themes
        for theme in ["dark", "light", "DARK", "LIGHT"]:
            config = UIConfig(theme=theme)
            assert config.theme in ["dark", "light"]

        # Invalid theme
        with pytest.raises(ValidationError):
            UIConfig(theme="invalid")


class TestConfig:
    """Test the main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()

        assert config.cluster_name is None
        assert config.project_id is None
        assert config.namespace == "default"
        assert config.verbose is False
        assert len(config.clusters) == 0
        assert isinstance(config.gemini, GeminiConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.streaming, StreamingConfig)

    def test_environment_variable_expansion(self):
        """Test environment variable expansion in configuration."""
        test_data = {
            "simple": "$HOME",
            "braces": "${USER}",
            "nested": {"path": "$HOME/test", "list": ["$HOME", "${USER}"]},
        }

        with patch.dict(os.environ, {"HOME": "/home/test", "USER": "testuser"}):
            expanded = Config._expand_environment_variables(test_data)

        assert expanded["simple"] == "/home/test"
        assert expanded["braces"] == "testuser"
        assert expanded["nested"]["path"] == "/home/test/test"
        assert expanded["nested"]["list"] == ["/home/test", "testuser"]

    def test_load_from_yaml_file(self):
        """Test loading configuration from a YAML file."""
        yaml_content = """
clusters:
  - name: test-cluster
    project_id: test-project
    zone: us-central1-a
    namespace: production

gemini:
  model: gemini-pro
  temperature: 0.2

ui:
  theme: light
  refresh_rate: 500

logging:
  level: DEBUG
  file: /tmp/test.log

streaming:
  max_buffer_size: 2000
  follow_logs: false
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config.load_from_file(temp_path)

            # Check cluster configuration
            assert len(config.clusters) == 1
            cluster = config.clusters[0]
            assert cluster.name == "test-cluster"
            assert cluster.project_id == "test-project"
            assert cluster.zone == "us-central1-a"
            assert cluster.namespace == "production"

            # Check other configurations
            assert config.gemini.model == "gemini-pro"
            assert config.gemini.temperature == 0.2
            assert config.ui.theme == "light"
            assert config.ui.refresh_rate == 500
            assert config.logging.level == "DEBUG"
            assert config.logging.file == "/tmp/test.log"
            assert config.streaming.max_buffer_size == 2000
            assert config.streaming.follow_logs is False

        finally:
            os.unlink(temp_path)

    def test_load_with_environment_variables(self):
        """Test loading configuration with environment variable expansion."""
        yaml_content = """
clusters:
  - name: ${CLUSTER_NAME}
    project_id: ${PROJECT_ID}
    zone: us-central1-a

gemini:
  api_key: ${GEMINI_API_KEY}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with patch.dict(
                os.environ,
                {
                    "CLUSTER_NAME": "my-cluster",
                    "PROJECT_ID": "my-project",
                    "GEMINI_API_KEY": "test-key",
                },
            ):
                config = Config.load_from_file(temp_path)

            cluster = config.clusters[0]
            assert cluster.name == "my-cluster"
            assert cluster.project_id == "my-project"
            assert config.gemini.api_key == "test-key"

        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.load_from_file("nonexistent.yaml")

    def test_find_config_file(self):
        """Test finding configuration files in search paths."""
        # Test with existing file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            search_paths = [temp_path]
            found_path = Config.find_config_file(search_paths)
            assert found_path == Path(temp_path).resolve()
        finally:
            os.unlink(temp_path)

        # Test with non-existent files
        search_paths = ["nonexistent1.yaml", "nonexistent2.yaml"]
        found_path = Config.find_config_file(search_paths)
        assert found_path is None

    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = Config()
        config.clusters = [
            ClusterConfig(
                name="test-cluster", project_id="test-project", zone="us-central1-a"
            )
        ]
        config.gemini.model = "gemini-pro"
        config.ui.theme = "light"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # Remove the temp file so we can test creation
            os.unlink(temp_path)

            config.save_to_file(temp_path)
            assert os.path.exists(temp_path)

            # Load it back and verify
            loaded_config = Config.load_from_file(temp_path)
            assert len(loaded_config.clusters) == 1
            assert loaded_config.clusters[0].name == "test-cluster"
            assert loaded_config.gemini.model == "gemini-pro"
            assert loaded_config.ui.theme == "light"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_create_template(self):
        """Test creating a configuration template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # Remove the temp file so we can test creation
            os.unlink(temp_path)

            Config.create_template(temp_path)
            assert os.path.exists(temp_path)

            # Check that the template contains expected content
            with open(temp_path, "r") as f:
                content = f.read()

            assert "# GKE Log Processor Configuration" in content
            assert "clusters:" in content
            assert "gemini:" in content
            assert "ui:" in content
            assert "logging:" in content
            assert "streaming:" in content
            assert "${GCP_PROJECT_ID}" in content
            assert "${GEMINI_API_KEY}" in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_with_overrides(self):
        """Test loading configuration with overrides."""
        yaml_content = """
ui:
  theme: dark
  refresh_rate: 1000

logging:
  level: INFO
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            overrides = {"verbose": True, "namespace": "production"}

            config = Config.load_with_overrides(temp_path, overrides)

            # Check loaded values
            assert config.ui.theme == "dark"
            assert config.logging.level == "INFO"

            # Check overrides
            assert config.verbose is True
            assert config.namespace == "production"

        finally:
            os.unlink(temp_path)

    def test_validate_config(self):
        """Test configuration validation."""
        config = Config()

        warnings = config.validate_config()

        # Should have warnings about missing configuration
        warning_messages = " ".join(warnings)
        assert "No clusters configured" in warning_messages
        assert "Gemini API key not configured" in warning_messages

    def test_current_cluster_property(self):
        """Test current_cluster computed property."""
        config = Config()

        # No cluster initially
        assert config.current_cluster is None

        # Set CLI arguments
        config.cluster_name = "test-cluster"
        config.project_id = "test-project"
        config.zone = "us-central1-a"

        current = config.current_cluster
        assert current is not None
        assert current.name == "test-cluster"
        assert current.project_id == "test-project"
        assert current.zone == "us-central1-a"

    def test_effective_gemini_api_key(self):
        """Test effective_gemini_api_key computed property."""
        config = Config()

        # No API key initially
        assert config.effective_gemini_api_key is None

        # Set in gemini config
        config.gemini.api_key = "config-key"
        assert config.effective_gemini_api_key == "config-key"

        # Environment variable takes precedence
        config.gemini_api_key = "env-key"
        assert config.effective_gemini_api_key == "config-key"

        # If config key is None, use environment
        config.gemini.api_key = None
        assert config.effective_gemini_api_key == "env-key"
