"""Test configuration and utilities."""

import pytest

from gke_log_processor.core.config import Config


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return Config(
        cluster_name="test-cluster", project_id="test-project", zone="us-central1-a"
    )
