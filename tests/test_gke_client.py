"""Unit tests for GKE client."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from google.auth.credentials import Credentials
from google.cloud import container_v1
from kubernetes.client import ApiException

from gke_log_processor.core.config import Config
from gke_log_processor.core.exceptions import ConfigurationError, GKEConnectionError
from gke_log_processor.gke.client import ClusterInfo, GKEClient


class TestGKEClient:
    """Test cases for GKEClient."""

    def test_initialization(self, sample_config):
        """Test GKE client initialization."""
        client = GKEClient(sample_config)
        assert client.config == sample_config
        assert client._credentials is None
        assert client._container_client is None
        assert client._k8s_client is None

    @patch('gke_log_processor.gke.client.default')
    def test_authentication_success(self, mock_default, sample_config):
        """Test successful authentication."""
        # Mock credentials
        mock_credentials = Mock(spec=Credentials)
        mock_default.return_value = (mock_credentials, "test-project")

        client = GKEClient(sample_config)
        credentials = client.credentials

        assert credentials == mock_credentials
        mock_default.assert_called_once()

    @patch('gke_log_processor.gke.client.default')
    def test_authentication_failure(self, mock_default, sample_config):
        """Test authentication failure."""
        mock_default.side_effect = Exception("Authentication failed")

        client = GKEClient(sample_config)

        with pytest.raises(GKEConnectionError) as exc_info:
            _ = client.credentials

        assert "Failed to authenticate" in str(exc_info.value)

    @patch('gke_log_processor.gke.client.container_v1.ClusterManagerClient')
    def test_container_client_creation(self, mock_client_class, sample_config):
        """Test container client creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with patch.object(GKEClient, 'credentials', Mock()):
            client = GKEClient(sample_config)
            container_client = client.container_client

            assert container_client == mock_client
            mock_client_class.assert_called_once()

    def test_cluster_info_caching(self, sample_config):
        """Test that cluster info is cached properly."""
        client = GKEClient(sample_config)

        # Mock the _fetch_cluster_info method
        mock_cluster_info = Mock(spec=ClusterInfo)
        with patch.object(client, '_fetch_cluster_info', return_value=mock_cluster_info) as mock_fetch:
            # First call should fetch
            info1 = client.cluster_info
            assert info1 == mock_cluster_info

            # Second call should use cache
            info2 = client.cluster_info
            assert info2 == mock_cluster_info

            # Should only call fetch once
            mock_fetch.assert_called_once()

    @patch('gke_log_processor.gke.client.container_v1.ClusterManagerClient')
    def test_fetch_cluster_info_success(
            self, mock_client_class, sample_config):
        """Test successful cluster info fetching."""
        # Mock the cluster response
        mock_cluster = Mock()
        mock_cluster.name = "test-cluster"
        mock_cluster.location = "us-central1-a"
        mock_cluster.endpoint = "1.2.3.4"
        mock_cluster.master_auth.cluster_ca_certificate = "dGVzdA=="  # base64 "test"
        mock_cluster.status.name = "RUNNING"
        mock_cluster.current_master_version = "1.28.0"
        mock_cluster.network = "default"
        mock_cluster.subnetwork = None
        mock_cluster.node_pools = [Mock(initial_node_count=3)]

        mock_client = Mock()
        mock_client.get_cluster.return_value = mock_cluster
        mock_client_class.return_value = mock_client

        with patch.object(GKEClient, 'credentials', Mock()):
            client = GKEClient(sample_config)
            cluster_info = client._fetch_cluster_info()

            assert cluster_info.name == "test-cluster"
            assert cluster_info.location == "us-central1-a"
            assert cluster_info.endpoint == "1.2.3.4"
            assert cluster_info.status == "RUNNING"
            assert cluster_info.node_count == 3
            assert cluster_info.kubernetes_version == "1.28.0"
            assert not cluster_info.is_regional

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        config = Config()
        config.gke.cluster_name = "test-cluster"
        config.gke.project_id = "test-project"
        config.gke.zone = "us-central1-a"
        return config
