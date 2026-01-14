"""
Tests for Kubernetes client functionality.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from kubernetes.client import ApiException  # type: ignore[import-untyped]
from kubernetes.client import (  # type: ignore[import-untyped]  # type: ignore[import-untyped]
    V1Container, V1ContainerStatus, V1ObjectMeta, V1Pod, V1PodCondition,
    V1PodSpec, V1PodStatus)

from gke_log_processor.core.exceptions import (KubernetesConnectionError,
                                               PodNotFoundError)
from gke_log_processor.gke.kubernetes_client import KubernetesClient, PodInfo


class TestPodInfo:
    """Test the PodInfo class."""

    def create_test_pod(self, name: str = "test-pod", namespace: str = "default",
                        phase: str = "Running", ready: bool = True) -> V1Pod:
        """Create a test V1Pod object."""
        pod = V1Pod()
        pod.metadata = V1ObjectMeta(
            name=name,
            namespace=namespace,
            uid="test-uid-123",
            labels={"app": "test-app", "version": "1.0"},
            annotations={"example.com/annotation": "value"},
            creation_timestamp=datetime.now(timezone.utc)
        )

        pod.status = V1PodStatus(
            phase=phase,
            pod_ip="10.0.0.1",
            host_ip="192.168.1.1",
            conditions=[],
            container_statuses=[
                V1ContainerStatus(
                    name="test-container",
                    ready=ready,
                    restart_count=0,
                    image="test:latest",
                    image_id="sha256:abc123"
                )
            ]
        )

        pod.spec = V1PodSpec(
            node_name="test-node",
            restart_policy="Always",
            containers=[
                V1Container(name="test-container", image="test:latest"),
                V1Container(name="sidecar", image="sidecar:latest")
            ]
        )

        return pod

    def test_pod_info_initialization(self):
        """Test PodInfo initialization from V1Pod."""
        pod = self.create_test_pod()
        pod_info = PodInfo(pod)

        assert pod_info.name == "test-pod"
        assert pod_info.namespace == "default"
        assert pod_info.uid == "test-uid-123"
        assert pod_info.phase == "Running"
        assert pod_info.node_name == "test-node"
        assert pod_info.pod_ip == "10.0.0.1"
        assert pod_info.host_ip == "192.168.1.1"
        assert pod_info.containers == ["test-container", "sidecar"]
        assert pod_info.restart_policy == "Always"

    def test_pod_info_is_ready_true(self):
        """Test PodInfo.is_ready when pod is ready."""
        pod = self.create_test_pod(phase="Running", ready=True)
        pod_info = PodInfo(pod)

        assert pod_info.is_ready is True

    def test_pod_info_is_ready_false_not_running(self):
        """Test PodInfo.is_ready when pod is not running."""
        pod = self.create_test_pod(phase="Pending", ready=True)
        pod_info = PodInfo(pod)

        assert pod_info.is_ready is False

    def test_pod_info_status_summary_running_ready(self):
        """Test PodInfo.status_summary for running ready pod."""
        pod = self.create_test_pod(phase="Running", ready=True)
        pod_info = PodInfo(pod)

        assert pod_info.status_summary == "Running (2 containers)"


class TestKubernetesClient:
    """Test the KubernetesClient class."""

    @pytest.fixture
    def mock_core_v1_api(self):
        """Create a mock Core V1 API."""
        mock_api = Mock()
        return mock_api

    @pytest.fixture
    def kubernetes_client(self, mock_core_v1_api):
        """Create a KubernetesClient with mocked API."""
        client = KubernetesClient()
        client._core_v1_api = mock_core_v1_api
        return client

    def test_client_initialization(self):
        """Test KubernetesClient initialization."""
        client = KubernetesClient()

        assert client._config is None
        assert client._core_v1_api is None
        assert client._pod_cache == {}
        assert client._cache_timestamp == {}

    @pytest.mark.asyncio
    async def test_list_namespaces_success(self, kubernetes_client, mock_core_v1_api):
        """Test successful namespace listing."""
        # Mock namespace response
        mock_ns1 = Mock()
        mock_ns1.metadata.name = "default"
        mock_ns2 = Mock()
        mock_ns2.metadata.name = "kube-system"

        mock_response = Mock()
        mock_response.items = [mock_ns1, mock_ns2]
        mock_core_v1_api.list_namespace.return_value = mock_response

        namespaces = await kubernetes_client.list_namespaces()

        assert namespaces == ["default", "kube-system"]
        mock_core_v1_api.list_namespace.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pod_success(self, kubernetes_client, mock_core_v1_api):
        """Test successful pod retrieval."""
        test_pod = V1Pod()
        test_pod.metadata = V1ObjectMeta(name="test-pod", namespace="default", uid="123")
        test_pod.status = V1PodStatus(phase="Running")
        test_pod.spec = V1PodSpec(containers=[V1Container(name="test", image="test:latest")])

        mock_core_v1_api.read_namespaced_pod.return_value = test_pod

        pod_info = await kubernetes_client.get_pod("test-pod", "default")

        assert isinstance(pod_info, PodInfo)
        assert pod_info.name == "test-pod"
        assert pod_info.namespace == "default"

    @pytest.mark.asyncio
    async def test_validate_connection_success(self, kubernetes_client, mock_core_v1_api):
        """Test successful connection validation."""
        mock_response = Mock()
        mock_response.items = []
        mock_core_v1_api.list_namespace.return_value = mock_response

        is_valid = await kubernetes_client.validate_connection()

        assert is_valid is True
