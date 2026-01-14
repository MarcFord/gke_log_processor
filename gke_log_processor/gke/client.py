"""GKE cluster client for authentication and connection management."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google.api_core import exceptions as gcp_exceptions
from google.auth import default
from google.auth.credentials import Credentials
from google.cloud import container_v1
from kubernetes import client as k8s_client  # type: ignore[import-untyped]
from kubernetes.client import ApiException  # type: ignore[import-untyped]

from ..core.config import Config
from ..core.exceptions import ConfigurationError, GKEConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a GKE cluster."""

    name: str
    location: str
    project_id: str
    endpoint: str
    ca_certificate: str
    is_regional: bool
    status: str
    node_count: int
    kubernetes_version: str
    network: str
    subnetwork: str


class GKEClient:
    """Client for connecting to and managing GKE clusters."""

    def __init__(self, config: Config):
        """Initialize GKE client with configuration.

        Args:
            config: Application configuration containing cluster details
        """
        self.config = config
        self._credentials: Optional[Credentials] = None
        self._container_client: Optional[container_v1.ClusterManagerClient] = None
        self._k8s_client: Optional[k8s_client.ApiClient] = None
        self._cluster_info: Optional[ClusterInfo] = None

    @property
    def credentials(self) -> Credentials:
        """Get Google Cloud credentials."""
        if self._credentials is None:
            self._credentials = self._authenticate()
        return self._credentials

    @property
    def container_client(self) -> container_v1.ClusterManagerClient:
        """Get Google Cloud Container client."""
        if self._container_client is None:
            self._container_client = container_v1.ClusterManagerClient(
                credentials=self.credentials
            )
        return self._container_client

    @property
    def kubernetes_client(self) -> k8s_client.ApiClient:
        """Get authenticated Kubernetes client."""
        if self._k8s_client is None:
            self._k8s_client = self._create_kubernetes_client()
        return self._k8s_client

    @property
    def cluster_info(self) -> ClusterInfo:
        """Get information about the connected cluster."""
        if self._cluster_info is None:
            self._cluster_info = self._fetch_cluster_info()
        return self._cluster_info

    def _authenticate(self) -> Credentials:
        """Authenticate with Google Cloud.

        Returns:
            Google Cloud credentials

        Raises:
            GKEConnectionError: If authentication fails
        """
        try:
            logger.info("Authenticating with Google Cloud...")

            # Try to get default credentials
            credentials, project = default()

            # Validate that we can use the credentials
            if not credentials:
                raise GKEConnectionError(
                    "No valid Google Cloud credentials found")

            # If project from credentials doesn't match config, use config
            # project
            if project and project != self.config.project_id:
                logger.warning(
                    f"Credentials project ({project}) differs from config project "
                    f"({self.config.project_id}). Using config project."
                )

            logger.info(
                f"Successfully authenticated with Google Cloud for project: {
                    self.config.project_id}")
            return credentials

        except Exception as e:
            raise GKEConnectionError(
                f"Failed to authenticate with Google Cloud: {e}") from e

    def _create_kubernetes_client(self) -> k8s_client.ApiClient:
        """Create authenticated Kubernetes client for the GKE cluster.

        Returns:
            Kubernetes API client

        Raises:
            GKEConnectionError: If client creation fails
        """
        try:
            logger.info(
                f"Creating Kubernetes client for cluster: {
                    self.config.cluster_name}")

            # Get cluster info first
            cluster_info = self.cluster_info

            # Create Kubernetes configuration
            configuration = k8s_client.Configuration()
            configuration.host = f"https://{cluster_info.endpoint}"
            configuration.verify_ssl = True

            # Set up authentication
            self.credentials.refresh(default()[1])  # Refresh credentials
            configuration.api_key_prefix['authorization'] = 'Bearer'
            configuration.api_key['authorization'] = self.credentials.token

            # Set up CA certificate
            import base64
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as ca_cert_file:
                ca_cert_data = base64.b64decode(cluster_info.ca_certificate)
                ca_cert_file.write(ca_cert_data)
                ca_cert_file.flush()
                configuration.ssl_ca_cert = ca_cert_file.name

            # Create and return API client
            api_client = k8s_client.ApiClient(configuration)

            logger.info("Successfully created Kubernetes client")
            return api_client

        except Exception as e:
            raise GKEConnectionError(
                f"Failed to create Kubernetes client: {e}") from e

    def _fetch_cluster_info(self) -> ClusterInfo:
        """Fetch detailed information about the GKE cluster.

        Returns:
            Cluster information

        Raises:
            GKEConnectionError: If cluster info cannot be fetched
        """
        try:
            logger.info(
                f"Fetching cluster info for: {
                    self.config.cluster_name}")

            current_cluster = self.config.current_cluster
            if not current_cluster:
                raise ConfigurationError("No valid cluster configuration")

            # Build cluster path
            if current_cluster.is_regional:
                parent = f"projects/{current_cluster.project_id}/locations/{current_cluster.location}"
            else:
                parent = f"projects/{current_cluster.project_id}/locations/{current_cluster.location}"

            # Get cluster details
            cluster_name = f"{parent}/clusters/{current_cluster.name}"
            cluster = self.container_client.get_cluster(name=cluster_name)

            # Extract cluster information
            cluster_info = ClusterInfo(
                name=cluster.name,
                location=cluster.location,
                project_id=current_cluster.project_id,
                endpoint=cluster.endpoint,
                ca_certificate=cluster.master_auth.cluster_ca_certificate,
                is_regional=current_cluster.is_regional,
                status=cluster.status.name,
                node_count=sum(
                    pool.initial_node_count for pool in cluster.node_pools),
                kubernetes_version=cluster.current_master_version,
                network=cluster.network,
                subnetwork=cluster.subnetwork or "default"
            )

            logger.info(
                f"Successfully fetched cluster info: {cluster_info.name} "
                f"(Status: {
                    cluster_info.status}, Nodes: {
                    cluster_info.node_count}, "
                f"K8s: {cluster_info.kubernetes_version})"
            )

            return cluster_info

        except gcp_exceptions.NotFound:
            if current_cluster is None:
                raise GKEConnectionError("No cluster configuration available")
            raise GKEConnectionError(
                f"Cluster '{current_cluster.name}' not found in "
                f"project '{current_cluster.project_id}', location '{current_cluster.location}'"
            )
        except gcp_exceptions.PermissionDenied:
            if current_cluster is None:
                raise GKEConnectionError("No cluster configuration available")
            raise GKEConnectionError(
                f"Permission denied accessing cluster '{current_cluster.name}'. "
                "Check your Google Cloud permissions."
            )
        except Exception as e:
            raise GKEConnectionError(
                f"Failed to fetch cluster info: {e}") from e

    def validate_connection(self) -> bool:
        """Validate the connection to the GKE cluster.

        Returns:
            True if connection is valid

        Raises:
            GKEConnectionError: If connection validation fails
        """
        try:
            logger.info("Validating GKE cluster connection...")

            # Check cluster status
            cluster_info = self.cluster_info
            if cluster_info.status not in ["RUNNING", "RECONCILING"]:
                raise GKEConnectionError(
                    f"Cluster is not in a running state. Current status: {
                        cluster_info.status}"
                )

            # Test Kubernetes API access
            v1 = k8s_client.CoreV1Api(self.kubernetes_client)

            # Try to list namespaces (basic connectivity test)
            try:
                namespaces = v1.list_namespace(limit=1)
                logger.info(
                    f"Successfully connected to Kubernetes API (found {len(namespaces.items)} namespaces)")
            except ApiException as e:
                if e.status == 403:
                    raise GKEConnectionError(
                        "Permission denied accessing Kubernetes API. "
                        "Check your cluster RBAC permissions."
                    )
                else:
                    raise GKEConnectionError(
                        f"Kubernetes API error: {e}") from e

            # Check if target namespace exists
            try:
                v1.read_namespace(name=self.config.namespace)
                logger.info(
                    f"Target namespace '{
                        self.config.namespace}' is accessible")
            except ApiException as e:
                if e.status == 404:
                    logger.warning(
                        f"Namespace '{
                            self.config.namespace}' does not exist")
                    return False
                elif e.status == 403:
                    raise GKEConnectionError(
                        f"Permission denied accessing namespace '{
                            self.config.namespace}'"
                    )
                else:
                    raise GKEConnectionError(
                        f"Error accessing namespace: {e}") from e

            logger.info("GKE cluster connection validation successful")
            return True

        except GKEConnectionError:
            raise
        except Exception as e:
            raise GKEConnectionError(
                f"Connection validation failed: {e}") from e

    def list_namespaces(self) -> List[str]:
        """List all available namespaces in the cluster.

        Returns:
            List of namespace names

        Raises:
            GKEConnectionError: If namespace listing fails
        """
        try:
            v1 = k8s_client.CoreV1Api(self.kubernetes_client)
            namespaces = v1.list_namespace()
            namespace_names = [ns.metadata.name for ns in namespaces.items]

            logger.info(
                f"Found {len(namespace_names)} namespaces: {namespace_names[:5]}...")
            return namespace_names

        except Exception as e:
            raise GKEConnectionError(f"Failed to list namespaces: {e}") from e

    def get_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Get information about cluster nodes.

        Returns:
            List of node information dictionaries

        Raises:
            GKEConnectionError: If node information cannot be retrieved
        """
        try:
            v1 = k8s_client.CoreV1Api(self.kubernetes_client)
            nodes = v1.list_node()

            node_info = []
            for node in nodes.items:
                info = {
                    "name": node.metadata.name,
                    "status": "Ready" if any(
                        condition.type == "Ready" and condition.status == "True"
                        for condition in node.status.conditions or []
                    ) else "NotReady",
                    "roles": list(node.metadata.labels.get("kubernetes.io/role", "worker").split(",")),
                    "version": node.status.node_info.kubelet_version,
                    "os": f"{node.status.node_info.os_image}",
                    "kernel": node.status.node_info.kernel_version,
                    "container_runtime": node.status.node_info.container_runtime_version,
                    "addresses": [
                        {"type": addr.type, "address": addr.address}
                        for addr in (node.status.addresses or [])
                    ]
                }
                node_info.append(info)

            logger.info(
                f"Retrieved information for {
                    len(node_info)} cluster nodes")
            return node_info

        except Exception as e:
            raise GKEConnectionError(
                f"Failed to get cluster nodes: {e}") from e

    def close(self):
        """Clean up resources."""
        if self._k8s_client:
            self._k8s_client.close()
        # Container client doesn't need explicit cleanup
        logger.info("GKE client resources cleaned up")
