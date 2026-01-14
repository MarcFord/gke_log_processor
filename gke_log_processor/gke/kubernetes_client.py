"""
Kubernetes client for pod discovery, selection, and monitoring.

This module provides a high-level interface for interacting with Kubernetes
clusters through the Kubernetes Python client library.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import kubernetes  # type: ignore[import-untyped]
from kubernetes.client import (  # type: ignore[import-untyped]
    ApiException,
    CoreV1Api,
    V1Pod,
    V1PodList,
)
from kubernetes.client.configuration import (
    Configuration,  # type: ignore[import-untyped]
)

from ..core.exceptions import KubernetesConnectionError, PodNotFoundError

logger = logging.getLogger(__name__)


class PodInfo:
    """Information about a Kubernetes pod."""

    def __init__(self, pod: V1Pod):
        """Initialize from a Kubernetes V1Pod object."""
        self.name = pod.metadata.name
        self.namespace = pod.metadata.namespace
        self.uid = pod.metadata.uid
        self.labels = pod.metadata.labels or {}
        self.annotations = pod.metadata.annotations or {}
        self.creation_timestamp = pod.metadata.creation_timestamp

        # Pod status information
        self.phase = pod.status.phase  # Pending, Running, Succeeded, Failed, Unknown
        self.conditions = pod.status.conditions or []
        self.container_statuses = pod.status.container_statuses or []

        # Node information
        self.node_name = pod.spec.node_name
        self.pod_ip = pod.status.pod_ip
        self.host_ip = pod.status.host_ip

        # Container information
        self.containers = [container.name for container in pod.spec.containers]
        self.restart_policy = pod.spec.restart_policy

    @property
    def is_ready(self) -> bool:
        """Check if the pod is ready."""
        if self.phase != "Running":
            return False

        # Check if all containers are ready
        for status in self.container_statuses:
            if not status.ready:
                return False

        return True

    @property
    def restart_count(self) -> int:
        """Get the total restart count for all containers."""
        return sum(status.restart_count for status in self.container_statuses)

    @property
    def age(self) -> str:
        """Get the age of the pod as a human-readable string."""
        if not self.creation_timestamp:
            return "Unknown"

        now = datetime.now(timezone.utc)
        age_delta = now - self.creation_timestamp.replace(tzinfo=timezone.utc)

        days = age_delta.days
        hours, remainder = divmod(age_delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if days > 0:
            return f"{days}d{hours}h"
        elif hours > 0:
            return f"{hours}h{minutes}m"
        else:
            return f"{minutes}m"

    @property
    def status_summary(self) -> str:
        """Get a summary of the pod's status."""
        if self.phase == "Running" and self.is_ready:
            return f"Running ({len(self.containers)} containers)"
        elif self.phase == "Running":
            ready_count = sum(1 for status in self.container_statuses if status.ready)
            return f"Running ({ready_count}/{len(self.containers)} ready)"
        else:
            return self.phase

    def __str__(self) -> str:
        """String representation of the pod."""
        return f"{self.namespace}/{self.name} ({self.status_summary})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"PodInfo(name='{self.name}', namespace='{self.namespace}', "
                f"phase='{self.phase}', ready={self.is_ready})")


class KubernetesClient:
    """High-level Kubernetes client for pod management and monitoring."""

    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the Kubernetes client.

        Args:
            config: Optional Kubernetes configuration. If None, will use
                   the configuration from GKE client.
        """
        self._config = config
        self._core_v1_api: Optional[CoreV1Api] = None
        self._pod_cache: Dict[str, List[PodInfo]] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 30  # Cache for 30 seconds

    def _get_api(self) -> CoreV1Api:
        """Get or create the Kubernetes Core V1 API client."""
        if self._core_v1_api is None:
            if self._config:
                self._core_v1_api = CoreV1Api(
                    api_client=kubernetes.client.ApiClient(configuration=self._config)
                )
            else:
                # Use default configuration (assumes kubectl is configured)
                kubernetes.config.load_incluster_config()
                self._core_v1_api = CoreV1Api()

        return self._core_v1_api

    def configure(self, config: Configuration) -> None:
        """Configure the client with a Kubernetes configuration."""
        self._config = config
        self._core_v1_api = None  # Reset API client

    async def list_namespaces(self) -> List[str]:
        """
        List all available namespaces in the cluster.

        Returns:
            List of namespace names.

        Raises:
            KubernetesConnectionError: If unable to connect to cluster.
        """
        try:
            api = self._get_api()
            response = api.list_namespace()
            return [ns.metadata.name for ns in response.items]
        except ApiException as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise KubernetesConnectionError(f"Failed to list namespaces: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error listing namespaces: {e}")
            raise KubernetesConnectionError(f"Unexpected error: {e}") from e

    async def list_pods(self, namespace: str = "default",
                        label_selector: Optional[str] = None,
                        field_selector: Optional[str] = None,
                        force_refresh: bool = False) -> List[PodInfo]:
        """
        List pods in a namespace.

        Args:
            namespace: Kubernetes namespace to list pods from.
            label_selector: Optional label selector (e.g., "app=my-app").
            field_selector: Optional field selector (e.g., "status.phase=Running").
            force_refresh: Force refresh of the pod cache.

        Returns:
            List of PodInfo objects.

        Raises:
            KubernetesConnectionError: If unable to connect to cluster.
        """
        cache_key = f"{namespace}:{label_selector}:{field_selector}"
        now = datetime.now()

        # Check cache
        if (not force_refresh and
            cache_key in self._pod_cache and
                cache_key in self._cache_timestamp):

            cache_age = (now - self._cache_timestamp[cache_key]).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                logger.debug(f"Using cached pod list for {namespace}")
                return self._pod_cache[cache_key]

        try:
            api = self._get_api()
            response: V1PodList = api.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                field_selector=field_selector
            )

            pods = [PodInfo(pod) for pod in response.items]

            # Update cache
            self._pod_cache[cache_key] = pods
            self._cache_timestamp[cache_key] = now

            logger.info(f"Found {len(pods)} pods in namespace '{namespace}'")
            return pods

        except ApiException as e:
            logger.error(f"Failed to list pods in namespace '{namespace}': {e}")
            if e.status == 404:
                raise PodNotFoundError(f"Namespace '{namespace}' not found") from e
            raise KubernetesConnectionError(f"Failed to list pods: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error listing pods: {e}")
            raise KubernetesConnectionError(f"Unexpected error: {e}") from e

    async def get_pod(self, name: str, namespace: str = "default") -> PodInfo:
        """
        Get information about a specific pod.

        Args:
            name: Pod name.
            namespace: Kubernetes namespace.

        Returns:
            PodInfo object for the pod.

        Raises:
            PodNotFoundError: If the pod is not found.
            KubernetesConnectionError: If unable to connect to cluster.
        """
        try:
            api = self._get_api()
            response = api.read_namespaced_pod(name=name, namespace=namespace)
            return PodInfo(response)

        except ApiException as e:
            logger.error(f"Failed to get pod '{namespace}/{name}': {e}")
            if e.status == 404:
                raise PodNotFoundError(f"Pod '{namespace}/{name}' not found") from e
            raise KubernetesConnectionError(f"Failed to get pod: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting pod: {e}")
            raise KubernetesConnectionError(f"Unexpected error: {e}") from e

    async def find_pods_by_label(self, labels: Dict[str, str],
                                 namespace: str = "default") -> List[PodInfo]:
        """
        Find pods matching specific labels.

        Args:
            labels: Dictionary of label key-value pairs.
            namespace: Kubernetes namespace to search in.

        Returns:
            List of PodInfo objects matching the labels.
        """
        label_selector = ",".join(f"{k}={v}" for k, v in labels.items())
        return await self.list_pods(namespace=namespace, label_selector=label_selector)

    async def find_pods_by_app(self, app_name: str,
                               namespace: str = "default") -> List[PodInfo]:
        """
        Find pods for a specific application.

        Args:
            app_name: Application name (looks for 'app' label).
            namespace: Kubernetes namespace to search in.

        Returns:
            List of PodInfo objects for the application.
        """
        return await self.find_pods_by_label({"app": app_name}, namespace)

    async def get_running_pods(self, namespace: str = "default") -> List[PodInfo]:
        """
        Get all running pods in a namespace.

        Args:
            namespace: Kubernetes namespace.

        Returns:
            List of PodInfo objects for running pods.
        """
        return await self.list_pods(
            namespace=namespace,
            field_selector="status.phase=Running"
        )

    async def watch_pod_status(self, callback: Callable[[PodInfo, str], None],
                               namespace: str = "default",
                               pod_name: Optional[str] = None,
                               label_selector: Optional[str] = None) -> None:
        """
        Watch for pod status changes.

        Args:
            callback: Function to call when pod status changes.
                     Receives (pod_info, event_type) where event_type is
                     'ADDED', 'MODIFIED', or 'DELETED'.
            namespace: Kubernetes namespace to watch.
            pod_name: Optional specific pod name to watch.
            label_selector: Optional label selector for filtering pods.
        """
        try:
            api = self._get_api()

            field_selector = None
            if pod_name:
                field_selector = f"metadata.name={pod_name}"

            # Create watch stream
            watch = kubernetes.watch.Watch()

            logger.info(f"Starting pod watch for namespace '{namespace}'")

            for event in watch.stream(
                api.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
                field_selector=field_selector,
                timeout_seconds=0  # Watch indefinitely
            ):
                event_type = event['type']
                pod = event['object']

                if isinstance(pod, V1Pod):
                    pod_info = PodInfo(pod)
                    logger.debug(f"Pod {event_type}: {pod_info}")

                    try:
                        callback(pod_info, event_type)
                    except Exception as e:
                        logger.error(f"Error in pod watch callback: {e}")

        except ApiException as e:
            logger.error(f"Failed to watch pods: {e}")
            raise KubernetesConnectionError(f"Failed to watch pods: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error watching pods: {e}")
            raise KubernetesConnectionError(f"Unexpected error: {e}") from e

    async def get_pod_containers(self, name: str, namespace: str = "default") -> List[str]:
        """
        Get list of container names in a pod.

        Args:
            name: Pod name.
            namespace: Kubernetes namespace.

        Returns:
            List of container names.
        """
        pod_info = await self.get_pod(name, namespace)
        return pod_info.containers

    async def is_pod_ready(self, name: str, namespace: str = "default") -> bool:
        """
        Check if a pod is ready.

        Args:
            name: Pod name.
            namespace: Kubernetes namespace.

        Returns:
            True if pod is ready, False otherwise.
        """
        try:
            pod_info = await self.get_pod(name, namespace)
            return pod_info.is_ready
        except PodNotFoundError:
            return False

    def clear_cache(self) -> None:
        """Clear the internal pod cache."""
        self._pod_cache.clear()
        self._cache_timestamp.clear()
        logger.debug("Pod cache cleared")

    async def validate_connection(self) -> bool:
        """
        Validate connection to the Kubernetes cluster.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            await self.list_namespaces()
            return True
        except KubernetesConnectionError:
            return False
        except Exception:
            return False

    async def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get basic cluster information.

        Returns:
            Dictionary with cluster information.
        """
        try:
            api = self._get_api()

            # Get cluster version
            version_api = kubernetes.client.VersionApi()
            version_info = version_api.get_code()

            # Get node count
            nodes = api.list_node()
            node_count = len(nodes.items)

            # Get namespace count
            namespaces = await self.list_namespaces()
            namespace_count = len(namespaces)

            return {
                "kubernetes_version": f"{version_info.major}.{version_info.minor}",
                "git_version": version_info.git_version,
                "node_count": node_count,
                "namespace_count": namespace_count,
                "connection_status": "Connected"
            }

        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {
                "connection_status": "Failed",
                "error": str(e)
            }
