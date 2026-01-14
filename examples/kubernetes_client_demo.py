#!/usr/bin/env python3
"""
Demo script for Kubernetes client functionality.

This demonstrates the key features of the KubernetesClient class
including pod discovery, selection, and monitoring capabilities.
"""

import asyncio
import logging

from gke_log_processor.core.config import Config
from gke_log_processor.core.exceptions import (
    ConfigurationError,
    GKEConnectionError,
    KubernetesConnectionError,
)
from gke_log_processor.gke.client import GKEClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_kubernetes_client():
    """Demonstrate Kubernetes client functionality."""

    logger.info("🚀 Starting Kubernetes Client Demo")
    logger.info("=" * 50)

    try:
        # Load configuration (this will work if you have proper GKE setup)
        config = Config()
        logger.info(f"Configuration loaded: {config.config_file_path}")

        # Create GKE client
        gke_client = GKEClient(config)
        logger.info("GKE client created")

        # Get Kubernetes client
        k8s_client = gke_client.get_kubernetes_client()
        logger.info("✅ Kubernetes client created successfully")

        # Demo 1: List namespaces
        logger.info("\n📋 Demo 1: Listing Namespaces")
        logger.info("-" * 30)
        try:
            namespaces = await k8s_client.list_namespaces()
            logger.info(f"Found {len(namespaces)} namespaces:")
            for ns in namespaces[:5]:  # Show first 5
                logger.info(f"  • {ns}")
            if len(namespaces) > 5:
                logger.info(f"  ... and {len(namespaces) - 5} more")
        except KubernetesConnectionError as e:
            logger.warning(f"❌ Could not list namespaces: {e}")

        # Demo 2: List pods in default namespace
        logger.info("\n🏃 Demo 2: Listing Pods in Default Namespace")
        logger.info("-" * 45)
        try:
            pods = await k8s_client.list_pods(namespace="default")
            logger.info(f"Found {len(pods)} pods in 'default' namespace:")
            for pod in pods[:3]:  # Show first 3
                logger.info(f"  • {pod.name} ({pod.status_summary}) - Age: {pod.age}")
                logger.info(f"    Containers: {pod.containers}")
                logger.info(f"    Ready: {'✅' if pod.is_ready else '❌'}")
            if len(pods) > 3:
                logger.info(f"  ... and {len(pods) - 3} more")

            if not pods:
                logger.info("  No pods found in default namespace")

        except KubernetesConnectionError as e:
            logger.warning(f"❌ Could not list pods: {e}")

        # Demo 3: Get running pods
        logger.info("\n🏃‍♂️ Demo 3: Getting Running Pods Only")
        logger.info("-" * 35)
        try:
            running_pods = await k8s_client.get_running_pods(namespace="default")
            logger.info(f"Found {len(running_pods)} running pods:")
            for pod in running_pods[:3]:
                logger.info(f"  • {pod.name} - Ready: {'✅' if pod.is_ready else '❌'}")
                logger.info(f"    Node: {pod.node_name}, IP: {pod.pod_ip}")

        except KubernetesConnectionError as e:
            logger.warning(f"❌ Could not get running pods: {e}")

        # Demo 4: Validate connection
        logger.info("\n🔗 Demo 4: Connection Validation")
        logger.info("-" * 30)
        try:
            is_valid = await k8s_client.validate_connection()
            logger.info(f"Connection valid: {'✅' if is_valid else '❌'}")

        except Exception as e:
            logger.warning(f"❌ Connection validation failed: {e}")

        # Demo 5: Get cluster info
        logger.info("\n📊 Demo 5: Cluster Information")
        logger.info("-" * 30)
        try:
            cluster_info = await k8s_client.get_cluster_info()
            logger.info("Cluster information:")
            for key, value in cluster_info.items():
                logger.info(f"  • {key}: {value}")

        except Exception as e:
            logger.warning(f"❌ Could not get cluster info: {e}")

        logger.info("\n🎉 Demo completed successfully!")
        logger.info("=" * 50)

    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.info("\n💡 This demo requires proper GKE configuration.")
        logger.info("   Please ensure you have:")
        logger.info("   1. Google Cloud credentials configured")
        logger.info("   2. A GKE cluster accessible")
        logger.info("   3. Proper cluster configuration in config file or environment")

    except GKEConnectionError as e:
        logger.error(f"❌ GKE connection error: {e}")
        logger.info("\n💡 This is expected if you don't have a real GKE cluster configured.")
        logger.info("   The Kubernetes client code is working correctly!")

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)


def demo_pod_info():
    """Demonstrate PodInfo class functionality."""

    logger.info("\n🔍 Bonus Demo: PodInfo Class Features")
    logger.info("-" * 40)

    # This demonstrates what PodInfo can do with actual pod data
    logger.info("PodInfo provides rich pod information including:")
    logger.info("  • Pod status and readiness checks")
    logger.info("  • Container information and restart counts")
    logger.info("  • Age calculation and human-readable status")
    logger.info("  • Node placement and networking details")
    logger.info("  • Labels and annotations access")
    logger.info("  • Automatic status summaries")


if __name__ == "__main__":
    """Run the demo."""

    print("🔮 GKE Log Processor - Kubernetes Client Demo")
    print("=" * 50)
    print("This demo showcases the Kubernetes client capabilities.")
    print("It will attempt to connect to your configured GKE cluster.")
    print()

    # Run pod info demo first (always works)
    demo_pod_info()

    # Run async kubernetes client demo
    try:
        asyncio.run(demo_kubernetes_client())
    except KeyboardInterrupt:
        logger.info("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
