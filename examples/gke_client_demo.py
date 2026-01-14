"""Demo script showing how to use the GKE client.

This script demonstrates the GKE client functionality with proper error handling.
To run this with real GKE credentials, set up your Google Cloud authentication:

1. Install gcloud CLI
2. Run: gcloud auth application-default login
3. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable
"""

import logging
import sys

from gke_log_processor.core.config import Config
from gke_log_processor.core.exceptions import ConfigurationError, GKEConnectionError
from gke_log_processor.gke.client import GKEClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def demo_gke_client():
    """Demonstrate GKE client usage."""
    print("GKE Log Processor - Client Demo")
    print("=" * 40)

    # Example 1: Using CLI-style configuration
    print("\n1. Creating configuration...")
    try:
        config = Config(
            cluster_name="my-gke-cluster",
            project_id="my-gcp-project",
            zone="us-central1-a",  # or use region="us-central1" for regional clusters
            namespace="default"
        )
        print(f"✅ Configuration created for cluster: {config.current_cluster.name}")
        print(f"   Project: {config.current_cluster.project_id}")
        print(f"   Location: {config.current_cluster.location}")
        print(f"   Regional: {config.current_cluster.is_regional}")
        print(f"   Namespace: {config.namespace}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    # Example 2: Initialize GKE client
    print("\n2. Initializing GKE client...")
    try:
        gke_client = GKEClient(config)
        print("✅ GKE client initialized")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return

    # Example 3: Test authentication (will fail without real credentials)
    print("\n3. Testing authentication...")
    try:
        credentials = gke_client.credentials
        print("✅ Successfully authenticated with Google Cloud")
        print(f"   Credentials type: {type(credentials).__name__}")

        # Example 4: Get cluster information
        print("\n4. Fetching cluster information...")
        cluster_info = gke_client.cluster_info
        print("✅ Successfully retrieved cluster information:")
        print(f"   Name: {cluster_info.name}")
        print(f"   Status: {cluster_info.status}")
        print(f"   Endpoint: {cluster_info.endpoint}")
        print(f"   Node Count: {cluster_info.node_count}")
        print(f"   Kubernetes Version: {cluster_info.kubernetes_version}")
        print(f"   Network: {cluster_info.network}")

        # Example 5: Validate connection
        print("\n5. Validating cluster connection...")
        is_valid = gke_client.validate_connection()
        if is_valid:
            print("✅ Cluster connection is valid and ready")
        else:
            print("⚠️  Cluster connection has issues (check logs)")

        # Example 6: List namespaces
        print("\n6. Listing available namespaces...")
        namespaces = gke_client.list_namespaces()
        print(f"✅ Found {len(namespaces)} namespaces:")
        for ns in namespaces[:5]:  # Show first 5
            print(f"   - {ns}")
        if len(namespaces) > 5:
            print(f"   ... and {len(namespaces) - 5} more")

        # Example 7: Get cluster nodes
        print("\n7. Getting cluster node information...")
        nodes = gke_client.get_cluster_nodes()
        print(f"✅ Found {len(nodes)} nodes:")
        for node in nodes[:3]:  # Show first 3
            print(f"   - {node['name']}: {node['status']} ({node['version']})")
        if len(nodes) > 3:
            print(f"   ... and {len(nodes) - 3} more")

    except GKEConnectionError as e:
        print(f"🔧 Expected connection error (no real credentials): {e}")
        print("\nTo test with real GKE cluster:")
        print("1. Set up Google Cloud authentication")
        print("2. Update cluster_name and project_id above")
        print("3. Ensure the cluster exists and you have access")

    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        # Clean up
        print("\n8. Cleaning up...")
        try:
            gke_client.close()
            print("✅ Client resources cleaned up")
        except BaseException:
            pass

    print("\n" + "=" * 40)
    print("Demo completed! Check the logs above for details.")


if __name__ == "__main__":
    demo_gke_client()
