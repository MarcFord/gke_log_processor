# GKE Client Documentation

The `GKEClient` class provides a high-level interface for connecting to and managing Google Kubernetes Engine (GKE) clusters.

## Features

- **Automatic Authentication**: Uses Google Cloud Application Default Credentials
- **Cluster Validation**: Verifies cluster exists and is accessible
- **Kubernetes Integration**: Creates authenticated Kubernetes API clients
- **Connection Management**: Handles both zonal and regional clusters
- **Error Handling**: Comprehensive error handling with custom exceptions

## Basic Usage

```python
from gke_log_processor.core.config import Config
from gke_log_processor.gke.client import GKEClient

# Create configuration
config = Config(
    cluster_name="my-cluster",
    project_id="my-project",
    zone="us-central1-a",  # or region="us-central1"
    namespace="default"
)

# Initialize client
gke_client = GKEClient(config)

# Validate connection
try:
    is_valid = gke_client.validate_connection()
    if is_valid:
        print("Connected successfully!")
        
        # Get cluster information
        cluster_info = gke_client.cluster_info
        print(f"Cluster: {cluster_info.name}")
        print(f"Status: {cluster_info.status}")
        print(f"Nodes: {cluster_info.node_count}")
        
        # List namespaces
        namespaces = gke_client.list_namespaces()
        print(f"Namespaces: {namespaces}")
        
finally:
    gke_client.close()
```

## Authentication Setup

### Option 1: Application Default Credentials (Recommended)
```bash
gcloud auth application-default login
```

### Option 2: Service Account Key
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Option 3: Compute Engine/Cloud Shell
Authentication is automatic when running on GCP compute resources.

## Configuration

### Zonal Cluster
```python
config = Config(
    cluster_name="my-cluster",
    project_id="my-project",
    zone="us-central1-a"
)
```

### Regional Cluster
```python
config = Config(
    cluster_name="my-cluster", 
    project_id="my-project",
    region="us-central1"
)
```

## Properties

### `credentials`
Returns Google Cloud credentials used for authentication.

### `container_client`
Returns authenticated Google Cloud Container API client.

### `kubernetes_client`
Returns authenticated Kubernetes API client.

### `cluster_info`
Returns detailed cluster information as a `ClusterInfo` object:
- `name`: Cluster name
- `location`: Zone or region
- `project_id`: GCP project ID
- `endpoint`: Kubernetes API server endpoint
- `status`: Cluster status (RUNNING, STOPPED, etc.)
- `node_count`: Number of nodes
- `kubernetes_version`: Kubernetes version
- `is_regional`: Whether cluster is regional

## Methods

### `validate_connection() -> bool`
Validates that the cluster is accessible and in a running state.
Tests Kubernetes API connectivity and namespace access.

### `list_namespaces() -> List[str]`
Returns list of all namespaces in the cluster.

### `get_cluster_nodes() -> List[Dict[str, Any]]`
Returns detailed information about all cluster nodes.

### `close()`
Cleans up client resources. Should be called when done.

## Error Handling

The client raises custom exceptions for different error scenarios:

- `GKEConnectionError`: Authentication, cluster access, or API errors
- `ConfigurationError`: Invalid configuration parameters

```python
try:
    gke_client = GKEClient(config)
    gke_client.validate_connection()
except GKEConnectionError as e:
    print(f"Connection failed: {e}")
except ConfigurationError as e:
    print(f"Configuration invalid: {e}")
```

## Best Practices

1. **Always call `close()`** when done with the client
2. **Use context managers** for automatic cleanup:
   ```python
   class GKEClientContext:
       def __init__(self, config):
           self.config = config
           self.client = None
           
       def __enter__(self):
           self.client = GKEClient(self.config)
           return self.client
           
       def __exit__(self, *args):
           if self.client:
               self.client.close()
   ```

3. **Check permissions** before attempting operations
4. **Handle authentication errors** gracefully
5. **Cache cluster info** - it's automatically cached by the client
6. **Use proper logging** to debug connection issues

## Examples

See `examples/gke_client_demo.py` for a comprehensive usage example.