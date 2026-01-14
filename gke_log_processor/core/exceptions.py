"""Custom exceptions for GKE Log Processor."""


class GKELogProcessorError(Exception):
    """Base exception for GKE Log Processor."""

    pass


class ConfigurationError(GKELogProcessorError):
    """Raised when there's a configuration error."""

    pass


class GKEConnectionError(GKELogProcessorError):
    """Raised when there's an error connecting to GKE."""

    pass


class KubernetesError(GKELogProcessorError):
    """Raised when there's a Kubernetes API error."""

    pass


class KubernetesConnectionError(KubernetesError):
    """Raised when there's a connection error to Kubernetes cluster."""

    pass


class PodNotFoundError(KubernetesError):
    """Raised when a requested pod is not found."""

    pass


class AIServiceError(GKELogProcessorError):
    """Raised when there's an error with the AI service."""

    pass


class LogProcessingError(GKELogProcessorError):
    """Raised when there's an error processing logs."""

    pass
