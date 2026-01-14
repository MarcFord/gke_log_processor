"""GKE Log Processor.

A CLI application for monitoring GKE pod logs with AI-powered analysis.
"""

__version__ = "0.1.0"
__author__ = "Marc Ford"
__description__ = (
    "A CLI application for monitoring GKE pod logs with AI-powered analysis"
)

from .core.config import Config
from .core.exceptions import GKELogProcessorError

__all__ = [
    "Config",
    "GKELogProcessorError",
    "__version__",
    "__author__",
    "__description__",
]
