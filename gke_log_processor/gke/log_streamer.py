"""
Real-time log streaming from Kubernetes pods.

This module provides functionality for streaming logs from multiple pods
with intelligent buffering, rate limiting, and error handling.
"""

import asyncio
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Callable, Dict, List, Optional

from kubernetes.stream import stream  # type: ignore[import-untyped]

from ..core.exceptions import LogProcessingError
from .kubernetes_client import KubernetesClient, PodInfo

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log severity levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


@dataclass
class LogEntry:
    """Represents a single log entry from a pod."""

    timestamp: datetime
    pod_name: str
    namespace: str
    container_name: str
    message: str
    level: LogLevel = LogLevel.INFO
    raw_line: str = ""

    def __post_init__(self):
        """Post-process the log entry."""
        if not self.raw_line:
            self.raw_line = self.message

        # Try to detect log level from message if it's the default
        if self.level == LogLevel.INFO:
            self.level = self._detect_log_level()

    def _detect_log_level(self) -> LogLevel:
        """Detect log level from the message content."""
        message_upper = self.message.upper()

        if any(keyword in message_upper for keyword in ["FATAL", "CRITICAL"]):
            return LogLevel.FATAL
        elif any(keyword in message_upper for keyword in ["ERROR", "ERR"]):
            return LogLevel.ERROR
        elif any(keyword in message_upper for keyword in ["WARN", "WARNING"]):
            return LogLevel.WARN
        elif any(keyword in message_upper for keyword in ["DEBUG", "DBG"]):
            return LogLevel.DEBUG
        elif any(keyword in message_upper for keyword in ["TRACE"]):
            return LogLevel.TRACE
        else:
            return LogLevel.INFO

    @property
    def severity_score(self) -> int:
        """Get numeric severity score for sorting/filtering."""
        scores = {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 2,
            LogLevel.WARN: 3,
            LogLevel.ERROR: 4,
            LogLevel.FATAL: 5,
        }
        return scores.get(self.level, 2)

    @property
    def is_error(self) -> bool:
        """Check if this is an error-level log entry."""
        return self.level in [LogLevel.ERROR, LogLevel.FATAL]

    @property
    def formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def __str__(self) -> str:
        """String representation of the log entry."""
        return (
            f"[{self.formatted_timestamp}] "
            f"{self.pod_name}/{self.container_name} "
            f"[{self.level.value}] {self.message}"
        )


@dataclass
class StreamConfig:
    """Configuration for log streaming."""

    # Buffer settings
    max_buffer_size: int = 1000
    buffer_flush_interval: float = 1.0  # seconds

    # Rate limiting
    max_logs_per_second: float = 100.0
    rate_limit_window: float = 1.0  # seconds

    # Log filtering
    min_log_level: LogLevel = LogLevel.INFO
    follow_logs: bool = True
    tail_lines: Optional[int] = 100

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds

    # Include timestamps
    timestamps: bool = True

    # Timeout settings
    stream_timeout: float = 300.0  # 5 minutes


class LogBuffer:
    """Thread-safe log buffer with automatic flushing."""

    def __init__(self, max_size: int = 1000, flush_interval: float = 1.0):
        """Initialize the log buffer."""
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._last_flush = time.time()
        self._callbacks: List[Callable[[List[LogEntry]], None]] = []

    async def add(self, entry: LogEntry) -> None:
        """Add a log entry to the buffer."""
        async with self._lock:
            self._buffer.append(entry)

            # Check if we need to flush
            current_time = time.time()
            should_flush = (
                len(self._buffer) >= self.max_size
                or (current_time - self._last_flush) >= self.flush_interval
            )

            if should_flush:
                await self._flush()

    async def _flush(self) -> None:
        """Flush the buffer to all registered callbacks."""
        if not self._buffer:
            return

        # Copy buffer contents
        entries = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()

        # Call all callbacks
        for callback in self._callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(entries)
                else:
                    callback(entries)
            except Exception as e:
                logger.error(f"Error in buffer callback: {e}")

    def add_callback(self, callback: Callable[[List[LogEntry]], None]) -> None:
        """Add a callback to receive flushed log entries."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[List[LogEntry]], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def force_flush(self) -> None:
        """Force flush the buffer."""
        async with self._lock:
            await self._flush()

    def get_current_entries(self) -> List[LogEntry]:
        """Get current buffer contents without flushing."""
        return list(self._buffer)


class RateLimiter:
    """Rate limiter for log streaming."""

    def __init__(self, max_rate: float, window: float = 1.0):
        """Initialize rate limiter."""
        self.max_rate = max_rate
        self.window = window
        self._timestamps: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a rate limit token."""
        async with self._lock:
            current_time = time.time()

            # Remove old timestamps outside the window
            while self._timestamps and current_time - self._timestamps[0] > self.window:
                self._timestamps.popleft()

            # Check if we can proceed
            if len(self._timestamps) < self.max_rate:
                self._timestamps.append(current_time)
                return True

            return False

    async def wait_if_needed(self) -> None:
        """Wait if rate limit is exceeded."""
        while not await self.acquire():
            await asyncio.sleep(0.01)  # Small delay before retry


class LogStreamer:
    """Real-time log streaming from Kubernetes pods."""

    def __init__(
        self, kubernetes_client: KubernetesClient, config: Optional[StreamConfig] = None
    ):
        """Initialize the log streamer."""
        self.k8s_client = kubernetes_client
        self.config = config or StreamConfig()

        # Internal state
        self._active_streams: Dict[str, asyncio.Task] = {}
        self._buffers: Dict[str, LogBuffer] = {}
        self._rate_limiter = RateLimiter(
            self.config.max_logs_per_second, self.config.rate_limit_window
        )
        self._shutdown_event = asyncio.Event()

        logger.info(f"LogStreamer initialized with config: {self.config}")

    async def stream_pod_logs(
        self, pod_info: PodInfo, container_name: Optional[str] = None
    ) -> AsyncGenerator[LogEntry, None]:
        """
        Stream logs from a specific pod and container.

        Args:
            pod_info: Pod to stream logs from.
            container_name: Specific container (if None, uses first container).

        Yields:
            LogEntry objects for each log line.

        Raises:
            LogProcessingError: If log streaming fails.
        """
        target_container = container_name or pod_info.containers[0]
        stream_id = f"{pod_info.namespace}/{pod_info.name}/{target_container}"

        logger.info(f"Starting log stream for {stream_id}")

        try:
            api = self._get_k8s_api()

            # Configure stream parameters
            stream_params = {
                "name": pod_info.name,
                "namespace": pod_info.namespace,
                "container": target_container,
                "follow": self.config.follow_logs,
                "timestamps": self.config.timestamps,
                "_preload_content": False,
            }

            if self.config.tail_lines:
                stream_params["tail_lines"] = self.config.tail_lines

            # Start streaming
            log_stream = stream(api.read_namespaced_pod_log, **stream_params)

            # Process log lines
            for line in log_stream:
                if self._shutdown_event.is_set():
                    break

                # Rate limiting
                await self._rate_limiter.wait_if_needed()

                # Parse log line
                try:
                    log_entry = self._parse_log_line(
                        line.strip(),
                        pod_info.name,
                        pod_info.namespace,
                        target_container,
                    )

                    # Filter by log level - compare severity scores
                    min_severity = {
                        LogLevel.TRACE: 0,
                        LogLevel.DEBUG: 1,
                        LogLevel.INFO: 2,
                        LogLevel.WARN: 3,
                        LogLevel.ERROR: 4,
                        LogLevel.FATAL: 5,
                    }.get(self.config.min_log_level, 2)

                    if log_entry.severity_score >= min_severity:
                        yield log_entry

                except Exception as e:
                    logger.warning(f"Failed to parse log line: {e}")
                    continue

            logger.info(f"Log stream ended for {stream_id}")

        except Exception as e:
            logger.error(f"Error streaming logs from {stream_id}: {e}")
            raise LogProcessingError(
                f"Failed to stream logs from {stream_id}: {e}"
            ) from e

    async def stream_multiple_pods(
        self, pods: List[PodInfo], container_name: Optional[str] = None
    ) -> AsyncGenerator[LogEntry, None]:
        """
        Stream logs from multiple pods simultaneously.

        Args:
            pods: List of pods to stream from.
            container_name: Specific container to stream from (optional).

        Yields:
            LogEntry objects from all pods.
        """
        if not pods:
            logger.warning("No pods provided for streaming")
            return

        logger.info(f"Starting multi-pod log streaming for {len(pods)} pods")

        # Create async generators for each pod
        pod_streams = []
        for pod_info in pods:
            try:
                stream_gen = self.stream_pod_logs(pod_info, container_name)
                pod_streams.append(stream_gen)
            except Exception as e:
                logger.error(f"Failed to create stream for pod {pod_info.name}: {e}")
                continue

        if not pod_streams:
            raise LogProcessingError("No valid pod streams could be created")

        # Merge streams using async generators
        try:
            async for log_entry in self._merge_log_streams(pod_streams):
                yield log_entry
        except Exception as e:
            logger.error(f"Error in multi-pod streaming: {e}")
            raise LogProcessingError(f"Multi-pod streaming failed: {e}") from e

    async def _merge_log_streams(
        self, streams: List[AsyncGenerator]
    ) -> AsyncGenerator[LogEntry, None]:
        """Merge multiple async log streams into a single stream."""
        # Create tasks for each stream
        stream_tasks = []
        stream_queues = []

        for i, log_stream in enumerate(streams):
            queue: asyncio.Queue = asyncio.Queue()
            stream_queues.append(queue)

            task = asyncio.create_task(
                self._stream_to_queue(log_stream, queue, f"stream_{i}")
            )
            stream_tasks.append(task)

        try:
            # Continuously yield from all queues
            active_queues = set(range(len(stream_queues)))

            while active_queues and not self._shutdown_event.is_set():
                # Wait for any queue to have data
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(queue.get())
                        for i, queue in enumerate(stream_queues)
                        if i in active_queues
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0,
                )

                for task in done:
                    try:
                        log_entry = task.result()
                        if log_entry is None:  # End of stream marker
                            continue
                        yield log_entry
                    except Exception as e:
                        logger.error(f"Error getting log entry from merged stream: {e}")

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

        finally:
            # Clean up stream tasks
            for task in stream_tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*stream_tasks, return_exceptions=True)

    async def _stream_to_queue(
        self, log_stream: AsyncGenerator, queue: asyncio.Queue, stream_id: str
    ) -> None:
        """Feed a stream into a queue."""
        try:
            async for log_entry in log_stream:
                await queue.put(log_entry)
        except Exception as e:
            logger.error(f"Error in stream {stream_id}: {e}")
        finally:
            await queue.put(None)  # End of stream marker

    def _get_k8s_api(self):
        """Get the Kubernetes API client."""
        # Using protected member access as this is within the same module context
        return self.k8s_client._get_api()  # pylint: disable=protected-access

    def _parse_log_line(
        self, line: str, pod_name: str, namespace: str, container_name: str
    ) -> LogEntry:
        """Parse a log line into a LogEntry object."""
        timestamp = datetime.now(timezone.utc)
        message = line

        # Try to extract timestamp if present
        if self.config.timestamps and line:
            # Common timestamp formats
            timestamp_formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
            ]

            for fmt in timestamp_formats:
                try:
                    # Look for timestamp at the beginning of the line
                    if " " in line:
                        timestamp_str = line.split(" ", 1)[0]
                        parsed_time = datetime.strptime(timestamp_str, fmt)
                        if parsed_time.tzinfo is None:
                            parsed_time = parsed_time.replace(tzinfo=timezone.utc)
                        timestamp = parsed_time
                        message = (
                            line.split(" ", 1)[1] if len(line.split(" ", 1)) > 1 else ""
                        )
                        break
                except ValueError:
                    continue

        return LogEntry(
            timestamp=timestamp,
            pod_name=pod_name,
            namespace=namespace,
            container_name=container_name,
            message=message,
            raw_line=line,
        )

    async def start_buffered_streaming(
        self,
        pods: List[PodInfo],
        callback: Callable[[List[LogEntry]], None],
        container_name: Optional[str] = None,
    ) -> str:
        """
        Start buffered log streaming from multiple pods.

        Args:
            pods: Pods to stream from.
            callback: Function to call with buffered log entries.
            container_name: Specific container to stream from.

        Returns:
            Stream ID for managing the stream.
        """
        if not pods:
            raise LogProcessingError("No pods provided for buffered streaming")

        stream_id = (
            f"buffered_{hash(tuple(pod.name for pod in pods))}_{int(time.time())}"
        )

        # Create buffer
        buffer = LogBuffer(
            max_size=self.config.max_buffer_size,
            flush_interval=self.config.buffer_flush_interval,
        )
        buffer.add_callback(callback)
        self._buffers[stream_id] = buffer

        # Start streaming task
        task = asyncio.create_task(
            self._run_buffered_stream(stream_id, pods, buffer, container_name)
        )
        self._active_streams[stream_id] = task

        logger.info(f"Started buffered streaming with ID: {stream_id}")
        return stream_id

    async def _run_buffered_stream(
        self,
        stream_id: str,
        pods: List[PodInfo],
        buffer: LogBuffer,
        container_name: Optional[str],
    ) -> None:
        """Run a buffered streaming session."""
        try:
            async for log_entry in self.stream_multiple_pods(pods, container_name):
                await buffer.add(log_entry)

                if self._shutdown_event.is_set():
                    break

        except Exception as e:
            logger.error(f"Error in buffered stream {stream_id}: {e}")
        finally:
            # Flush any remaining logs
            await buffer.force_flush()
            logger.info(f"Buffered stream {stream_id} ended")

    async def stop_stream(self, stream_id: str) -> None:
        """Stop a specific streaming session."""
        if stream_id in self._active_streams:
            task = self._active_streams[stream_id]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            del self._active_streams[stream_id]

            if stream_id in self._buffers:
                await self._buffers[stream_id].force_flush()
                del self._buffers[stream_id]

            logger.info(f"Stopped stream: {stream_id}")

    async def stop_all_streams(self) -> None:
        """Stop all active streaming sessions."""
        logger.info("Stopping all log streams")
        self._shutdown_event.set()

        # Cancel all active streams
        for stream_id in list(self._active_streams.keys()):
            await self.stop_stream(stream_id)

        logger.info("All log streams stopped")

    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs."""
        return list(self._active_streams.keys())

    async def get_recent_logs(
        self,
        pods: List[PodInfo],
        lines: int = 100,
        container_name: Optional[str] = None,
    ) -> List[LogEntry]:
        """
        Get recent logs from pods (non-streaming).

        Args:
            pods: Pods to get logs from.
            lines: Number of recent lines to fetch.
            container_name: Specific container to get logs from.

        Returns:
            List of recent log entries.
        """
        if not pods:
            return []

        all_logs = []

        for pod_info in pods:
            try:
                target_container = container_name or pod_info.containers[0]
                api = self._get_k8s_api()

                log_lines = api.read_namespaced_pod_log(
                    name=pod_info.name,
                    namespace=pod_info.namespace,
                    container=target_container,
                    tail_lines=lines,
                    timestamps=True,
                )

                # Parse log lines
                for line in log_lines.strip().split("\n"):
                    if line.strip():
                        try:
                            log_entry = self._parse_log_line(
                                line.strip(),
                                pod_info.name,
                                pod_info.namespace,
                                target_container,
                            )
                            all_logs.append(log_entry)
                        except Exception as e:
                            logger.warning(f"Failed to parse log line: {e}")
                            continue

            except Exception as e:
                logger.error(f"Failed to get logs from pod {pod_info.name}: {e}")
                continue

        # Sort by timestamp
        all_logs.sort(key=lambda x: x.timestamp)
        return all_logs[-lines:] if len(all_logs) > lines else all_logs
