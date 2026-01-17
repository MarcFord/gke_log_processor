"""Data models for GKE Log Processor using Pydantic."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from .utils import utc_now


class PodPhase(str, Enum):
    """Kubernetes Pod phases."""

    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class ContainerState(str, Enum):
    """Container states within a Pod."""

    WAITING = "Waiting"
    RUNNING = "Running"
    TERMINATED = "Terminated"


class LogLevel(str, Enum):
    """Log severity levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"


class SeverityLevel(str, Enum):
    """AI-analyzed severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternType(str, Enum):
    """Types of patterns detected in logs."""

    ERROR_PATTERN = "error_pattern"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_CONCERN = "security_concern"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ISSUE = "network_issue"
    DATABASE_ISSUE = "database_issue"
    AUTHENTICATION_FAILURE = "authentication_failure"
    TIMEOUT = "timeout"
    CRASH = "crash"
    STARTUP_ISSUE = "startup_issue"
    CONFIGURATION_ERROR = "configuration_error"


class ContainerStatus(BaseModel):
    """Model for container status within a Pod."""

    name: str = Field(..., description="Container name")
    image: str = Field(..., description="Container image")
    state: ContainerState = Field(..., description="Current container state")
    ready: bool = Field(..., description="Whether container is ready")
    restart_count: int = Field(0, description="Number of times container has restarted")
    started_at: Optional[datetime] = Field(None, description="Container start time")
    finished_at: Optional[datetime] = Field(None, description="Container finish time")
    exit_code: Optional[int] = Field(
        None, description="Container exit code if terminated"
    )
    reason: Optional[str] = Field(None, description="Reason for current state")
    message: Optional[str] = Field(None, description="Detailed message about state")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_healthy(self) -> bool:
        """Check if the container is in a healthy state."""
        return self.state == ContainerState.RUNNING and self.ready

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uptime_seconds(self) -> Optional[int]:
        """Calculate container uptime in seconds."""
        if self.started_at and self.state == ContainerState.RUNNING:
            return int((utc_now() - self.started_at).total_seconds())
        return None


class PodCondition(BaseModel):
    """Model for Pod conditions."""

    type: str = Field(..., description="Condition type (e.g., Ready, PodScheduled)")
    status: str = Field(..., description="Condition status (True/False/Unknown)")
    last_probe_time: Optional[datetime] = Field(None, description="Last probe time")
    last_transition_time: Optional[datetime] = Field(
        None, description="Last transition time"
    )
    reason: Optional[str] = Field(None, description="Reason for condition")
    message: Optional[str] = Field(None, description="Human-readable message")


class PodInfo(BaseModel):
    """Model for Kubernetes Pod information."""

    name: str = Field(..., description="Pod name")
    namespace: str = Field(..., description="Pod namespace")
    cluster: str = Field(..., description="Cluster name")
    uid: str = Field(..., description="Pod UID")
    phase: PodPhase = Field(..., description="Pod phase")
    node_name: Optional[str] = Field(None, description="Node where pod is scheduled")
    pod_ip: Optional[str] = Field(None, description="Pod IP address")
    host_ip: Optional[str] = Field(None, description="Host IP address")
    created_at: datetime = Field(..., description="Pod creation time")
    started_at: Optional[datetime] = Field(None, description="Pod start time")
    labels: Dict[str, str] = Field(default_factory=dict, description="Pod labels")
    annotations: Dict[str, str] = Field(
        default_factory=dict, description="Pod annotations"
    )
    containers: List[ContainerStatus] = Field(
        default_factory=list, description="Container statuses"
    )
    conditions: List[PodCondition] = Field(
        default_factory=list, description="Pod conditions"
    )
    owner_references: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pod owner references"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate pod name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Pod name must be a non-empty string")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_ready(self) -> bool:
        """Check if the pod is ready."""
        ready_condition = next((c for c in self.conditions if c.type == "Ready"), None)
        return ready_condition is not None and ready_condition.status == "True"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ready_containers(self) -> int:
        """Count of ready containers."""
        return sum(1 for container in self.containers if container.ready)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_containers(self) -> int:
        """Total number of containers."""
        return len(self.containers)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def container_ready_ratio(self) -> str:
        """Container readiness ratio as string (e.g., '2/3')."""
        return f"{self.ready_containers}/{self.total_containers}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_seconds(self) -> int:
        """Pod age in seconds."""
        return int((utc_now() - self.created_at).total_seconds())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def controller_name(self) -> Optional[str]:
        """Get the name of the controlling resource (Deployment, StatefulSet, etc.)."""
        if self.owner_references:
            owner = self.owner_references[0]
            return owner.get("name")
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def controller_kind(self) -> Optional[str]:
        """Get the kind of the controlling resource."""
        if self.owner_references:
            owner = self.owner_references[0]
            return owner.get("kind")
        return None

    def get_container(self, name: str) -> Optional[ContainerStatus]:
        """Get container status by name."""
        return next((c for c in self.containers if c.name == name), None)

    def has_label(self, key: str, value: Optional[str] = None) -> bool:
        """Check if pod has a specific label."""
        if value is None:
            return key in self.labels
        return self.labels.get(key) == value


class LogEntry(BaseModel):
    """Model for a single log entry."""

    id: UUID = Field(default_factory=uuid4, description="Unique log entry ID")
    timestamp: datetime = Field(..., description="Log timestamp")
    message: str = Field(..., description="Log message content")
    level: Optional[LogLevel] = Field(None, description="Detected log level")
    source: str = Field(..., description="Log source (container name)")
    pod_name: str = Field(..., description="Pod name")
    namespace: str = Field(..., description="Namespace")
    cluster: str = Field(..., description="Cluster name")
    container_name: str = Field(..., description="Container name")
    stream: str = Field("stdout", description="Log stream (stdout/stderr)")
    raw_message: str = Field(..., description="Original unprocessed message")
    parsed_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed structured fields"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context information"
    )
    tags: List[str] = Field(default_factory=list, description="User-defined tags")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Validate log message."""
        if not isinstance(v, str):
            raise ValueError("Log message must be a string")
        return v

    @field_validator("level", mode="before")
    @classmethod
    def validate_level(cls, v):
        """Normalize log level."""
        if v is None:
            return None
        if isinstance(v, str):
            # Try to match common log level patterns
            v_upper = v.upper()
            if v_upper in ["WARN", "WARNING"]:
                return LogLevel.WARNING.value
            elif v_upper in ["ERR", "ERROR"]:
                return LogLevel.ERROR.value
            elif v_upper in ["FATAL", "CRITICAL"]:
                return LogLevel.CRITICAL.value
            try:
                return LogLevel(v_upper).value
            except ValueError:
                return None
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_error(self) -> bool:
        """Check if this is an error-level log."""
        return self.level in [LogLevel.ERROR, LogLevel.FATAL, LogLevel.CRITICAL]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_warning(self) -> bool:
        """Check if this is a warning-level log."""
        return self.level in [LogLevel.WARN, LogLevel.WARNING]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity_score(self) -> int:
        """Get numeric severity score (higher = more severe)."""
        severity_map = {
            LogLevel.TRACE: 1,
            LogLevel.DEBUG: 2,
            LogLevel.INFO: 3,
            LogLevel.WARN: 4,
            LogLevel.WARNING: 4,
            LogLevel.ERROR: 5,
            LogLevel.FATAL: 6,
            LogLevel.CRITICAL: 6,
        }
        return severity_map.get(self.level, 0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_identifier(self) -> str:
        """Get unique source identifier."""
        return f"{self.cluster}/{self.namespace}/{self.pod_name}/{self.container_name}"

    def add_tag(self, tag: str) -> None:
        """Add a tag to the log entry."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the log entry."""
        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if log entry has a specific tag."""
        return tag in self.tags


class DetectedPattern(BaseModel):
    """Model for detected patterns in logs."""

    id: UUID = Field(default_factory=uuid4, description="Unique pattern ID")
    type: PatternType = Field(..., description="Type of pattern detected")
    pattern: str = Field(..., description="Pattern description or regex")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    first_seen: datetime = Field(..., description="First occurrence")
    last_seen: datetime = Field(..., description="Last occurrence")
    occurrence_count: int = Field(1, description="Number of occurrences")
    affected_pods: List[str] = Field(
        default_factory=list, description="List of affected pod names"
    )
    sample_messages: List[str] = Field(
        default_factory=list, description="Sample log messages matching pattern"
    )
    severity: SeverityLevel = Field(..., description="AI-assessed severity")
    recommendation: Optional[str] = Field(None, description="Recommended action or fix")
    related_patterns: List[UUID] = Field(
        default_factory=list, description="Related pattern IDs"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def add_occurrence(self, pod_name: str, message: str) -> None:
        """Record a new occurrence of this pattern."""
        self.occurrence_count += 1
        self.last_seen = utc_now()

        if pod_name not in self.affected_pods:
            self.affected_pods.append(pod_name)

        # Keep only recent sample messages (max 5)
        if len(self.sample_messages) >= 5:
            self.sample_messages.pop(0)
        self.sample_messages.append(message)


class AIAnalysisResult(BaseModel):
    """Model for AI analysis results of logs."""

    id: UUID = Field(default_factory=uuid4, description="Unique analysis ID")
    analysis_timestamp: datetime = Field(
        default_factory=utc_now, description="When analysis was performed"
    )
    log_entries_analyzed: int = Field(..., description="Number of log entries analyzed")
    time_window_start: datetime = Field(..., description="Analysis time window start")
    time_window_end: datetime = Field(..., description="Analysis time window end")
    overall_severity: SeverityLevel = Field(
        ..., description="Overall severity assessment"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Analysis confidence (0.0-1.0)"
    )
    detected_patterns: List[DetectedPattern] = Field(
        default_factory=list, description="Detected log patterns"
    )
    severity_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count of logs by severity level"
    )
    error_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Percentage of error logs"
    )
    warning_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Percentage of warning logs"
    )
    top_error_messages: List[str] = Field(
        default_factory=list, description="Most frequent error messages"
    )
    anomalies: List[str] = Field(default_factory=list, description="Detected anomalies")
    recommendations: List[str] = Field(
        default_factory=list, description="AI-generated recommendations"
    )
    summary: Optional[str] = Field(None, description="AI-generated summary of logs")
    tags: List[str] = Field(default_factory=list, description="Analysis tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional analysis metadata"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def analysis_duration_seconds(self) -> int:
        """Duration of analyzed time window in seconds."""
        return int((self.time_window_end - self.time_window_start).total_seconds())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def critical_patterns(self) -> List[DetectedPattern]:
        """Get patterns with critical severity."""
        return [
            p for p in self.detected_patterns if p.severity == SeverityLevel.CRITICAL
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def high_confidence_patterns(self) -> List[DetectedPattern]:
        """Get patterns with high confidence (>0.8)."""
        return [p for p in self.detected_patterns if p.confidence > 0.8]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_immediate_attention(self) -> bool:
        """Check if analysis results require immediate attention."""
        return (
            self.overall_severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
            or self.error_rate > 0.1  # More than 10% error rate
            or len(self.critical_patterns) > 0
        )

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[DetectedPattern]:
        """Get patterns of a specific type."""
        return [p for p in self.detected_patterns if p.type == pattern_type]

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the analysis."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)


class LogSummary(BaseModel):
    """Model for log summary over a time period."""

    id: UUID = Field(default_factory=uuid4, description="Unique summary ID")
    created_at: datetime = Field(
        default_factory=utc_now, description="Summary creation time"
    )
    time_window_start: datetime = Field(..., description="Summary time window start")
    time_window_end: datetime = Field(..., description="Summary time window end")
    total_log_count: int = Field(0, description="Total number of logs")
    pod_count: int = Field(0, description="Number of unique pods")
    namespace_count: int = Field(0, description="Number of unique namespaces")
    container_count: int = Field(0, description="Number of unique containers")
    log_level_counts: Dict[str, int] = Field(
        default_factory=dict, description="Count by log level"
    )
    top_pods_by_volume: List[Dict[str, Union[str, int]]] = Field(
        default_factory=list, description="Pods with most log volume"
    )
    error_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Error analysis summary"
    )
    ai_insights: Optional[AIAnalysisResult] = Field(
        None, description="AI analysis results if available"
    )
    key_events: List[str] = Field(
        default_factory=list, description="Notable events during time window"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary_duration_minutes(self) -> float:
        """Duration of summary window in minutes."""
        return (self.time_window_end - self.time_window_start).total_seconds() / 60

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logs_per_minute(self) -> float:
        """Average logs per minute."""
        duration = self.summary_duration_minutes
        return self.total_log_count / duration if duration > 0 else 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_percentage(self) -> float:
        """Percentage of error-level logs."""
        error_count = self.log_level_counts.get("ERROR", 0) + self.log_level_counts.get(
            "CRITICAL", 0
        )
        return (
            (error_count / self.total_log_count * 100)
            if self.total_log_count > 0
            else 0
        )


class StreamingStats(BaseModel):
    """Model for real-time streaming statistics."""

    session_id: UUID = Field(default_factory=uuid4, description="Streaming session ID")
    start_time: datetime = Field(
        default_factory=utc_now, description="Streaming start time"
    )
    last_update: datetime = Field(
        default_factory=utc_now, description="Last statistics update"
    )
    total_logs_received: int = Field(0, description="Total logs received")
    logs_per_second: float = Field(0.0, description="Current logs per second rate")
    bytes_received: int = Field(0, description="Total bytes received")
    active_pod_count: int = Field(0, description="Number of actively logging pods")
    buffer_size: int = Field(0, description="Current buffer size")
    max_buffer_size: int = Field(1000, description="Maximum buffer size")
    dropped_logs_count: int = Field(0, description="Number of dropped logs")
    connection_status: Dict[str, str] = Field(
        default_factory=dict, description="Connection status per pod"
    )
    error_count: int = Field(0, description="Number of streaming errors")
    reconnection_count: int = Field(0, description="Number of reconnections")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uptime_seconds(self) -> float:
        """Streaming session uptime in seconds."""
        return (self.last_update - self.start_time).total_seconds()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def buffer_utilization(self) -> float:
        """Buffer utilization as percentage."""
        return (
            (self.buffer_size / self.max_buffer_size * 100)
            if self.max_buffer_size > 0
            else 0
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def drop_rate(self) -> float:
        """Log drop rate as percentage."""
        total_expected = self.total_logs_received + self.dropped_logs_count
        return (
            (self.dropped_logs_count / total_expected * 100)
            if total_expected > 0
            else 0
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_logs_per_second(self) -> float:
        """Average logs per second since start."""
        uptime = self.uptime_seconds
        return self.total_logs_received / uptime if uptime > 0 else 0

    def update_stats(
        self,
        new_logs: int,
        new_bytes: int,
        current_buffer_size: int,
        active_pods: int,
        current_rate: float,
    ) -> None:
        """Update streaming statistics."""
        self.last_update = utc_now()
        self.total_logs_received += new_logs
        self.bytes_received += new_bytes
        self.buffer_size = current_buffer_size
        self.active_pod_count = active_pods
        self.logs_per_second = current_rate


class QueryType(str, Enum):
    """Types of custom queries."""

    SEARCH = "search"  # Simple search queries
    ANALYSIS = "analysis"  # Analytical questions
    TROUBLESHOOTING = "troubleshooting"  # Problem-solving queries
    METRICS = "metrics"  # Performance and usage metrics
    PATTERNS = "patterns"  # Pattern-based queries
    CUSTOM = "custom"  # User-defined query types


class QueryRequest(BaseModel):
    """Model for custom query requests."""

    id: UUID = Field(default_factory=uuid4, description="Unique query ID")
    question: str = Field(..., description="Natural language question", min_length=1)
    query_type: QueryType = Field(default=QueryType.ANALYSIS, description="Type of query")
    context_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional filters (pod names, time ranges, log levels, etc.)"
    )
    max_log_entries: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of log entries to analyze"
    )
    enable_pattern_matching: bool = Field(
        default=True,
        description="Whether to include pattern detection in the query"
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include analysis context in the response"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Query creation time")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question format."""
        if len(v.strip()) < 3:
            raise ValueError("Question must be at least 3 characters long")
        return v.strip()


class QueryResponse(BaseModel):
    """Model for custom query responses."""

    request_id: UUID = Field(..., description="ID of the original request")
    answer: str = Field(..., description="AI-generated answer to the query")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer (0.0-1.0)"
    )
    sources_analyzed: int = Field(..., description="Number of log entries analyzed")
    query_duration_seconds: float = Field(..., description="Time taken to process query")
    related_patterns: List[str] = Field(
        default_factory=list,
        description="Related patterns found during analysis"
    )
    suggested_followups: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional query metadata"
    )
    generated_at: datetime = Field(
        default_factory=utc_now,
        description="Response generation time"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_high_confidence(self) -> bool:
        """Check if response has high confidence."""
        return self.confidence_score >= 0.8

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processing_rate(self) -> float:
        """Logs processed per second."""
        return (
            self.sources_analyzed / self.query_duration_seconds
            if self.query_duration_seconds > 0 else 0
        )


class QueryConfig(BaseModel):
    """Configuration for custom query processing."""

    default_max_logs: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Default maximum logs to analyze"
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for responses"
    )
    enable_followup_suggestions: bool = Field(
        default=True,
        description="Whether to generate follow-up question suggestions"
    )
    max_followup_suggestions: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of follow-up suggestions"
    )
    enable_pattern_integration: bool = Field(
        default=True,
        description="Whether to integrate with pattern detection"
    )
    query_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for query processing"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable query response caching"
    )
    cache_ttl_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Cache time-to-live in minutes"
    )

    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v
