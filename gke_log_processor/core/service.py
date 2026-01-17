"""Core service layer for GKE Log Processor."""

import asyncio
from typing import List, Optional

from ..ai.analyzer import LogAnalysisEngine
from ..ai.client import GeminiConfig
from ..ai.summarizer import LogSummaryReport, SummarizerConfig
from ..core.config import Config
from ..core.models import AIAnalysisResult, LogEntry, LogLevel
from ..gke.client import GKEClient


class LogProcessingService:
    """Service to handle log fetching, processing, and analysis."""

    def __init__(self, config: Config):
        self.config = config
        self._analysis_engine: Optional[LogAnalysisEngine] = None

    @property
    def analysis_engine(self) -> LogAnalysisEngine:
        """Lazy initialization of the analysis engine."""
        if self._analysis_engine is None:
            self._analysis_engine = self._build_analysis_engine()
        return self._analysis_engine

    def _build_analysis_engine(self) -> LogAnalysisEngine:
        """Construct a log analysis engine respecting configuration."""
        gemini_config: Optional[GeminiConfig] = None
        if self.config.ai.analysis_enabled:
            api_key = self.config.effective_gemini_api_key
            if api_key:
                max_tokens = max(1, min(self.config.ai.max_tokens, 32768))
                gemini_config = GeminiConfig(
                    api_key=api_key,
                    model=self.config.ai.model_name,
                    temperature=self.config.ai.temperature,
                    max_output_tokens=max_tokens,
                )

        summary_length = max(100, min(self.config.ai.max_tokens, 2000))
        summarizer_config = SummarizerConfig(
            max_summary_length=summary_length,
            enable_ai_summarization=bool(gemini_config),
        )

        return LogAnalysisEngine(
            gemini_config=gemini_config,
            summarizer_config=summarizer_config,
        )

    async def get_pod_logs(
        self,
        namespace: str,
        pod_name: str,
        container: Optional[str] = None,
        tail_lines: int = 200,
    ) -> List[LogEntry]:
        """Fetch and parse logs for a specific pod."""
        client = GKEClient(self.config)
        try:
            k8s_client = client.get_kubernetes_client()
            pod = await k8s_client.get_pod(pod_name, namespace)

            container_name = container or (
                pod.containers[0] if pod.containers else None
            )
            if not container_name:
                raise ValueError("Pod has no containers")

            raw_logs = await k8s_client.get_pod_logs(
                pod.name,
                namespace=pod.namespace,
                container=container_name,
                tail_lines=tail_lines,
                timestamps=True,
            )

            cluster_name = (
                self.config.gke.cluster_name
                or (
                    self.config.current_cluster.name
                    if self.config.current_cluster
                    else None
                )
                or "unknown"
            )

            log_entries = []
            current_entry: Optional[LogEntry] = None

            for line in raw_logs:
                new_entry = self._parse_log_line(
                    line,
                    pod_name=pod.name,
                    namespace=pod.namespace,
                    cluster=cluster_name,
                    container_name=container_name,
                )
                
                is_continuation = False
                if current_entry and not new_entry.level:
                    # Heuristics for continuation:
                    # 1. No detected log level
                    # 2. Message starts with typical stack trace patterns or whitespace
                    if (
                        new_entry.message.startswith(" ") 
                        or new_entry.message.startswith("\t")
                        or new_entry.message.startswith("Traceback") 
                        or new_entry.message.startswith("Caused by")
                        or new_entry.message.startswith("The above exception was the direct cause")
                    ):
                         is_continuation = True
                
                if is_continuation and current_entry:
                    # Append strictly the message part
                    # Note: We append with newline to preserve stack trace formatting
                    current_entry.message += "\n" + new_entry.message
                    # Update raw message too
                    current_entry.raw_message += "\n" + new_entry.raw_message
                else:
                    if current_entry:
                        log_entries.append(current_entry)
                    current_entry = new_entry
            
            if current_entry:
                log_entries.append(current_entry)

            return log_entries

        finally:
            client.close()

    async def get_matching_pods_logs(
        self,
        namespace: str,
        pod_regex: str,
        container: Optional[str] = None,
        tail_lines: int = 200,
    ) -> List[LogEntry]:
        """Fetch logs from multiple pods matching a regex."""
        import re

        client = GKEClient(self.config)
        try:
            k8s_client = client.get_kubernetes_client()
            pods = await k8s_client.list_pods(namespace=namespace)
            
            pattern = re.compile(pod_regex)
            matching_pods = [p for p in pods if pattern.search(p.name)]
            
            if not matching_pods:
                return []
                
            tasks = []
            for pod in matching_pods:
                # Reuse existing logic but we need to handle GKEClient lifecycle 
                # effectively. Since we are inside a method using a client, 
                # calling self.get_pod_logs would create a NEW client each time.
                # It's better to refactor valid logic or just call it and accept overhead 
                # or optimize later. For now, calling get_pod_logs in parallel is fine 
                # as each will create its own connection which is safe but maybe slightly heavy.
                # Optimization: pass client to internal method?
                # Let's keep it simple and clean first.
                tasks.append(self.get_pod_logs(
                    namespace=namespace,
                    pod_name=pod.name,
                    container=container,
                    tail_lines=tail_lines
                ))
            
            # Gather all logs
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_logs = []
            for res in results:
                if isinstance(res, list):
                    all_logs.extend(res)
                # Ignore failures for individual pods to avoid partial failure blocking everything?
                # Or log warning. For CLI summary, maybe partial is okay.
            
            # Sort by timestamp
            all_logs.sort(key=lambda x: x.timestamp)
            return all_logs

        finally:
            client.close()

    async def analyze_logs(
        self,
        log_entries: List[LogEntry],
        analysis_type: str = "comprehensive",
        skip_pattern_ai: bool = False
    ) -> AIAnalysisResult:
        """Perform AI analysis on a set of log entries."""
        if not log_entries:
            raise ValueError("No logs to analyze")

        use_ai = (
            self.config.ai.analysis_enabled
            and self.analysis_engine.gemini_client is not None
        )

        return await self.analysis_engine.analyze_logs_comprehensive(
            log_entries, 
            use_ai=use_ai, 
            analysis_type=analysis_type,
            skip_pattern_ai=skip_pattern_ai
        )

    async def summarize_logs(
        self, 
        log_entries: List[LogEntry],
        ai_summary: Optional[str] = None
    ) -> LogSummaryReport:
        """Generate a summary report for log entries."""
        if not log_entries:
            raise ValueError("No logs to summarize")

        return await self.analysis_engine.summarizer.summarize_logs(
            log_entries,
            ai_summary=ai_summary
        )

    def _parse_log_line(
        self,
        line: str,
        pod_name: str,
        namespace: str,
        cluster: str,
        container_name: str,
    ) -> LogEntry:
        """Parse a single raw log line."""
        from datetime import datetime, timezone
        from ..core.models import LogEntry, LogLevel

        default_timestamp = datetime.now(timezone.utc)
        message = line
        timestamp = default_timestamp

        if line:
            # Simple heuristic parsing similar to CLI
            ts_candidate, _, remainder = line.partition(" ")
            parsed_ts = self._parse_timestamp(ts_candidate)
            if parsed_ts and remainder:
                timestamp = parsed_ts
                message = remainder
            else:
                parts = line.split(" ", 3)
                if len(parts) >= 4 and parts[1] in {"stdout", "stderr"}:
                     parsed_ts = self._parse_timestamp(parts[0])
                     if parsed_ts:
                         timestamp = parsed_ts
                         message = parts[3]

        level = self._detect_log_level(message)

        return LogEntry(
            timestamp=timestamp,
            message=message,
            level=level.value if level else None,
            source=container_name,
            pod_name=pod_name,
            namespace=namespace,
            cluster=cluster,
            container_name=container_name,
            raw_message=line,
        )

    def _parse_timestamp(self, value: str) -> Optional["datetime"]:
        from datetime import datetime
        if not value:
            return None
        normalised = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalised)
        except ValueError:
            return None

    def _detect_log_level(self, message: str) -> Optional[LogLevel]:
        if not message:
            return None
            
        # Common log levels to look for
        level_map = {
            "TRACE": LogLevel.TRACE,
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARN": LogLevel.WARNING,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "ERR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL,
            "FATAL": LogLevel.CRITICAL,
        }
        
        # Check first token (e.g. "INFO:")
        first_token = message.split(" ", 1)[0].strip("[]:").upper()
        if first_token in level_map:
            return level_map[first_token]
            
        # Check for bracketed levels in the first few tokens
        # e.g. [2024-01-01] [INFO] ...
        tokens = message.split(" ")
        # Check first 5 tokens
        for token in tokens[:5]:
            clean_token = token.strip("[]:").upper()
            if clean_token in level_map:
                # To be safer, ensure it was properly bracketed if it wasn't the very first token
                # But strict bracketing check might be too fragile.
                # If we see [DEBUG] in the first few words, it's likely the level.
                if f"[{clean_token}]" in token.upper() or f"[{clean_token}" in token.upper() or f"{clean_token}]" in token.upper() or token.upper() == clean_token:
                     return level_map[clean_token]
                     
        return None
