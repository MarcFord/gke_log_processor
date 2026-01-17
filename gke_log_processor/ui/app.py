"""Main Textual application for GKE Log Processor."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from ..ai.analyzer import LogAnalysisEngine
from ..ai.client import GeminiConfig
from ..ai.summarizer import SummarizerConfig
from ..core.config import Config
from ..core.exceptions import (
    GKEConnectionError,
    KubernetesConnectionError,
    PodNotFoundError,
)
from ..core.logging import get_logger
from ..core.models import (
    ContainerState,
    ContainerStatus,
    LogEntry,
    LogLevel,
    PodCondition,
    PodPhase,
    QueryRequest,
    QueryType,
)
from ..core.models import (
    PodInfo as CorePodInfo,
)
from ..gke.client import GKEClient
from ..gke.kubernetes_client import (
    KubernetesClient,
)
from ..gke.kubernetes_client import (
    PodInfo as KubernetesPodInfo,
)
from .components.ai_insights_panel import AIInsightsPanel
from .components.log_viewer import LogViewer
from .components.pod_list import PodListWidget
from .components.status_bar import StatusBarWidget
from .dialogs import ConfigDialog, ConnectionDialog, ExportLogsDialog
from .widgets import (
    AIResultsViewer,
    ConfigManagerWidget,
    PodSelector,
    RealTimeLogDisplay,
)


class GKELogProcessorApp(App):
    """Main Textual application for GKE Log Processor."""

    # CSS styles for the application
    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        dock: top;
        height: 1;
    }

    Footer {
        dock: bottom;
        height: 1;
    }

    .app-container {
        height: 1fr;
        border: none;
    }

    .main-layout {
        height: 1fr;
    }

    .left-panel {
        width: 30%;
        min-width: 25;
        max-width: 50;
        border-right: solid $primary;
    }

    .center-panel {
        width: 1fr;
        min-width: 40;
    }

    .right-panel {
        width: 35%;
        min-width: 30;
        max-width: 55;
        border-left: solid $primary;
    }

    .panel-header {
        dock: top;
        height: 3;
        background: $panel;
        text-style: bold;
        content-align: center middle;
    }

    .connection-status {
        dock: top;
        height: 1;
        background: $warning;
        color: $text;
        text-style: bold;
        content-align: center middle;
    }

    .connection-status.connected {
        background: $success;
    }

    .connection-status.error {
        background: $error;
    }

    .help-overlay {
        display: none;
        layer: help;
        background: $surface 80%;
        border: solid $primary;
        padding: 2 4;
    }

    .help-overlay.show {
        display: block;
    }
    """

    # Application bindings
    BINDINGS = [
        Binding("ctrl+c,ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+h,f1", "help", "Help"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+p", "toggle_pod_panel", "Toggle Pods"),
        Binding("ctrl+l", "focus_logs", "Focus Logs"),
        Binding("ctrl+a", "toggle_ai_panel", "Toggle AI"),
        Binding("ctrl+s", "toggle_status_bar", "Toggle Status"),
        Binding("ctrl+n", "connect_cluster", "New Connection"),
        Binding("ctrl+o", "open_config", "Open Config"),
        Binding("ctrl+e", "export_logs", "Export Logs"),
        Binding("f5", "refresh_all", "Refresh All"),
        Binding("escape", "close_overlays", "Close Overlays"),
    ]

    # Reactive attributes
    current_cluster: reactive[Optional[str]] = reactive(None)
    current_namespace: reactive[str] = reactive("default")
    selected_pod: reactive[Optional[CorePodInfo]] = reactive(None)
    connection_status: reactive[str] = reactive("disconnected")
    show_help: reactive[bool] = reactive(False)
    show_pod_panel: reactive[bool] = reactive(True)
    show_ai_panel: reactive[bool] = reactive(True)
    show_status_bar: reactive[bool] = reactive(True)

    def __init__(self, config: Config):
        """Initialize the application."""
        super().__init__()
        self.title = "GKE Log Processor"
        self.sub_title = "Monitor & Analyze Pod Logs"
        self.config = config
        self.logger = get_logger(__name__)

        # Component references
        self.pod_selector: Optional[PodSelector] = None
        self.log_display: Optional[RealTimeLogDisplay] = None
        self.ai_viewer: Optional[AIResultsViewer] = None
        self.config_widget: Optional[ConfigManagerWidget] = None
        self.pod_list: Optional[PodListWidget] = None
        self.log_viewer: Optional[LogViewer] = None
        self.ai_panel: Optional[AIInsightsPanel] = None
        self.status_bar: Optional[StatusBarWidget] = None
        self._last_export: Optional[Dict[str, Any]] = None
        self._gke_client: Optional[GKEClient] = None
        self._kubernetes_client: Optional[KubernetesClient] = None
        self._analysis_engine: Optional[LogAnalysisEngine] = None
        self._current_logs: List[LogEntry] = []

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()

        with Container(classes="app-container"):
            # Connection status bar
            yield Static(
                "🔴 Disconnected - Press Ctrl+N to connect to a cluster",
                id="connection-status",
                classes="connection-status"
            )

            with Horizontal(classes="main-layout"):
                # Left Panel - Pod List
                with Vertical(classes="left-panel", id="pod-panel"):
                    self.pod_selector = PodSelector(id="pod-selector")
                    yield self.pod_selector
                    self.pod_list = self.pod_selector.pod_list

                # Center Panel - Log Viewer
                with Vertical(classes="center-panel"):
                    self.log_display = RealTimeLogDisplay(id="log-display")
                    yield self.log_display
                    self.log_viewer = self.log_display.log_viewer

                # Right Panel - AI Insights
                with Vertical(classes="right-panel", id="ai-panel"):
                    self.ai_viewer = AIResultsViewer(id="ai-results")
                    yield self.ai_viewer
                    self.ai_panel = self.ai_viewer.panel

                    self.config_widget = ConfigManagerWidget(self.config, id="config-widget")
                    yield self.config_widget

            # Status Bar
            self.status_bar = StatusBarWidget(id="status-bar")
            yield self.status_bar

        # Help overlay (initially hidden)
        with Container(classes="help-overlay", id="help-overlay"):
            yield Static(self._get_help_text(), id="help-text")

        yield Footer()

    def on_mount(self) -> None:
        """Handle application mount."""
        self.logger.info("GKE Log Processor application started")

        # Set initial status
        if self.status_bar:
            self.status_bar.set_processing_status("Ready")
            self.status_bar.update_connection_status("No cluster connected")
            self.status_bar.update_pods_info(0)

        if self.config_widget:
            self.config_widget.update_config(self.config)

        if self.pod_selector:
            self.pod_selector.namespace = self.config.kubernetes.default_namespace

        # Try to auto-connect if configuration is available
        self.call_later(self._try_auto_connect)

    async def _try_auto_connect(self) -> None:
        """Try to automatically connect using configuration."""
        try:
            if self._gke_client is not None:
                return

            cluster = self.config.current_cluster
            if not cluster:
                return

            namespace = (
                self.config.kubernetes.default_namespace
                or cluster.namespace
                or "default"
            )

            connection_info = {
                "cluster_name": cluster.name,
                "project_id": cluster.project_id,
                "zone": cluster.zone or "",
                "region": cluster.region or "",
                "namespace": namespace,
                "connection_type": "region" if cluster.is_regional else "zone",
            }

            self.logger.info("Attempting auto-connection with existing config")
            await self._handle_connection_request(connection_info)
        except Exception as e:
            self.logger.warning(f"Auto-connection failed: {e}")

    def watch_connection_status(self, status: str) -> None:
        """Watch connection status changes."""
        status_widget = self.query_one("#connection-status", Static)

        if status == "connected":
            status_widget.update(f"🟢 Connected to {self.current_cluster}")
            status_widget.remove_class("error")
            status_widget.add_class("connected")
        elif status == "error":
            status_widget.update("🔴 Connection Error - Check configuration")
            status_widget.remove_class("connected")
            status_widget.add_class("error")
        else:
            status_widget.update("🔴 Disconnected - Press Ctrl+N to connect")
            status_widget.remove_class("connected", "error")

    def watch_show_pod_panel(self, show: bool) -> None:
        """Toggle pod panel visibility."""
        panel = self.query_one("#pod-panel")
        panel.display = show

    def watch_show_ai_panel(self, show: bool) -> None:
        """Toggle AI panel visibility."""
        panel = self.query_one("#ai-panel")
        panel.display = show

    def watch_show_status_bar(self, show: bool) -> None:
        """Toggle status bar visibility."""
        if self.status_bar:
            self.status_bar.display = show

    def on_unmount(self) -> None:
        """Ensure external clients are cleaned up on exit."""
        self._close_clients()

    def watch_show_help(self, show: bool) -> None:
        """Toggle help overlay."""
        overlay = self.query_one("#help-overlay")
        if show:
            overlay.add_class("show")
        else:
            overlay.remove_class("show")

    # Message handlers for component communication
    async def on_pod_list_widget_pod_selected(self, message: PodListWidget.PodSelected) -> None:
        """Handle pod selection from pod list."""
        pod = message.pods[-1] if message.pods else None
        if not pod:
            return

        self.selected_pod = pod
        self.logger.info(f"Pod selected: {pod.name}")

        if self.status_bar:
            self.status_bar.set_processing_status(f"Loading logs for {pod.name}")

        if self.log_display:
            self.log_display.clear_logs()
            self.log_display.update_status(f"Loading logs for {pod.name}", streaming=self.log_display.streaming, source=pod.name)

        # Clear previous logs and load new ones
        if self.log_display:
            # TODO: Implement actual log loading from GKE
            await self._load_pod_logs(pod)

    async def on_log_viewer_log_entry_selected(self, message: LogViewer.LogEntrySelected) -> None:
        """Handle log entry selection for AI analysis."""
        self.logger.debug("Log entry selected: %s", message.log_entry.id)

    async def on_ai_insights_panel_query_requested(self, message: AIInsightsPanel.QueryRequested) -> None:
        """Handle AI query requests."""
        self.logger.info(f"AI query requested: {message.query}")
        if self.status_bar:
            self.status_bar.set_processing_status("Processing AI query...")

        # TODO: Implement actual AI query processing
        await self._process_ai_query(message.query)

    async def on_ai_insights_panel_analysis_requested(self, message: AIInsightsPanel.AnalysisRequested) -> None:
        """Handle AI analysis requests from the insights panel."""

        logs = list(self._current_logs)
        if not logs:
            self.logger.info("AI analysis requested with no logs loaded")
            if self.status_bar:
                self.status_bar.set_processing_status("Load logs before running analysis")
            if self.ai_viewer:
                self.ai_viewer.show_message("No logs loaded yet. Select a pod and try again.")
            return

        engine = self._ensure_analysis_engine()
        analysis_type = message.analysis_type or "comprehensive"
        use_ai = (
            analysis_type != "quick"
            and self.config.ai.analysis_enabled
            and engine.gemini_client is not None
        )

        if self.status_bar:
            self.status_bar.set_processing_status("Running AI analysis...")
        if self.ai_viewer:
            self.ai_viewer.show_message("Running analysis…")

        try:
            result = await engine.analyze_logs_comprehensive(
                logs,
                use_ai=use_ai,
                analysis_type=analysis_type,
            )

            summary_report = None
            if analysis_type in {"comprehensive", "summary"}:
                try:
                    summary_report = await engine.summarizer.summarize_logs(logs)
                except Exception as summary_error:
                    self.logger.warning("Log summarization failed: %s", summary_error)

            if self.ai_viewer:
                self.ai_viewer.update_analysis(result, logs)
                if summary_report:
                    self.ai_viewer.update_summary(summary_report)

            if self.status_bar:
                self.status_bar.set_processing_status(
                    f"Analysis complete · severity {result.overall_severity.value}")

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("AI analysis failed: %s", exc)
            if self.status_bar:
                self.status_bar.set_processing_status(f"AI analysis failed: {exc}")
            if self.ai_viewer:
                self.ai_viewer.show_message(f"Analysis failed: {exc}")

    async def on_pod_selector_namespace_changed(self, message: PodSelector.NamespaceChanged) -> None:
        """Handle namespace filter changes."""

        self.current_namespace = message.namespace
        self.config.kubernetes.default_namespace = message.namespace

        if self.status_bar and self.current_cluster:
            self.status_bar.update_connection_status(f"{self.current_cluster} ({self.current_namespace})")
            self.status_bar.set_processing_status(f"Namespace set to {message.namespace}")

    async def on_pod_selector_refresh_requested(self, message: PodSelector.RefreshRequested) -> None:
        """Handle refresh requests coming from the pod selector."""

        await self._refresh_pod_list()

    async def on_real_time_log_display_start_streaming(self, message: RealTimeLogDisplay.StartStreaming) -> None:
        """Start streaming logs for the selected pod."""

        if not self.selected_pod:
            if self.log_display:
                self.log_display.update_status("Select a pod to stream", streaming=False)
            if self.status_bar:
                self.status_bar.set_processing_status("No pod selected for streaming")
            return

        pod_name = self.selected_pod.name
        if self.log_display:
            self.log_display.update_status(f"Streaming logs for {pod_name}", streaming=True, source=pod_name)
        if self.status_bar:
            self.status_bar.set_processing_status(f"Streaming logs for {pod_name}")

        await self._load_pod_logs(self.selected_pod)

    async def on_real_time_log_display_pause_streaming(self, message: RealTimeLogDisplay.PauseStreaming) -> None:
        """Pause log streaming."""

        if self.log_display:
            self.log_display.update_status("Streaming paused", streaming=False)
        if self.status_bar:
            self.status_bar.set_processing_status("Log streaming paused")

    async def on_real_time_log_display_clear_logs(self, message: RealTimeLogDisplay.ClearLogs) -> None:
        """Clear log display contents."""

        if self.log_display:
            self.log_display.clear_logs()
            self.log_display.update_status("Logs cleared", streaming=self.log_display.streaming)
        self._current_logs = []
        if self.ai_viewer:
            self.ai_viewer.clear()
            self.ai_viewer.show_message("Logs cleared. Select a pod to load fresh data.")
        if self.status_bar:
            self.status_bar.set_processing_status("Logs cleared")

    async def on_real_time_log_display_export_logs(
        self, message: RealTimeLogDisplay.ExportLogs
    ) -> None:
        """Export current logs via the display widget."""

        if message.format_type == "prompt":
            await self._open_export_dialog()
            return

        self._perform_log_export(message.format_type)

    async def on_export_logs_dialog_export_confirmed(
        self, message: ExportLogsDialog.ExportConfirmed
    ) -> None:
        """Handle export confirmation from dialog."""

        self._perform_log_export(message.format_type)

    async def on_config_manager_widget_edit_config_requested(
        self, message: ConfigManagerWidget.EditConfigRequested
    ) -> None:
        """Open the configuration dialog from the summary widget."""

        await self.action_open_config()

    async def on_config_manager_widget_test_connection_requested(
        self, message: ConfigManagerWidget.TestConnectionRequested
    ) -> None:
        """Trigger a connection test using current configuration."""

        if self.status_bar:
            self.status_bar.set_processing_status("Testing current connection...")

        await self._refresh_pod_list()

    async def on_config_manager_widget_reload_config_requested(
        self, message: ConfigManagerWidget.ReloadConfigRequested
    ) -> None:
        """Reload configuration from in-memory state (placeholder)."""

        if self.config_widget:
            self.config_widget.update_config(self.config)
        if self.status_bar:
            self.status_bar.set_processing_status("Configuration reloaded")

    async def _load_pod_logs(self, pod: CorePodInfo) -> None:
        """Load recent logs for the selected pod."""
        if self._kubernetes_client is None:
            self.logger.warning("Cannot load logs; Kubernetes client is not initialized")
            if self.status_bar:
                self.status_bar.set_processing_status("Connect to a cluster to load logs")
            if self.log_display:
                self.log_display.update_status(
                    "No cluster connection",
                    streaming=False,
                    source=pod.name,
                )
            return

        container_name = pod.containers[0].name if pod.containers else None
        tail_lines = getattr(self.config.streaming, "tail_lines", 200)
        if tail_lines <= 0:
            tail_lines = 200

        try:
            log_lines = await self._kubernetes_client.get_pod_logs(
                pod.name,
                namespace=pod.namespace,
                container=container_name,
                tail_lines=tail_lines,
                timestamps=True,
            )

            log_entries = [
                self._build_log_entry(line, pod, container_name)
                for line in log_lines
            ]

            self._current_logs = log_entries

            if self.ai_viewer:
                self.ai_viewer.clear()
                if log_entries:
                    self.ai_viewer.show_message(
                        f"Loaded {len(log_entries)} logs for {pod.name}. Choose a feature and press ▶ Run."
                    )
                else:
                    self.ai_viewer.show_message(
                        f"No logs available for {pod.name}. Try streaming or refreshing."
                    )

            if self.log_display:
                self.log_display.set_logs(log_entries)
                line_count = len(log_entries)
                status_message = (
                    f"{line_count} log lines loaded"
                    if line_count
                    else "No logs available"
                )
                self.log_display.update_status(
                    status_message,
                    streaming=self.log_display.streaming,
                    source=pod.name,
                )

            if self.status_bar:
                if log_entries:
                    self.status_bar.set_processing_status(
                        f"Loaded {len(log_entries)} logs for {pod.name}"
                    )
                else:
                    self.status_bar.set_processing_status(
                        f"No logs available for {pod.name}"
                    )

        except (KubernetesConnectionError, GKEConnectionError, PodNotFoundError) as exc:
            self.logger.error("Failed to load logs for %s: %s", pod.name, exc)
            self._current_logs = []
            if self.ai_viewer:
                self.ai_viewer.clear()
                self.ai_viewer.show_message(
                    f"Unable to load logs for {pod.name}: {exc}"
                )
            if self.status_bar:
                self.status_bar.set_processing_status(f"Error loading logs: {exc}")
            if self.log_display:
                self.log_display.update_status(
                    f"Failed to load logs for {pod.name}",
                    streaming=False,
                    source=pod.name,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Unexpected error loading logs for %s", pod.name)
            self._current_logs = []
            if self.ai_viewer:
                self.ai_viewer.clear()
                self.ai_viewer.show_message(
                    f"Unexpected error loading logs for {pod.name}: {exc}"
                )
            if self.status_bar:
                self.status_bar.set_processing_status(f"Error loading logs: {exc}")
            if self.log_display:
                self.log_display.update_status(
                    f"Failed to load logs for {pod.name}",
                    streaming=False,
                    source=pod.name,
                )

    async def _process_ai_query(self, query: str) -> None:
        """Process an AI query."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            self.logger.info("Ignoring empty AI query request")
            if self.status_bar:
                self.status_bar.set_processing_status("Enter a question to query the logs")
            if self.ai_viewer:
                self.ai_viewer.show_message("Enter a question to run an AI query.", switch_to="query")
            return

        logs = list(self._current_logs)
        if not logs:
            self.logger.info("AI query requested with no logs loaded")
            if self.status_bar:
                self.status_bar.set_processing_status("Load logs before running queries")
            if self.ai_viewer:
                self.ai_viewer.show_message("No logs loaded yet. Select a pod and try again.", switch_to="query")
            return

        engine = self._ensure_analysis_engine()
        if not engine.gemini_client or not self.config.ai.analysis_enabled:
            message = "Gemini API key missing; AI queries are unavailable"
            self.logger.info(message)
            if self.status_bar:
                self.status_bar.set_processing_status(message)
            if self.ai_viewer:
                self.ai_viewer.show_message(
                    "Configure a Gemini API key in Settings to enable AI queries.",
                    switch_to="query",
                )
            return

        if self.ai_viewer:
            self.ai_viewer.show_message("Running AI query…", switch_to="query")

        try:
            request = QueryRequest(
                question=cleaned_query,
                query_type=QueryType.ANALYSIS,
                max_log_entries=min(len(logs), 200),
                enable_pattern_matching=True,
                include_context=True,
            )

            response = await engine.query_logs_natural_language(logs, request)

            if self.ai_viewer:
                self.ai_viewer.update_query_response(response)

            if self.status_bar:
                self.status_bar.set_processing_status(
                    f"Query complete · confidence {response.confidence_score:.0%}"
                )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("AI query failed: %s", exc)
            if self.status_bar:
                self.status_bar.set_processing_status(f"AI query error: {exc}")
            if self.ai_viewer:
                self.ai_viewer.show_message(f"AI query error: {exc}", switch_to="query")

    def _ensure_analysis_engine(self) -> LogAnalysisEngine:
        """Initialize and cache the log analysis engine."""

        if self._analysis_engine is not None:
            return self._analysis_engine

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
            else:
                self.logger.info(
                    "Gemini API key not configured; running analysis without LLM support"
                )
        else:
            self.logger.info("AI analysis disabled via configuration")

        summary_length = max(100, min(self.config.ai.max_tokens, 2000))
        summarizer_config = SummarizerConfig(
            max_summary_length=summary_length,
            enable_ai_summarization=bool(gemini_config),
        )

        self._analysis_engine = LogAnalysisEngine(
            gemini_config=gemini_config,
            summarizer_config=summarizer_config,
        )
        return self._analysis_engine

    def _close_clients(self) -> None:
        """Dispose of any active GKE or Kubernetes clients."""
        if self._gke_client:
            try:
                self._gke_client.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                self.logger.debug(f"Error closing GKE client: {exc}")
        self._gke_client = None
        self._kubernetes_client = None

    async def _refresh_pod_list(self) -> None:
        """Refresh the pod list from the connected Kubernetes cluster."""

        namespace = self.config.kubernetes.default_namespace or "default"

        if self.status_bar:
            self.status_bar.set_processing_status(
                f"Refreshing pod list for namespace '{namespace}'..."
            )

        if self._kubernetes_client is None:
            self.logger.debug("Pod refresh skipped; Kubernetes client not initialized")
            if self.status_bar:
                self.status_bar.set_processing_status("Connect to a cluster to list pods")
            return

        try:
            pods = await self._kubernetes_client.list_pods(
                namespace=namespace,
                force_refresh=True,
            )

            ui_pods = [self._convert_pod_info(pod) for pod in pods]

            if self.pod_selector:
                self.pod_selector.set_pods(ui_pods)

            if self.status_bar:
                self.status_bar.update_pods_info(len(ui_pods))
                self.status_bar.set_processing_status("Pod list updated")

        except (KubernetesConnectionError, GKEConnectionError) as exc:
            self.logger.error(f"Failed to refresh pod list: {exc}")
            self.connection_status = "error"
            if self.status_bar:
                self.status_bar.set_processing_status(f"Pod refresh failed: {exc}")
            if self.pod_selector:
                self.pod_selector.set_pods([])
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Unexpected error refreshing pod list: %s", exc)
            if self.status_bar:
                self.status_bar.set_processing_status(f"Pod refresh error: {exc}")

    def _convert_pod_info(self, pod: KubernetesPodInfo) -> CorePodInfo:
        """Translate Kubernetes pod information into UI pod model."""

        containers: List[ContainerStatus] = []
        for status in getattr(pod, "container_statuses", []) or []:
            state_enum = ContainerState.WAITING
            started_at = None
            finished_at = None
            exit_code = None
            reason = None
            message = None

            state = getattr(status, "state", None)
            if state:
                running = getattr(state, "running", None)
                terminated = getattr(state, "terminated", None)
                waiting = getattr(state, "waiting", None)

                if running:
                    state_enum = ContainerState.RUNNING
                    started_at = getattr(running, "started_at", None)
                elif terminated:
                    state_enum = ContainerState.TERMINATED
                    started_at = getattr(terminated, "started_at", None)
                    finished_at = getattr(terminated, "finished_at", None)
                    exit_code = getattr(terminated, "exit_code", None)
                    reason = getattr(terminated, "reason", None)
                    message = getattr(terminated, "message", None)
                elif waiting:
                    state_enum = ContainerState.WAITING
                    reason = getattr(waiting, "reason", None)
                    message = getattr(waiting, "message", None)

            containers.append(
                ContainerStatus(
                    name=getattr(status, "name", ""),
                    image=getattr(status, "image", ""),
                    state=state_enum,
                    ready=bool(getattr(status, "ready", False)),
                    restart_count=int(getattr(status, "restart_count", 0) or 0),
                    started_at=started_at,
                    finished_at=finished_at,
                    exit_code=exit_code,
                    reason=reason,
                    message=message,
                )
            )

        conditions: List[PodCondition] = []
        for condition in getattr(pod, "conditions", []) or []:
            condition_type = getattr(condition, "type", None)
            condition_status = getattr(condition, "status", None)
            if not condition_type or not condition_status:
                continue
            conditions.append(
                PodCondition(
                    type=condition_type,
                    status=condition_status,
                    last_probe_time=getattr(condition, "last_probe_time", None),
                    last_transition_time=getattr(condition, "last_transition_time", None),
                    reason=getattr(condition, "reason", None),
                    message=getattr(condition, "message", None),
                )
            )

        phase_value = getattr(pod, "phase", None) or PodPhase.UNKNOWN.value
        try:
            phase = PodPhase(phase_value)
        except ValueError:
            phase = PodPhase.UNKNOWN

        cluster_name = (
            self.config.gke.cluster_name
            or (self.config.current_cluster.name if self.config.current_cluster else None)
            or "unknown"
        )

        created_at = getattr(pod, "creation_timestamp", None) or datetime.now()
        started_at = next((status.started_at for status in containers if status.started_at), None)

        return CorePodInfo(
            name=getattr(pod, "name", "unknown"),
            namespace=getattr(pod, "namespace", "default"),
            cluster=cluster_name,
            uid=str(getattr(pod, "uid", "")),
            phase=phase,
            node_name=getattr(pod, "node_name", None),
            pod_ip=getattr(pod, "pod_ip", None),
            host_ip=getattr(pod, "host_ip", None),
            created_at=created_at,
            started_at=started_at,
            labels=dict(getattr(pod, "labels", {}) or {}),
            annotations=dict(getattr(pod, "annotations", {}) or {}),
            containers=containers,
            conditions=conditions,
            owner_references=[],
        )

    def _build_log_entry(
        self,
        raw_line: str,
        pod: CorePodInfo,
        container_name: Optional[str],
    ) -> LogEntry:
        """Convert a raw log line into a structured log entry."""

        timestamp, message = self._parse_log_line(raw_line)
        level = self._detect_log_level(message)

        container = container_name or (
            pod.containers[0].name if pod.containers else "unknown"
        )
        cluster_name = (
            pod.cluster
            or self.current_cluster
            or self.config.gke.cluster_name
            or "unknown"
        )

        return LogEntry(
            timestamp=timestamp,
            message=message,
            level=level.value if level else None,
            pod_name=pod.name,
            container_name=container,
            namespace=pod.namespace,
            cluster=cluster_name,
            source=container,
            raw_message=raw_line,
        )

    def _parse_log_line(self, line: str) -> Tuple[datetime, str]:
        """Parse a Kubernetes log line into timestamp and message components."""

        default_timestamp = datetime.now(timezone.utc)
        if not line:
            return default_timestamp, ""

        # Format: 2026-01-16T21:22:38.123456Z message
        ts_candidate, _, remainder = line.partition(" ")
        parsed_ts = self._parse_timestamp(ts_candidate)
        if parsed_ts and remainder:
            return parsed_ts, remainder

        # Format: 2026-...Z stdout F message
        parts = line.split(" ", 3)
        if len(parts) >= 4 and parts[1] in {"stdout", "stderr"}:
            parsed_ts = self._parse_timestamp(parts[0])
            if parsed_ts:
                message = parts[3]
                return parsed_ts, message

        return default_timestamp, line

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        """Parse RFC3339 timestamps emitted by Kubernetes logs."""

        if not value:
            return None

        try:
            normalised = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalised)
        except ValueError:
            return None

    def _detect_log_level(self, message: str) -> Optional[LogLevel]:
        """Best-effort detection of log level from message prefix."""

        if not message:
            return None

        token = message.split(" ", 1)[0].strip("[]:").upper()
        for level in LogLevel:
            if token == level.value:
                return level
        return None

    async def _handle_connection_request(self, connection_info: dict) -> None:
        """Handle connection request from dialog."""
        try:
            cluster_name = connection_info["cluster_name"]
            namespace = connection_info["namespace"] or "default"

            self.logger.info(f"Connecting to cluster: {cluster_name}")

            if self.status_bar:
                self.status_bar.set_processing_status("Connecting to cluster...")

            # Update configuration with requested connection values
            self.config.gke.cluster_name = cluster_name
            self.config.gke.project_id = connection_info["project_id"]

            if connection_info["connection_type"] == "zone":
                self.config.gke.zone = connection_info["zone"]
                self.config.gke.region = None
            else:
                self.config.gke.region = connection_info["region"]
                self.config.gke.zone = None

            self.config.kubernetes.default_namespace = namespace

            # Tear down any existing clients before establishing a new connection
            self._close_clients()

            # Initialize GKE and Kubernetes clients in a thread to avoid blocking the UI loop
            self._gke_client = GKEClient(self.config)
            self._kubernetes_client = await asyncio.to_thread(
                self._gke_client.get_kubernetes_client
            )

            # Ensure cluster information is cached for status updates
            cluster_info = self._gke_client.cluster_info

            # Update application state
            self.current_cluster = cluster_info.name or cluster_name
            self.current_namespace = namespace
            self.connection_status = "connected"

            if self.status_bar:
                self.status_bar.set_processing_status(
                    f"Connected to {self.current_cluster}"
                )
                self.status_bar.update_connection_status(
                    f"{self.current_cluster} ({self.current_namespace})"
                )

            if self.config_widget:
                self.config_widget.update_config(self.config)

            if self.pod_selector:
                self.pod_selector.namespace = namespace

            await self._refresh_pod_list()

        except (GKEConnectionError, KubernetesConnectionError) as exc:
            self.logger.error(f"Connection failed: {exc}")
            self.connection_status = "error"
            if self.status_bar:
                self.status_bar.set_processing_status(f"Connection failed: {exc}")
            self._close_clients()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Unexpected error establishing connection: %s", exc)
            self.connection_status = "error"
            if self.status_bar:
                self.status_bar.set_processing_status(f"Connection error: {exc}")
            self._close_clients()

    async def _handle_config_save(self, config: Config) -> None:
        """Handle configuration save from dialog."""
        try:
            self.logger.info("Saving configuration")

            if self.status_bar:
                self.status_bar.set_processing_status("Saving configuration...")

            # Update application configuration
            self.config = config
            self._analysis_engine = None

            if self.config_widget:
                self.config_widget.update_config(self.config)

            # TODO: Persist configuration to file
            # config.save_to_file("config.yaml")

            if self.status_bar:
                self.status_bar.set_processing_status("Configuration saved successfully")

            self.logger.info("Configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            if self.status_bar:
                self.status_bar.set_processing_status(f"Config save failed: {e}")

    # Dialog message handlers
    async def on_connection_dialog_connection_requested(
        self, message: ConnectionDialog.ConnectionRequested
    ) -> None:
        """Handle connection request from dialog."""
        await self._handle_connection_request(message.connection_info)

    async def on_connection_dialog_connection_cancelled(
        self, message: ConnectionDialog.ConnectionCancelled
    ) -> None:
        """Handle connection cancellation from dialog."""
        self.logger.info("Connection dialog cancelled")

    async def on_config_dialog_config_saved(
        self, message: ConfigDialog.ConfigSaved
    ) -> None:
        """Handle configuration save from dialog."""
        await self._handle_config_save(message.config)

    async def on_config_dialog_config_cancelled(
        self, message: ConfigDialog.ConfigCancelled
    ) -> None:
        """Handle configuration cancellation from dialog."""
        self.logger.info("Configuration dialog cancelled")

    # Action handlers
    async def action_help(self) -> None:
        """Toggle help overlay."""
        self.show_help = not self.show_help

    async def action_refresh(self) -> None:
        """Refresh current view."""
        if self.status_bar:
            self.status_bar.set_processing_status("Refreshing...")

        await self._refresh_pod_list()

        if self.selected_pod:
            await self._load_pod_logs(self.selected_pod)

        if self.status_bar:
            self.status_bar.set_processing_status("Refreshed")

    async def action_refresh_all(self) -> None:
        """Refresh all components."""
        await self.action_refresh()

    async def action_toggle_pod_panel(self) -> None:
        """Toggle pod panel visibility."""
        self.show_pod_panel = not self.show_pod_panel

    async def action_focus_logs(self) -> None:
        """Focus the log viewer."""
        if self.log_viewer:
            self.log_viewer.focus()

    async def action_toggle_ai_panel(self) -> None:
        """Toggle AI insights panel."""
        self.show_ai_panel = not self.show_ai_panel

    async def action_toggle_status_bar(self) -> None:
        """Toggle status bar."""
        self.show_status_bar = not self.show_status_bar

    async def action_connect_cluster(self) -> None:
        """Show cluster connection dialog."""
        self.logger.info("Opening cluster connection dialog")

        # Create and push the connection dialog
        dialog = ConnectionDialog(self.config)
        await self.push_screen(dialog)

    async def action_open_config(self) -> None:
        """Open configuration dialog."""
        self.logger.info("Opening configuration dialog")

        # Create and push the configuration dialog
        dialog = ConfigDialog(self.config)
        await self.push_screen(dialog)

    async def action_export_logs(self) -> None:
        """Open export dialog to choose format."""
        await self._open_export_dialog()

    async def _open_export_dialog(self) -> None:
        """Open the export format selection dialog."""
        dialog = ExportLogsDialog()
        await self.push_screen(dialog)

    def _perform_log_export(self, format_type: str) -> None:
        """Perform the log export and update UI state."""
        if not self.log_display:
            return

        data = self.log_display.export_logs(format_type)
        byte_size = len(data.encode("latin-1", "replace"))
        self.logger.info("Exported logs as %s (%s bytes)", format_type, byte_size)

        if self.status_bar:
            self.status_bar.set_processing_status(
                f"Logs exported as {format_type.upper()}"
            )

        self._last_export = {
            "format": format_type,
            "content": data,
            "size": byte_size,
        }

    async def action_close_overlays(self) -> None:
        """Close all overlay dialogs."""
        self.show_help = False

    def _get_help_text(self) -> str:
        """Get help text for the overlay."""
        return """
# GKE Log Processor - Keyboard Shortcuts

## Navigation
- **Ctrl+P**: Toggle Pod Panel
- **Ctrl+L**: Focus Log Viewer
- **Ctrl+A**: Toggle AI Panel
- **Ctrl+S**: Toggle Status Bar

## Actions
- **Ctrl+N**: New Cluster Connection
- **Ctrl+O**: Open Configuration
- **Ctrl+R**: Refresh Current View
- **F5**: Refresh All Components
- **Ctrl+E**: Export Logs

## General
- **Ctrl+H / F1**: Toggle This Help
- **Ctrl+Q / Ctrl+C**: Quit Application
- **Escape**: Close Overlays

## Tips
- Use Tab to navigate between panels
- Right-click for context menus
- Double-click pod names to select
- Use filters in each panel to narrow results

Press Escape to close this help.
        """
