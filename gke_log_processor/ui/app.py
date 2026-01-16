"""Main Textual application for GKE Log Processor."""

import asyncio
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from ..core.config import Config
from ..core.logging import get_logger
from ..core.models import LogEntry, PodInfo
from .components.ai_insights_panel import AIInsightsPanel
from .components.log_viewer import LogViewer
from .components.pod_list import PodListWidget
from .components.status_bar import StatusBarWidget
from .dialogs.config_dialog import ConfigDialog
from .dialogs.connection_dialog import ConnectionDialog


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
    selected_pod: reactive[Optional[PodInfo]] = reactive(None)
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
        self.pod_list: Optional[PodListWidget] = None
        self.log_viewer: Optional[LogViewer] = None
        self.ai_panel: Optional[AIInsightsPanel] = None
        self.status_bar: Optional[StatusBarWidget] = None

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
                    yield Static("📋 Kubernetes Pods", classes="panel-header")
                    self.pod_list = PodListWidget(id="pod-list")
                    yield self.pod_list

                # Center Panel - Log Viewer
                with Vertical(classes="center-panel"):
                    yield Static("📜 Pod Logs", classes="panel-header")
                    self.log_viewer = LogViewer(id="log-viewer")
                    yield self.log_viewer

                # Right Panel - AI Insights
                with Vertical(classes="right-panel", id="ai-panel"):
                    yield Static("🤖 AI Insights", classes="panel-header")
                    self.ai_panel = AIInsightsPanel(id="ai-insights")
                    yield self.ai_panel

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

        # Try to auto-connect if configuration is available
        self.call_later(self._try_auto_connect)

    async def _try_auto_connect(self) -> None:
        """Try to automatically connect using configuration."""
        try:
            if (self.config.gke.cluster_name and
                self.config.gke.project_id and
                    (self.config.gke.zone or self.config.gke.region)):

                self.logger.info("Attempting auto-connection with existing config")
                await self.action_connect_cluster()
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
        self.selected_pod = message.pod
        self.logger.info(f"Pod selected: {message.pod.name}")

        if self.status_bar:
            self.status_bar.set_processing_status(f"Loading logs for {message.pod.name}")

        # Clear previous logs and load new ones
        if self.log_viewer:
            self.log_viewer.clear_logs()
            # TODO: Implement actual log loading from GKE
            await self._load_pod_logs(message.pod)

    async def on_log_viewer_log_entry_selected(self, message: LogViewer.LogEntrySelected) -> None:
        """Handle log entry selection for AI analysis."""
        if self.ai_panel:
            # TODO: Trigger AI analysis of the selected log entry
            self.ai_panel.analyze_log_entry(message.log_entry)

    async def on_ai_insights_panel_query_requested(self, message: AIInsightsPanel.QueryRequested) -> None:
        """Handle AI query requests."""
        self.logger.info(f"AI query requested: {message.query}")
        if self.status_bar:
            self.status_bar.set_processing_status("Processing AI query...")

        # TODO: Implement actual AI query processing
        await self._process_ai_query(message.query)

    async def _load_pod_logs(self, pod: PodInfo) -> None:
        """Load logs for the selected pod."""
        try:
            # TODO: Implement actual GKE log loading
            # This is a placeholder that would integrate with the GKE client
            sample_logs = [
                LogEntry(
                    timestamp=datetime.now(),
                    message=f"Sample log from {pod.name}",
                    level="INFO",
                    pod_name=pod.name,
                    container_name=pod.containers[0].name if pod.containers else "unknown",
                    namespace=pod.namespace,
                    cluster=self.current_cluster or "unknown",
                    source=pod.containers[0].name if pod.containers else "unknown",
                    raw_message=f"Sample log from {pod.name}"
                )
            ]

            if self.log_viewer:
                self.log_viewer.set_logs(sample_logs)

            if self.status_bar:
                self.status_bar.set_processing_status(f"Loaded {len(sample_logs)} logs for {pod.name}")

        except Exception as e:
            self.logger.error(f"Failed to load logs for {pod.name}: {e}")
            if self.status_bar:
                self.status_bar.set_processing_status(f"Error loading logs: {e}")

    async def _process_ai_query(self, query: str) -> None:
        """Process an AI query."""
        try:
            # TODO: Implement actual AI query processing
            # This would integrate with the AI analyzer
            if self.status_bar:
                self.status_bar.set_processing_status("AI query completed")
        except Exception as e:
            self.logger.error(f"AI query failed: {e}")
            if self.status_bar:
                self.status_bar.set_processing_status(f"AI query error: {e}")

    async def _handle_connection_request(self, connection_info: dict) -> None:
        """Handle connection request from dialog."""
        try:
            self.logger.info(f"Connecting to cluster: {connection_info['cluster_name']}")

            if self.status_bar:
                self.status_bar.set_processing_status("Connecting to cluster...")

            # Update configuration
            self.config.gke.cluster_name = connection_info["cluster_name"]
            self.config.gke.project_id = connection_info["project_id"]

            if connection_info["connection_type"] == "zone":
                self.config.gke.zone = connection_info["zone"]
                self.config.gke.region = None
            else:
                self.config.gke.region = connection_info["region"]
                self.config.gke.zone = None

            self.config.kubernetes.default_namespace = connection_info["namespace"]

            # TODO: Implement actual GKE client connection
            # This would use the GKE client to establish connection
            await asyncio.sleep(1)  # Simulate connection

            # Update application state
            self.current_cluster = connection_info["cluster_name"]
            self.current_namespace = connection_info["namespace"]
            self.connection_status = "connected"

            if self.status_bar:
                self.status_bar.set_processing_status(f"Connected to {connection_info['cluster_name']}")
                self.status_bar.update_connection_status(f"{connection_info['cluster_name']} ({connection_info['namespace']})")

            # Refresh pod list if available
            if self.pod_list:
                await self.pod_list.refresh_pods()

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.connection_status = "error"
            if self.status_bar:
                self.status_bar.set_processing_status(f"Connection failed: {e}")

    async def _handle_config_save(self, config: Config) -> None:
        """Handle configuration save from dialog."""
        try:
            self.logger.info("Saving configuration")

            if self.status_bar:
                self.status_bar.set_processing_status("Saving configuration...")

            # Update application configuration
            self.config = config

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

        if self.pod_list:
            await self.pod_list.refresh_pods()

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
        """Export current logs."""
        if self.log_viewer:
            # TODO: Implement log export
            self.logger.info("Exporting logs")
            if self.status_bar:
                self.status_bar.set_processing_status("Exporting logs...")

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
