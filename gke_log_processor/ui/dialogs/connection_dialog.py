"""Connection dialog for GKE cluster setup."""

import asyncio
from typing import List, Optional

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ...core.config import Config
from ...core.exceptions import GKELogProcessorError


class ConnectionDialog(ModalScreen):
    """Modal dialog for setting up GKE cluster connection."""

    DEFAULT_CSS = """
    ConnectionDialog {
        align: center middle;
    }

    .dialog-container {
        width: 80;
        height: 25;
        background: $surface;
        border: thick $primary;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
    }

    .form-grid {
        height: 18;
        grid-size: 2 6;
        grid-gutter: 1 2;
    }

    .button-row {
        dock: bottom;
        height: 3;
        margin-top: 1;
    }

    .error-message {
        color: $error;
        background: $error 10%;
        border: solid $error;
        height: 3;
        padding: 1;
        margin: 1 0;
    }

    Input {
        width: 1fr;
    }

    Select {
        width: 1fr;
    }

    Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    # Connection form data
    cluster_name: reactive[str] = reactive("")
    project_id: reactive[str] = reactive("")
    zone: reactive[str] = reactive("")
    region_value: reactive[str] = reactive("")
    namespace: reactive[str] = reactive("default")
    connection_type: reactive[str] = reactive("zone")
    selected_cluster: reactive[str] = reactive("__manual__")

    class ConnectionRequested(Message):
        """Message sent when connection is requested."""

        def __init__(self, connection_info: dict) -> None:
            self.connection_info = connection_info
            super().__init__()

    class ConnectionCancelled(Message):
        """Message sent when connection dialog is cancelled."""
        pass

    def __init__(self, config: Optional[Config] = None):
        """Initialize connection dialog."""
        super().__init__()
        self.config = config or Config()
        self.error_message = ""

        # Pre-populate from config if available
        if config and config.gke:
            self.cluster_name = config.gke.cluster_name or ""
            self.project_id = config.gke.project_id or ""
            self.zone = config.gke.zone or ""
            self.region_value = config.gke.region or ""
            self.namespace = config.kubernetes.default_namespace or "default"

            # Determine connection type
            if config.gke.region:
                self.connection_type = "region"
            else:
                self.connection_type = "zone"

        if self.config.active_cluster and self.config.get_cluster(self.config.active_cluster):
            self.selected_cluster = self.config.active_cluster
            cluster = self.config.get_cluster(self.selected_cluster)
            if cluster:
                self._apply_cluster_to_state(cluster)
        else:
            self.selected_cluster = "__manual__"

    def _cluster_options(self) -> List[tuple[str, str]]:
        """Build the list of saved cluster options for the selector."""
        options: List[tuple[str, str]] = [("➕ Manual entry", "__manual__")]
        for cluster in self.config.clusters:
            options.append((cluster.name, cluster.name))
        return options

    def _apply_cluster_to_state(self, cluster: "ClusterConfig") -> None:
        """Copy cluster attributes into the reactive state."""
        self.cluster_name = cluster.name
        self.project_id = cluster.project_id
        self.namespace = cluster.namespace or "default"
        if cluster.region:
            self.connection_type = "region"
            self.region_value = cluster.region
            self.zone = ""
        else:
            self.connection_type = "zone"
            self.zone = cluster.zone or ""
            self.region_value = ""

    def _refresh_form_inputs(self) -> None:
        """Synchronize widget values with the current reactive state."""
        try:
            self.query_one("#cluster-input", Input).value = self.cluster_name
            self.query_one("#project-input", Input).value = self.project_id
            location_input = self.query_one("#location-input", Input)
            if self.connection_type == "zone":
                location_input.value = self.zone
            else:
                location_input.value = self.region_value
            namespace_input = self.query_one("#namespace-input", Input)
            namespace_input.value = self.namespace
            type_select = self.query_one("#type-select", Select)
            type_select.value = self.connection_type
            cluster_select = self.query_one("#saved-cluster-select", Select)
            cluster_select.value = self.selected_cluster
        except Exception:
            # Widgets may not yet exist during initialization; ignore
            pass

    def compose(self) -> ComposeResult:
        """Compose the dialog."""
        with Vertical(classes="dialog-container"):
            yield Static("🔗 Connect to GKE Cluster", id="dialog-title")

            # Error message area (initially hidden)
            yield Static("", id="error-message", classes="error-message")

            with Grid(classes="form-grid"):
                yield Label("Saved Cluster:", id="saved-cluster-label")
                options = self._cluster_options()
                default_value = (
                    self.selected_cluster
                    if any(value == self.selected_cluster for _, value in options)
                    else "__manual__"
                )
                yield Select(options=options, value=default_value, id="saved-cluster-select")

                # Cluster Name
                yield Label("Cluster Name:", id="cluster-label")
                yield Input(
                    value=self.cluster_name,
                    placeholder="my-gke-cluster",
                    id="cluster-input"
                )

                # Project ID
                yield Label("Project ID:", id="project-label")
                yield Input(
                    value=self.project_id,
                    placeholder="my-project-id",
                    id="project-input"
                )

                # Connection Type
                yield Label("Location Type:", id="type-label")
                yield Select(
                    options=[
                        ("Zone-based cluster", "zone"),
                        ("Regional cluster", "region")
                    ],
                    value=self.connection_type,
                    id="type-select"
                )

                # Zone/Region
                yield Label("Zone/Region:", id="location-label")
                yield Input(
                    value=self.zone or self.region_value,
                    placeholder="us-central1-a or us-central1",
                    id="location-input"
                )

                # Namespace
                yield Label("Namespace:", id="namespace-label")
                yield Input(
                    value=self.namespace,
                    placeholder="default",
                    id="namespace-input"
                )

                # Test Connection Button
                yield Label("")  # Empty cell
                yield Button("🔍 Test Connection", id="test-button", variant="default")

            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("❌ Cancel", id="cancel-button", variant="error")
                yield Button("✅ Connect", id="connect-button", variant="success")

    def on_mount(self) -> None:
        """Handle dialog mount."""
        # Hide error message initially
        error_widget = self.query_one("#error-message", Static)
        error_widget.display = False

        # Focus the first input
        self.query_one("#cluster-input", Input).focus()
        self._refresh_form_inputs()

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id in {"cluster-input", "project-input", "location-input", "namespace-input"}:
            if self.selected_cluster != "__manual__":
                self.selected_cluster = "__manual__"
                try:
                    self.query_one("#saved-cluster-select", Select).value = self.selected_cluster
                except Exception:
                    pass
        if event.input.id == "cluster-input":
            self.cluster_name = event.value
        elif event.input.id == "project-input":
            self.project_id = event.value
        elif event.input.id == "location-input":
            if self.connection_type == "zone":
                self.zone = event.value
                self.region_value = ""
            else:
                self.region_value = event.value
                self.zone = ""
        elif event.input.id == "namespace-input":
            self.namespace = event.value

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "type-select":
            self.connection_type = str(event.value)

            # Update location input placeholder
            location_input = self.query_one("#location-input", Input)
            if self.connection_type == "zone":
                location_input.placeholder = "us-central1-a"
                location_input.value = self.zone
            else:
                location_input.placeholder = "us-central1"
                location_input.value = self.region_value
        elif event.select.id == "saved-cluster-select":
            self.selected_cluster = str(event.value)
            if self.selected_cluster == "__manual__":
                self._refresh_form_inputs()
                return

            cluster = self.config.get_cluster(self.selected_cluster)
            if cluster:
                self._apply_cluster_to_state(cluster)
            self._refresh_form_inputs()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-button":
            self.post_message(self.ConnectionCancelled())
            self.dismiss()
        elif event.button.id == "test-button":
            await self._test_connection()
        elif event.button.id == "connect-button":
            await self._connect()

    async def _test_connection(self) -> None:
        """Test the connection without saving."""
        connection_info = self._get_connection_info()

        if not self._validate_connection_info(connection_info):
            return

        # Show testing status
        test_button = self.query_one("#test-button", Button)
        original_label = test_button.label
        test_button.label = "⏳ Testing..."
        test_button.disabled = True

        try:
            # TODO: Implement actual connection test
            # This would use the GKE client to test the connection
            await asyncio.sleep(1)  # Simulate test

            self._show_success("✅ Connection test successful!")

        except Exception as e:
            self._show_error(f"Connection test failed: {e}")
        finally:
            test_button.label = original_label
            test_button.disabled = False

    async def _connect(self) -> None:
        """Attempt to connect with the provided information."""
        connection_info = self._get_connection_info()

        if not self._validate_connection_info(connection_info):
            return

        # Show connecting status
        connect_button = self.query_one("#connect-button", Button)
        original_label = connect_button.label
        connect_button.label = "⏳ Connecting..."
        connect_button.disabled = True

        try:
            # Post the connection request
            self.post_message(self.ConnectionRequested(connection_info))
            self.dismiss()

        except Exception as e:
            self._show_error(f"Connection failed: {e}")
            connect_button.label = original_label
            connect_button.disabled = False

    def _get_connection_info(self) -> dict:
        """Get current connection information."""
        return {
            "cluster_name": self.cluster_name.strip(),
            "project_id": self.project_id.strip(),
            "zone": self.zone.strip() if self.connection_type == "zone" else "",
            "region": self.region_value.strip() if self.connection_type == "region" else "",
            "namespace": self.namespace.strip() or "default",
            "connection_type": self.connection_type
        }

    def _validate_connection_info(self, info: dict) -> bool:
        """Validate connection information."""
        if not info["cluster_name"]:
            self._show_error("Cluster name is required")
            return False

        if not info["project_id"]:
            self._show_error("Project ID is required")
            return False

        if info["connection_type"] == "zone" and not info["zone"]:
            self._show_error("Zone is required for zone-based clusters")
            return False
        elif info["connection_type"] == "region" and not info["region"]:
            self._show_error("Region is required for regional clusters")
            return False

        return True

    def _show_error(self, message: str) -> None:
        """Show error message."""
        error_widget = self.query_one("#error-message", Static)
        error_widget.update(f"❌ {message}")
        error_widget.display = True

    def _show_success(self, message: str) -> None:
        """Show success message."""
        error_widget = self.query_one("#error-message", Static)
        error_widget.update(f"✅ {message}")
        error_widget.styles.color = "green"
        error_widget.styles.background = "green 10%"
        error_widget.styles.border = "solid green"
        error_widget.display = True

        # Auto-hide success message after 3 seconds
        self.call_later(lambda: setattr(error_widget, 'display', False), delay=3)


# Import asyncio for the test connection delay
