"""Configuration dialog for application settings."""

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TextArea

from ...core.config import Config


class ConfigDialog(ModalScreen):
    """Modal dialog for editing application configuration."""

    DEFAULT_CSS = """
    ConfigDialog {
        align: center middle;
    }

    .dialog-container {
        width: 90;
        height: 30;
        background: $surface;
        border: thick $primary;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
    }

    .config-tabs {
        height: 3;
        dock: top;
        margin-bottom: 1;
    }

    .form-grid {
        height: 20;
        grid-size: 2 8;
        grid-gutter: 1 2;
    }

    .button-row {
        dock: bottom;
        height: 3;
        margin-top: 1;
    }

    .section-header {
        color: $accent;
        text-style: bold;
        grid-span: 2;
        height: 1;
        margin: 1 0;
    }

    .error-message {
        color: $error;
        background: $error 10%;
        border: solid $error;
        height: 3;
        padding: 1;
        margin: 1 0;
        grid-span: 2;
    }

    Input {
        width: 1fr;
    }

    Select {
        width: 1fr;
    }

    TextArea {
        width: 1fr;
        height: 3;
    }

    Button {
        margin: 0 1;
        min-width: 12;
    }

    .tab-button {
        min-width: 15;
    }

    .tab-button.active {
        background: $accent;
        color: $text;
    }
    """

    # Configuration sections
    current_tab: reactive[str] = reactive("gke")

    class ConfigSaved(Message):
        """Message sent when configuration is saved."""

        def __init__(self, config: Config) -> None:
            self.config = config
            super().__init__()

    class ConfigCancelled(Message):
        """Message sent when configuration dialog is cancelled."""
        pass

    def __init__(self, config: Optional[Config] = None):
        """Initialize configuration dialog."""
        super().__init__()
        self.config = config or Config()
        self.original_config = self.config.model_copy() if config else Config()
        self.error_message = ""

    def compose(self) -> ComposeResult:
        """Compose the dialog."""
        with Vertical(classes="dialog-container"):
            yield Static("⚙️ Application Configuration", id="dialog-title")

            # Tab buttons
            with Horizontal(classes="config-tabs"):
                yield Button("🏗️ GKE", id="gke-tab", classes="tab-button active")
                yield Button("🔧 Logging", id="logging-tab", classes="tab-button")
                yield Button("🤖 AI", id="ai-tab", classes="tab-button")

            # Error message area (initially hidden)
            yield Static("", id="error-message", classes="error-message")

            # Configuration form
            yield self._render_current_tab()

            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("🔄 Reset", id="reset-button", variant="default")
                yield Button("❌ Cancel", id="cancel-button", variant="error")
                yield Button("💾 Save", id="save-button", variant="success")

    def on_mount(self) -> None:
        """Handle dialog mount."""
        # Hide error message initially
        error_widget = self.query_one("#error-message", Static)
        error_widget.display = False

    def watch_current_tab(self, tab: str) -> None:
        """Handle tab changes."""
        # Update tab button styles
        for tab_id in ["gke-tab", "logging-tab", "ai-tab"]:
            button = self.query_one(f"#{tab_id}", Button)
            if tab_id == f"{tab}-tab":
                button.add_class("active")
            else:
                button.remove_class("active")

        # Re-render the form
        self._update_form_content()

    def _render_current_tab(self) -> Vertical:
        """Render the current tab content."""
        if self.current_tab == "gke":
            return self._render_gke_tab()
        elif self.current_tab == "logging":
            return self._render_logging_tab()
        elif self.current_tab == "ai":
            return self._render_ai_tab()
        else:
            return Vertical()

    def _render_gke_tab(self) -> Vertical:
        """Render GKE configuration tab."""
        return Vertical(
            Grid(
                # GKE Section
                Static("🏗️ GKE Configuration", classes="section-header"),
                Label(""),

                Label("Default Project ID:", id="project-label"),
                Input(
                    value=self.config.gke.project_id or "",
                    placeholder="my-gcp-project",
                    id="project-input"
                ),

                Label("Default Cluster:", id="cluster-label"),
                Input(
                    value=self.config.gke.cluster_name or "",
                    placeholder="my-gke-cluster",
                    id="cluster-input"
                ),

                Label("Default Zone:", id="zone-label"),
                Input(
                    value=self.config.gke.zone or "",
                    placeholder="us-central1-a",
                    id="zone-input"
                ),

                Label("Default Region:", id="region-label"),
                Input(
                    value=self.config.gke.region or "",
                    placeholder="us-central1",
                    id="region-input"
                ),

                # Kubernetes Section
                Static("☸️ Kubernetes Configuration", classes="section-header"),
                Label(""),

                Label("Default Namespace:", id="namespace-label"),
                Input(
                    value=self.config.kubernetes.default_namespace,
                    placeholder="default",
                    id="namespace-input"
                ),

                Label("Request Timeout (s):", id="timeout-label"),
                Input(
                    value=str(self.config.kubernetes.request_timeout_seconds),
                    placeholder="30",
                    id="timeout-input"
                ),

                classes="form-grid"
            ),
            id="gke-form"
        )

    def _render_logging_tab(self) -> Vertical:
        """Render logging configuration tab."""
        return Vertical(
            Grid(
                # Logging Section
                Static("📝 Logging Configuration", classes="section-header"),
                Label(""),

                Label("Log Level:", id="level-label"),
                Select(
                    options=[
                        ("DEBUG", "DEBUG"),
                        ("INFO", "INFO"),
                        ("WARNING", "WARNING"),
                        ("ERROR", "ERROR")
                    ],
                    value=self.config.logging.level,
                    id="level-select"
                ),

                Label("Console Logging:", id="console-label"),
                Checkbox(
                    value=self.config.logging.console_enabled,
                    id="console-checkbox"
                ),

                Label("File Logging:", id="file-label"),
                Checkbox(
                    value=self.config.logging.file_enabled,
                    id="file-checkbox"
                ),

                Label("Log File Path:", id="path-label"),
                Input(
                    value=self.config.logging.file_path,
                    placeholder="/tmp/gke-logs.log",
                    id="path-input"
                ),

                Label("Structured Logging:", id="structured-label"),
                Checkbox(
                    value=self.config.logging.structured,
                    id="structured-checkbox"
                ),

                Label("Log Rotation (MB):", id="rotation-label"),
                Input(
                    value=str(self.config.logging.max_file_size_mb),
                    placeholder="100",
                    id="rotation-input"
                ),

                classes="form-grid"
            ),
            id="logging-form"
        )

    def _render_ai_tab(self) -> Vertical:
        """Render AI configuration tab."""
        return Vertical(
            Grid(
                # AI Section
                Static("🤖 AI Configuration", classes="section-header"),
                Label(""),

                Label("Gemini API Key:", id="api-key-label"),
                Input(
                    value=self.config.ai.gemini_api_key or "",
                    placeholder="your-gemini-api-key",
                    password=True,
                    id="api-key-input"
                ),

                Label("Model Name:", id="model-label"),
                Select(
                    options=[
                        ("gemini-1.5-flash", "gemini-1.5-flash"),
                        ("gemini-1.5-pro", "gemini-1.5-pro"),
                        ("gemini-1.0-pro", "gemini-1.0-pro")
                    ],
                    value=self.config.ai.model_name,
                    id="model-select"
                ),

                Label("Max Tokens:", id="tokens-label"),
                Input(
                    value=str(self.config.ai.max_tokens),
                    placeholder="2048",
                    id="tokens-input"
                ),

                Label("Temperature:", id="temperature-label"),
                Input(
                    value=str(self.config.ai.temperature),
                    placeholder="0.7",
                    id="temperature-input"
                ),

                Label("Analysis Enabled:", id="analysis-label"),
                Checkbox(
                    value=self.config.ai.analysis_enabled,
                    id="analysis-checkbox"
                ),

                Label("Query Timeout (s):", id="query-timeout-label"),
                Input(
                    value=str(self.config.ai.query_timeout_seconds),
                    placeholder="30",
                    id="query-timeout-input"
                ),

                classes="form-grid"
            ),
            id="ai-form"
        )

    def _update_form_content(self) -> None:
        """Update the form content based on current tab."""
        try:
            # Remove existing form
            existing_forms = ["gke-form", "logging-form", "ai-form"]
            for form_id in existing_forms:
                try:
                    form = self.query_one(f"#{form_id}")
                    form.remove()
                except BaseException:
                    pass

            # Add new form content
            container = self.query_one(".dialog-container", Vertical)
            new_form = self._render_current_tab()

            # Insert before the button row
            button_row = self.query_one(".button-row", Horizontal)
            container.mount(new_form, before=button_row)

        except Exception as e:
            self._show_error(f"Error updating form: {e}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "gke-tab":
            self.current_tab = "gke"
        elif event.button.id == "logging-tab":
            self.current_tab = "logging"
        elif event.button.id == "ai-tab":
            self.current_tab = "ai"
        elif event.button.id == "reset-button":
            await self._reset_config()
        elif event.button.id == "cancel-button":
            self.post_message(self.ConfigCancelled())
            self.dismiss()
        elif event.button.id == "save-button":
            await self._save_config()

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        try:
            # Update config based on input changes
            if event.input.id == "project-input":
                self.config.gke.project_id = event.value.strip() or None
            elif event.input.id == "cluster-input":
                self.config.gke.cluster_name = event.value.strip() or None
            elif event.input.id == "zone-input":
                self.config.gke.zone = event.value.strip() or None
            elif event.input.id == "region-input":
                self.config.gke.region = event.value.strip() or None
            elif event.input.id == "namespace-input":
                self.config.kubernetes.default_namespace = event.value.strip() or "default"
            elif event.input.id == "timeout-input":
                try:
                    self.config.kubernetes.request_timeout_seconds = int(event.value)
                except ValueError:
                    pass  # Keep old value
            elif event.input.id == "path-input":
                self.config.logging.file_path = event.value.strip()
            elif event.input.id == "rotation-input":
                try:
                    self.config.logging.max_file_size_mb = int(event.value)
                except ValueError:
                    pass
            elif event.input.id == "api-key-input":
                self.config.ai.gemini_api_key = event.value.strip() or None
            elif event.input.id == "tokens-input":
                try:
                    self.config.ai.max_tokens = int(event.value)
                except ValueError:
                    pass
            elif event.input.id == "temperature-input":
                try:
                    self.config.ai.temperature = float(event.value)
                except ValueError:
                    pass
            elif event.input.id == "query-timeout-input":
                try:
                    self.config.ai.query_timeout_seconds = int(event.value)
                except ValueError:
                    pass
        except Exception as e:
            self._show_error(f"Invalid input: {e}")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "level-select":
            self.config.logging.level = str(event.value)
        elif event.select.id == "model-select":
            self.config.ai.model_name = str(event.value)

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "console-checkbox":
            self.config.logging.console_enabled = event.value
        elif event.checkbox.id == "file-checkbox":
            self.config.logging.file_enabled = event.value
        elif event.checkbox.id == "structured-checkbox":
            self.config.logging.structured = event.value
        elif event.checkbox.id == "analysis-checkbox":
            self.config.ai.analysis_enabled = event.value

    async def _reset_config(self) -> None:
        """Reset configuration to original values."""
        self.config = self.original_config.model_copy()
        self._update_form_content()
        self._show_success("Configuration reset to original values")

    async def _save_config(self) -> None:
        """Save the configuration."""
        try:
            # Validate configuration
            if not self._validate_config():
                return

            # Show saving status
            save_button = self.query_one("#save-button", Button)
            original_label = save_button.label
            save_button.label = "💾 Saving..."
            save_button.disabled = True

            # Simulate save delay
            await asyncio.sleep(0.5)

            # Post the save message
            self.post_message(self.ConfigSaved(self.config))
            self.dismiss()

        except Exception as e:
            self._show_error(f"Failed to save configuration: {e}")
            save_button = self.query_one("#save-button", Button)
            save_button.label = original_label
            save_button.disabled = False

    def _validate_config(self) -> bool:
        """Validate the current configuration."""
        # Basic validation
        if self.config.ai.temperature < 0 or self.config.ai.temperature > 2:
            self._show_error("AI temperature must be between 0 and 2")
            return False

        if self.config.ai.max_tokens < 1:
            self._show_error("Max tokens must be positive")
            return False

        if self.config.kubernetes.request_timeout_seconds < 1:
            self._show_error("Request timeout must be positive")
            return False

        if self.config.logging.max_file_size_mb < 1:
            self._show_error("Log file size must be positive")
            return False

        return True

    def _show_error(self, message: str) -> None:
        """Show error message."""
        error_widget = self.query_one("#error-message", Static)
        error_widget.update(f"❌ {message}")
        error_widget.styles.color = "red"
        error_widget.styles.background = "red 10%"
        error_widget.styles.border = "solid red"
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
