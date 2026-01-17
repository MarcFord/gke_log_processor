"""Configuration dialog for application settings."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TextArea,
)

from ...core.config import ClusterConfig, Config, ProfileConfig


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
    selected_cluster: reactive[str] = reactive("__new__")
    selected_profile: reactive[str] = reactive("__new__")
    selected_template: reactive[str] = reactive("")

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
        self.templates = self.config.available_templates

        if self.config.active_cluster and self.config.get_cluster(self.config.active_cluster):
            self.selected_cluster = self.config.active_cluster
        elif self.config.clusters:
            self.selected_cluster = self.config.clusters[0].name
        else:
            self.selected_cluster = "__new__"

        if self.config.active_profile and self.config.get_profile(self.config.active_profile):
            self.selected_profile = self.config.active_profile
        elif self.config.profiles:
            # take first profile key
            self.selected_profile = next(iter(self.config.profiles))
        else:
            self.selected_profile = "__new__"

        template_names = list(self.templates.keys())
        if template_names:
            self.selected_template = template_names[0]

        self.profile_form = {
            "name": "",
            "cluster": None,
            "namespace": "",
            "description": "",
        }

    # Helper data providers -------------------------------------------------
    def _cluster_options(self) -> List[tuple[str, str]]:
        options: List[tuple[str, str]] = [("➕ New cluster", "__new__")]
        options.extend((cluster.name, cluster.name) for cluster in self.config.clusters)
        return options

    def _profile_options(self) -> List[tuple[str, str]]:
        options: List[tuple[str, str]] = [("➕ New profile", "__new__")]
        options.extend((name, name) for name in self.config.profiles)
        return options

    def _template_options(self) -> List[tuple[str, str]]:
        return [(name, name) for name in self.templates.keys()]

    # Cluster form helpers --------------------------------------------------
    def _load_cluster_into_form(self, cluster_name: str) -> None:
        cluster = self.config.get_cluster(cluster_name)
        if not cluster:
            return
        self.config.gke.cluster_name = cluster.name
        self.config.gke.project_id = cluster.project_id
        self.config.gke.zone = cluster.zone
        self.config.gke.region = cluster.region
        self.config.kubernetes.default_namespace = cluster.namespace
        self.selected_cluster = cluster.name
        self._refresh_gke_inputs()

    def _refresh_gke_inputs(self) -> None:
        try:
            cluster_select = self.query_one("#cluster-select", Select)
            cluster_select.set_options(self._cluster_options())
            cluster_select.value = self.selected_cluster
        except Exception:
            pass
        try:
            self.query_one("#project-input", Input).value = self.config.gke.project_id or ""
            self.query_one("#cluster-input", Input).value = self.config.gke.cluster_name or ""
            self.query_one("#namespace-input", Input).value = self.config.kubernetes.default_namespace
            location_input = self.query_one("#location-input", Input)
            if self.config.gke.region:
                self.connection_type = "region"
                location_input.value = self.config.gke.region
            else:
                self.connection_type = "zone"
                location_input.value = self.config.gke.zone or ""
            self.query_one("#type-select", Select).value = self.connection_type
        except Exception:
            pass

    # Profile form helpers --------------------------------------------------
    def _load_profile_into_form(self, profile_name: str) -> None:
        profile = self.config.get_profile(profile_name)
        if not profile:
            return
        self.profile_form.update(
            {
                "name": profile.name,
                "cluster": profile.cluster,
                "namespace": profile.namespace or "",
                "description": profile.description or "",
            }
        )
        self.selected_profile = profile.name
        self._refresh_profile_inputs()

    def _refresh_profile_inputs(self) -> None:
        try:
            profile_select = self.query_one("#profile-select", Select)
            profile_select.set_options(self._profile_options())
            profile_select.value = self.selected_profile
        except Exception:
            pass
        try:
            self.query_one("#profile-name-input", Input).value = self.profile_form.get("name", "")
            cluster_select = self.query_one("#profile-cluster-select", Select)
            cluster_select.set_options(self._cluster_options())
            cluster_value = self.profile_form.get("cluster") or "__new__"
            cluster_select.value = cluster_value
            self.query_one("#profile-namespace-input", Input).value = self.profile_form.get("namespace", "")
            self.query_one("#profile-description-input", TextArea).value = self.profile_form.get("description", "")
        except Exception:
            pass

    # Template helpers ------------------------------------------------------
    def _refresh_template_inputs(self) -> None:
        try:
            template_select = self.query_one("#template-select", Select)
            template_select.set_options(self._template_options())
            if self.selected_template:
                template_select.value = self.selected_template
        except Exception:
            pass
        try:
            markdown = self.query_one("#template-summary", Markdown)
            if self.selected_template and self.selected_template in self.templates:
                summary = "\n".join(
                    f"• {line}" for line in self.templates[self.selected_template].summary_lines()
                )
                markdown.update(f"**{self.selected_template}**\n{summary}")
            else:
                markdown.update("Select a template to view details")
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        """Compose the dialog."""
        with Vertical(classes="dialog-container"):
            yield Static("⚙️ Application Configuration", id="dialog-title")

            # Tab buttons
            with Horizontal(classes="config-tabs"):
                yield Button("🏗️ GKE", id="gke-tab", classes="tab-button active")
                yield Button("🔧 Logging", id="logging-tab", classes="tab-button")
                yield Button("🤖 AI", id="ai-tab", classes="tab-button")
                yield Button("🧩 Profiles", id="profiles-tab", classes="tab-button")
                yield Button("📦 Templates", id="templates-tab", classes="tab-button")

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
        for tab_id in ["gke-tab", "logging-tab", "ai-tab", "profiles-tab", "templates-tab"]:
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
        elif self.current_tab == "profiles":
            return self._render_profiles_tab()
        elif self.current_tab == "templates":
            return self._render_templates_tab()
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

    def _render_profiles_tab(self) -> Vertical:
        """Render configuration profiles tab."""
        profile_options = self._profile_options()
        cluster_options = self._cluster_options()
        profile_value = (
            self.selected_profile
            if any(value == self.selected_profile for _, value in profile_options)
            else "__new__"
        )
        cluster_value = (
            self.profile_form.get("cluster")
            if self.profile_form.get("cluster") in [value for _, value in cluster_options]
            else "__new__"
        )

        return Vertical(
            Grid(
                Static("🧩 Profile Management", classes="section-header"),
                Label(""),

                Label("Saved Profiles:", id="profile-select-label"),
                Select(options=profile_options, value=profile_value, id="profile-select"),

                Label("Profile Name:", id="profile-name-label"),
                Input(
                    value=self.profile_form.get("name", ""),
                    placeholder="staging-canary",
                    id="profile-name-input"
                ),

                Label("Cluster:", id="profile-cluster-label"),
                Select(options=cluster_options, value=cluster_value, id="profile-cluster-select"),

                Label("Namespace Override:", id="profile-namespace-label"),
                Input(
                    value=self.profile_form.get("namespace", ""),
                    placeholder="default",
                    id="profile-namespace-input"
                ),

                Label("Description:", id="profile-description-label"),
                TextArea(
                    value=self.profile_form.get("description", ""),
                    placeholder="Describe when to use this profile",
                    id="profile-description-input"
                ),

                classes="form-grid"
            ),
            Horizontal(
                Button("💾 Save Profile", id="save-profile-button", variant="success"),
                Button("✅ Activate", id="activate-profile-button", variant="primary"),
                Button("🗑️ Delete", id="delete-profile-button", variant="error"),
                classes="button-row"
            ),
            id="profiles-form"
        )

    def _render_templates_tab(self) -> Vertical:
        """Render built-in templates tab."""
        template_options = self._template_options()
        template_value = (
            self.selected_template
            if self.selected_template in [value for _, value in template_options]
            else (template_options[0][1] if template_options else "")
        )

        summary_text = "Select a template to view details"
        if template_value and template_value in self.templates:
            summary_lines = self.templates[template_value].summary_lines()
            summary_text = "\n".join(f"• {line}" for line in summary_lines)

        return Vertical(
            Grid(
                Static("📦 Environment Templates", classes="section-header"),
                Label(""),

                Label("Available Templates:", id="template-label"),
                Select(options=template_options, value=template_value, id="template-select"),

                Static("Template Summary:", id="template-summary-label"),
                Markdown(f"**{template_value or 'Templates'}**\n{summary_text}", id="template-summary"),

                classes="form-grid"
            ),
            Horizontal(
                Button("⚙️ Apply Template", id="apply-template-button", variant="primary"),
                Button("🔄 Reload Summary", id="refresh-template-button", variant="default"),
                classes="button-row"
            ),
            id="templates-form"
        )

    def _update_form_content(self) -> None:
        """Update the form content based on current tab."""
        try:
            # Remove existing form
            existing_forms = [
                "gke-form",
                "logging-form",
                "ai-form",
                "profiles-form",
                "templates-form",
            ]
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
        elif event.button.id == "profiles-tab":
            self.current_tab = "profiles"
        elif event.button.id == "templates-tab":
            self.current_tab = "templates"
        elif event.button.id == "reset-button":
            await self._reset_config()
        elif event.button.id == "cancel-button":
            self.post_message(self.ConfigCancelled())
            self.dismiss()
        elif event.button.id == "save-button":
            await self._save_config()
        elif event.button.id == "save-cluster-button":
            await self._handle_save_cluster()
        elif event.button.id == "delete-cluster-button":
            await self._handle_delete_cluster()
        elif event.button.id == "activate-cluster-button":
            await self._handle_activate_cluster()
        elif event.button.id == "save-profile-button":
            await self._handle_save_profile()
        elif event.button.id == "delete-profile-button":
            await self._handle_delete_profile()
        elif event.button.id == "activate-profile-button":
            await self._handle_activate_profile()
        elif event.button.id == "apply-template-button":
            await self._handle_apply_template()
        elif event.button.id == "refresh-template-button":
            self._refresh_template_inputs()

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
            elif event.input.id == "profile-name-input":
                self.profile_form["name"] = event.value.strip()
            elif event.input.id == "profile-namespace-input":
                self.profile_form["namespace"] = event.value.strip()
        except Exception as e:
            self._show_error(f"Invalid input: {e}")

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "level-select":
            self.config.logging.level = str(event.value)
        elif event.select.id == "model-select":
            self.config.ai.model_name = str(event.value)
        elif event.select.id == "cluster-select":
            self.selected_cluster = str(event.value)
            if self.selected_cluster == "__new__":
                self.config.gke.cluster_name = None
                self.config.gke.project_id = None
                self.config.gke.zone = None
                self.config.gke.region = None
                self._refresh_gke_inputs()
            else:
                self._load_cluster_into_form(self.selected_cluster)
        elif event.select.id == "profile-select":
            self.selected_profile = str(event.value)
            if self.selected_profile == "__new__":
                self.profile_form.update({
                    "name": "",
                    "cluster": None,
                    "namespace": "",
                    "description": "",
                })
                self._refresh_profile_inputs()
            else:
                self._load_profile_into_form(self.selected_profile)
        elif event.select.id == "profile-cluster-select":
            value = str(event.value)
            self.profile_form["cluster"] = value if value != "__new__" else None
        elif event.select.id == "template-select":
            self.selected_template = str(event.value)
            self._refresh_template_inputs()

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

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area updates."""
        if event.text_area.id == "profile-description-input":
            self.profile_form["description"] = event.value

    async def _reset_config(self) -> None:
        """Reset configuration to original values."""
        self.config = self.original_config.model_copy()
        self.templates = self.config.available_templates
        if self.config.active_cluster and self.config.get_cluster(self.config.active_cluster):
            self.selected_cluster = self.config.active_cluster
        elif self.config.clusters:
            self.selected_cluster = self.config.clusters[0].name
        else:
            self.selected_cluster = "__new__"

        if self.config.active_profile and self.config.get_profile(self.config.active_profile):
            self.selected_profile = self.config.active_profile
        elif self.config.profiles:
            self.selected_profile = next(iter(self.config.profiles))
        else:
            self.selected_profile = "__new__"

        template_names = list(self.templates.keys())
        if template_names:
            self.selected_template = template_names[0]
        else:
            self.selected_template = ""

        if self.selected_cluster != "__new__":
            self._load_cluster_into_form(self.selected_cluster)
        else:
            self.config.gke.cluster_name = None
            self.config.gke.project_id = None
            self.config.gke.zone = None
            self.config.gke.region = None

        if self.selected_profile != "__new__":
            self._load_profile_into_form(self.selected_profile)
        else:
            self.profile_form.update({
                "name": "",
                "cluster": None,
                "namespace": "",
                "description": "",
            })

        if self.selected_template and self.selected_template not in self.templates:
            self.selected_template = ""

        self._update_form_content()
        if self.current_tab == "gke":
            self._refresh_gke_inputs()
        elif self.current_tab == "profiles":
            self._refresh_profile_inputs()
        elif self.current_tab == "templates":
            self._refresh_template_inputs()
        self._show_success("Configuration reset to original values")

    async def _handle_save_cluster(self) -> None:
        """Persist current cluster settings into saved clusters."""
        name = (self.config.gke.cluster_name or "").strip()
        project = (self.config.gke.project_id or "").strip()
        zone = (self.config.gke.zone or "").strip() or None
        region = (self.config.gke.region or "").strip() or None
        namespace = (self.config.kubernetes.default_namespace or "default").strip()

        if not name or not project or (not zone and not region):
            self._show_error("Cluster name, project, and either zone or region are required")
            return

        try:
            cluster = ClusterConfig(
                name=name,
                project_id=project,
                zone=zone,
                region=region,
                namespace=namespace,
            )
        except Exception as exc:
            self._show_error(f"Invalid cluster settings: {exc}")
            return

        self.config.upsert_cluster(cluster)
        self.selected_cluster = cluster.name
        self._refresh_gke_inputs()
        self._refresh_profile_inputs()
        self._show_success(f"Cluster '{cluster.name}' saved")

    async def _handle_delete_cluster(self) -> None:
        """Remove the selected cluster from saved definitions."""
        if self.selected_cluster in {"", "__new__"}:
            self._show_error("Select a saved cluster to delete")
            return

        removed = self.config.remove_cluster(self.selected_cluster)
        if not removed:
            self._show_error(f"Cluster '{self.selected_cluster}' not found")
            return

        self.selected_cluster = self.config.active_cluster or (
            self.config.clusters[0].name if self.config.clusters else "__new__"
        )
        self._refresh_gke_inputs()
        self._refresh_profile_inputs()
        self._show_success("Cluster removed")

    async def _handle_activate_cluster(self) -> None:
        """Mark the selected cluster as active and sync form fields."""
        if self.selected_cluster in {"", "__new__"}:
            self._show_error("Select a saved cluster to activate")
            return

        try:
            self.config.set_active_cluster(self.selected_cluster)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._refresh_gke_inputs()
        self._show_success(f"Cluster '{self.selected_cluster}' activated")

    async def _handle_save_profile(self) -> None:
        """Save or update the current profile form."""
        name = (self.profile_form.get("name") or "").strip()
        cluster = self.profile_form.get("cluster")
        namespace = (self.profile_form.get("namespace") or "").strip() or None
        description = (self.profile_form.get("description") or "").strip() or None

        if not name:
            self._show_error("Profile name is required")
            return
        if not cluster or cluster == "__new__":
            self._show_error("Select a target cluster for the profile")
            return
        if not self.config.get_cluster(cluster):
            self._show_error(f"Cluster '{cluster}' must be saved before creating a profile")
            return

        profile = ProfileConfig(
            name=name,
            cluster=cluster,
            namespace=namespace,
            description=description,
        )

        self.config.upsert_profile(profile)
        self.selected_profile = profile.name
        self.profile_form.update(
            {
                "name": profile.name,
                "cluster": profile.cluster,
                "namespace": profile.namespace or "",
                "description": profile.description or "",
            }
        )
        self._refresh_profile_inputs()
        self._show_success(f"Profile '{profile.name}' saved")

    async def _handle_delete_profile(self) -> None:
        """Delete the currently selected profile."""
        if self.selected_profile in {"", "__new__"}:
            self._show_error("Select a saved profile to delete")
            return

        removed = self.config.remove_profile(self.selected_profile)
        if not removed:
            self._show_error(f"Profile '{self.selected_profile}' not found")
            return

        self.selected_profile = self.config.active_profile or (
            next(iter(self.config.profiles)) if self.config.profiles else "__new__"
        )
        if self.selected_profile != "__new__":
            self._load_profile_into_form(self.selected_profile)
        else:
            self.profile_form.update({
                "name": "",
                "cluster": None,
                "namespace": "",
                "description": "",
            })
            self._refresh_profile_inputs()
        self._show_success("Profile removed")

    async def _handle_activate_profile(self) -> None:
        """Apply the selected profile to the configuration."""
        if self.selected_profile in {"", "__new__"}:
            self._show_error("Select a saved profile to activate")
            return

        try:
            self.config.apply_profile(self.selected_profile)
        except Exception as exc:
            self._show_error(str(exc))
            return

        profile = self.config.get_profile(self.selected_profile)
        self.profile_form.update({
            "name": self.selected_profile,
            "cluster": self.config.active_cluster,
            "namespace": self.config.kubernetes.default_namespace,
            "description": profile.description if profile and profile.description else "",
        })
        self.selected_cluster = self.config.active_cluster or self.selected_cluster
        self._refresh_gke_inputs()
        self._refresh_profile_inputs()
        self._show_success(f"Profile '{self.selected_profile}' activated")

    async def _handle_apply_template(self) -> None:
        """Apply the selected environment template to the configuration."""
        if not self.selected_template:
            self._show_error("Select a template to apply")
            return
        try:
            self.config.apply_template(self.selected_template)
        except Exception as exc:
            self._show_error(f"Failed to apply template: {exc}")
            return

        # Refresh selections from updated config
        self.selected_cluster = self.config.active_cluster or (
            self.config.clusters[0].name if self.config.clusters else "__new__"
        )
        self.selected_profile = self.config.active_profile or (
            next(iter(self.config.profiles)) if self.config.profiles else "__new__"
        )
        if self.selected_profile != "__new__":
            self._load_profile_into_form(self.selected_profile)
        else:
            self.profile_form.update({
                "name": "",
                "cluster": None,
                "namespace": "",
                "description": "",
            })
        self._refresh_gke_inputs()
        self._refresh_profile_inputs()
        self._refresh_template_inputs()
        self._show_success(f"Template '{self.selected_template}' applied")

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
