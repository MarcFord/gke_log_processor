"""Widget that summarizes configuration and provides quick actions."""

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Markdown, Static

from ...core.config import Config


class ConfigManagerWidget(Widget):
    """Show current configuration state with quick actions."""

    DEFAULT_CSS = """
    ConfigManagerWidget {
        border: solid $primary;
        height: 100%;
    }

    ConfigManagerWidget > .config-header {
        dock: top;
        height: 3;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }

    ConfigManagerWidget > .config-toolbar {
        dock: top;
        height: 3;
        background: $surface;
        column-gap: 1;
        padding: 0 1;
    }

    ConfigManagerWidget > .config-summary {
        padding: 1;
    }
    """

    class EditConfigRequested(Message):
        """Request to open the configuration dialog."""

    class TestConnectionRequested(Message):
        """Request to validate the current cluster configuration."""

    class ReloadConfigRequested(Message):
        """Request to reload configuration from disk."""

    def __init__(self, config: Optional[Config] = None, *, name: Optional[str] = None,
                 id: Optional[str] = None, classes: Optional[str] = None) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._config = config or Config()
        self._summary = Markdown(self._render_summary(), id="config-summary-text", classes="config-summary")

    @property
    def config(self) -> Config:
        return self._config

    def compose(self) -> ComposeResult:
        yield Static("⚙️ Configuration", classes="config-header")
        with Horizontal(classes="config-toolbar"):
            yield Button("✏️ Edit", id="edit-config", variant="primary")
            yield Button("🧪 Test", id="test-config", variant="default")
            yield Button("🔁 Reload", id="reload-config", variant="default")
        yield self._summary

    def update_config(self, config: Config) -> None:
        """Replace stored config and refresh summary."""

        self._config = config
        self._summary.update(self._render_summary())

    # Internal helpers --------------------------------------------------
    def _render_summary(self) -> str:
        gke = self._config.gke
        kubernetes = self._config.kubernetes
        ai = self._config.ai

        location = gke.zone or gke.region or "(not set)"
        api_key_state = "set" if ai.gemini_api_key else "missing"

        return (
            "**GKE**\n"
            f"• Project: {gke.project_id or '(not set)'}\n"
            f"• Cluster: {gke.cluster_name or '(not set)'}\n"
            f"• Location: {location}\n\n"
            "**Kubernetes**\n"
            f"• Namespace: {kubernetes.default_namespace}\n"
            f"• Request timeout: {kubernetes.request_timeout_seconds}s\n\n"
            "**AI**\n"
            f"• Model: {ai.model_name}\n"
            f"• API key: {api_key_state}\n"
            f"• Max tokens: {ai.max_tokens}"
        )

    # Event handlers ----------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "edit-config":
            self.post_message(self.EditConfigRequested())
        elif button_id == "test-config":
            self.post_message(self.TestConnectionRequested())
        elif button_id == "reload-config":
            self.post_message(self.ReloadConfigRequested())
