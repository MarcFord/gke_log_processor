"""Composite widget that wraps the pod list with additional controls."""

from typing import Optional, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from ...core.models import PodInfo
from ..components.pod_list import PodListWidget


class PodSelector(Widget):
    """Wrapper around ``PodListWidget`` with namespace controls."""

    DEFAULT_CSS = """
    PodSelector {
        border: solid $primary;
        height: 100%;
    }

    PodSelector > .pod-selector-header {
        dock: top;
        height: 3;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }

    PodSelector > .pod-selector-toolbar {
        dock: top;
        height: 3;
        background: $surface;
        column-gap: 2;
        padding: 0 1;
    }

    PodSelector > PodListWidget {
        height: 1fr;
    }
    """

    namespace: reactive[str] = reactive("default")

    class NamespaceChanged(Message):
        """Namespace filter changed."""

        def __init__(self, namespace: str) -> None:
            self.namespace = namespace
            super().__init__()

    class RefreshRequested(Message):
        """Request to refresh pod information."""

    def __init__(self, *, name: Optional[str] = None, id: Optional[str] = None, classes: Optional[str] = None) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._pod_list = PodListWidget(id="pod-selector-list")
        self._namespace_input = Input(placeholder="Namespace", value="default", id="namespace-filter")

    @property
    def pod_list(self) -> PodListWidget:
        """Expose the underlying pod list widget."""

        return self._pod_list

    def compose(self) -> ComposeResult:
        yield Static("☸️ Pods", classes="pod-selector-header")
        with Horizontal(classes="pod-selector-toolbar"):
            yield self._namespace_input
            yield Button("🔄 Refresh", id="refresh-pods", variant="primary")
        yield self._pod_list

    # Proxy helpers -----------------------------------------------------
    def set_pods(self, pods: Sequence[PodInfo]) -> None:
        self._pod_list.update_pods(list(pods))

    async def refresh_pods(self) -> None:
        self.post_message(self.RefreshRequested())

    def watch_namespace(self, namespace: str) -> None:
        self._namespace_input.value = namespace

    # Event handlers ----------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "namespace-filter":
            namespace = event.value.strip() or "default"
            if namespace != self.namespace:
                self.namespace = namespace
                self.post_message(self.NamespaceChanged(namespace))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-pods":
            self.post_message(self.RefreshRequested())
