"""Pod list widget for selecting and displaying Kubernetes pods."""

from datetime import datetime
from typing import Callable, List, Optional, Set

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive, var
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from ...core.models import ContainerState, PodInfo, PodPhase


class PodListWidget(Widget):
    """Interactive widget for displaying and selecting Kubernetes pods."""

    DEFAULT_CSS = """
    PodListWidget {
        border: solid $primary;
        height: 100%;
        min-height: 15;
    }

    PodListWidget > .pod-header {
        dock: top;
        height: 3;
        background: $panel;
    }

    PodListWidget > .pod-filters {
        dock: top;
        height: 3;
        background: $surface;
    }

    PodListWidget > .pod-table {
        border: none;
    }

    PodListWidget Input {
        margin: 0 1;
    }

    PodListWidget Select {
        margin: 0 1;
        max-width: 15;
    }

    PodListWidget Label {
        margin: 0 1;
        text-align: center;
        content-align: center middle;
    }
    """

    # Reactive attributes
    pods: reactive[List[PodInfo]] = reactive(list, layout=True)
    selected_pods: reactive[Set[str]] = reactive(set, layout=True)
    filter_text: reactive[str] = reactive("", layout=True)
    namespace_filter: reactive[Optional[str]] = reactive(None, layout=True)
    phase_filter: reactive[Optional[PodPhase]] = reactive(None, layout=True)

    class PodSelected(Message):
        """Message sent when pods are selected."""

        def __init__(self, pod_names: Set[str], pods: List[PodInfo]) -> None:
            self.pod_names = pod_names
            self.pods = pods
            super().__init__()

    class RefreshRequested(Message):
        """Message sent when pod list refresh is requested."""
        pass

    def __init__(
        self,
        pods: Optional[List[PodInfo]] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.pods = pods or []
        self._filtered_pods: List[PodInfo] = []
        self._on_selection_change: Optional[Callable[[Set[str], List[PodInfo]], None]] = None

    def compose(self) -> ComposeResult:
        """Compose the pod list widget."""
        with Vertical():
            # Header
            with Horizontal(classes="pod-header"):
                yield Label("📦 Kubernetes Pods", classes="header-label")
                yield Button("🔄 Refresh", id="refresh-button", variant="primary")

            # Filters
            with Horizontal(classes="pod-filters"):
                yield Input(
                    placeholder="Filter pods...",
                    id="pod-filter-input",
                    classes="filter-input"
                )
                yield Select(
                    [("All Namespaces", None)] + [(ns, ns) for ns in self._get_namespaces()],
                    value=None,
                    id="namespace-select",
                    classes="namespace-filter"
                )
                yield Select(
                    [("All Phases", None)] + [(phase.value, phase) for phase in PodPhase],
                    value=None,
                    id="phase-select",
                    classes="phase-filter"
                )

            # Pod table
            yield DataTable(
                id="pod-table",
                classes="pod-table",
                show_header=True,
                zebra_stripes=True,
                cursor_type="row"
            )

    def on_mount(self) -> None:
        """Set up the pod table when the widget is mounted."""
        table = self.query_one("#pod-table", DataTable)

        # Add columns
        table.add_columns(
            "Status", "Name", "Namespace", "Phase", "Ready", "Restarts", "Age"
        )

        # Update table with initial data
        self._update_table()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        if event.input.id == "pod-filter-input":
            self.filter_text = event.value
            self._apply_filters()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select filter changes."""
        if event.select.id == "namespace-select":
            self.namespace_filter = event.value
            self._apply_filters()
        elif event.select.id == "phase-select":
            self.phase_filter = event.value
            self._apply_filters()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-button":
            self.post_message(self.RefreshRequested())

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle pod selection in the table."""
        if event.row_key is None:
            return

        pod_name = str(event.row_key.value)

        # Toggle selection
        if pod_name in self.selected_pods:
            self.selected_pods.remove(pod_name)
        else:
            self.selected_pods.add(pod_name)

        # Update table highlighting
        self._update_selection_highlighting()

        # Get selected pod objects
        selected_pod_objects = [pod for pod in self._filtered_pods if pod.name in self.selected_pods]

        # Send selection message
        self.post_message(self.PodSelected(self.selected_pods.copy(), selected_pod_objects))

    def update_pods(self, new_pods: List[PodInfo]) -> None:
        """Update the pod list with new data."""
        self.pods = new_pods
        self._apply_filters()

    def clear_selection(self) -> None:
        """Clear all selected pods."""
        self.selected_pods.clear()
        self._update_selection_highlighting()

    def select_pods(self, pod_names: Set[str]) -> None:
        """Select specific pods by name."""
        # Filter to only include existing pods
        available_names = {pod.name for pod in self._filtered_pods}
        self.selected_pods = pod_names & available_names
        self._update_selection_highlighting()

    def _get_namespaces(self) -> List[str]:
        """Get unique namespaces from current pods."""
        if not self.pods:
            return []
        return sorted(set(pod.namespace for pod in self.pods))

    def _apply_filters(self) -> None:
        """Apply current filters to the pod list."""
        filtered = self.pods

        # Apply text filter
        if self.filter_text.strip():
            search_text = self.filter_text.lower()
            filtered = [
                pod for pod in filtered
                if (search_text in pod.name.lower() or
                    search_text in pod.namespace.lower() or
                    (pod.labels and any(search_text in f"{k}={v}".lower()
                                        for k, v in pod.labels.items())))
            ]

        # Apply namespace filter
        if self.namespace_filter:
            filtered = [pod for pod in filtered if pod.namespace == self.namespace_filter]

        # Apply phase filter
        if self.phase_filter:
            filtered = [pod for pod in filtered if pod.phase == self.phase_filter]

        self._filtered_pods = filtered
        self._update_table()

    def _update_table(self) -> None:
        """Update the data table with filtered pods."""
        table = self.query_one("#pod-table", DataTable)

        # Clear existing rows
        table.clear(columns=False)

        # Add filtered pods
        for pod in self._filtered_pods:
            status_icon = self._get_status_icon(pod)
            ready_containers = sum(1 for container in pod.containers
                                   if container.ready) if pod.containers else 0
            total_containers = len(pod.containers) if pod.containers else 0
            ready_text = f"{ready_containers}/{total_containers}"

            restart_count = sum(container.restart_count for container in pod.containers
                                if pod.containers) if pod.containers else 0

            age = self._format_age(pod.created_at) if pod.created_at else "Unknown"

            table.add_row(
                status_icon,
                pod.name,
                pod.namespace,
                pod.phase.value if pod.phase else "Unknown",
                ready_text,
                str(restart_count),
                age,
                key=pod.name
            )

        self._update_selection_highlighting()

    def _update_selection_highlighting(self) -> None:
        """Update visual highlighting for selected rows."""
        table = self.query_one("#pod-table", DataTable)

        # Note: Textual's DataTable selection highlighting is handled automatically
        # This method can be extended for custom styling if needed
        pass

    def _get_status_icon(self, pod: PodInfo) -> Text:
        """Get a colored status icon for the pod."""
        if pod.phase == PodPhase.RUNNING:
            if pod.containers and all(c.ready for c in pod.containers):
                return Text("✅", style="green bold")
            else:
                return Text("⚠️", style="yellow bold")
        elif pod.phase == PodPhase.PENDING:
            return Text("🔄", style="blue bold")
        elif pod.phase == PodPhase.SUCCEEDED:
            return Text("✅", style="green bold")
        elif pod.phase == PodPhase.FAILED:
            return Text("❌", style="red bold")
        else:
            return Text("❓", style="dim")

    def _format_age(self, timestamp: datetime) -> str:
        """Format pod age in a human-readable way."""
        now = datetime.now(timestamp.tzinfo)
        delta = now - timestamp

        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if days > 0:
            return f"{days}d{hours}h"
        elif hours > 0:
            return f"{hours}h{minutes}m"
        else:
            return f"{minutes}m"

    def set_selection_callback(self, callback: Callable[[Set[str], List[PodInfo]], None]) -> None:
        """Set callback function for selection changes."""
        self._on_selection_change = callback
