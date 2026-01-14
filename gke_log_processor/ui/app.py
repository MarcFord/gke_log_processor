"""Placeholder for main Textual application."""

from textual.app import App

from ..core.config import Config


class GKELogProcessorApp(App):
    """Main Textual application for GKE Log Processor."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def compose(self):
        """Compose the UI."""
        # TODO: Implement the actual UI components
        pass
