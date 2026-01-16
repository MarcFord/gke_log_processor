"""Main CLI entry point for GKE Log Processor."""

import click
from rich.console import Console

from .core.config import Config
from .core.exceptions import GKELogProcessorError
from .ui.app import GKELogProcessorApp

console = Console()


@click.command()
@click.option("--cluster", "-c", help="GKE cluster name")
@click.option("--project", "-p", help="GCP project ID")
@click.option("--zone", "-z", help="GKE cluster zone")
@click.option("--region", "-r", help="GKE cluster region")
@click.option("--namespace", "-n", default="default",
              help="Kubernetes namespace")
@click.option("--config-file", help="Path to configuration file")
@click.option("--gemini-api-key", help="Gemini AI API key")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.version_option()
def main(
    cluster, project, zone, region, namespace, config_file, gemini_api_key, verbose
):
    """GKE Log Processor - Monitor and analyze GKE pod logs with AI."""
    try:
        # Load base configuration
        if config_file:
            config = Config.load_from_file(config_file)
        else:
            config = Config()

        # Override with CLI arguments
        if cluster:
            config.gke.cluster_name = cluster
        if project:
            config.gke.project_id = project
        if zone:
            config.gke.zone = zone
            config.gke.region = None
        if region:
            config.gke.region = region
            config.gke.zone = None
        if namespace != "default":
            config.kubernetes.default_namespace = namespace
        if gemini_api_key:
            config.ai.gemini_api_key = gemini_api_key
        config.verbose = verbose

        # Validate that we have required information
        if not config.gke.cluster_name:
            raise click.ClickException(
                "Cluster name is required (use --cluster or config file)")
        if not config.gke.project_id:
            raise click.ClickException(
                "Project ID is required (use --project or config file)")
        if not config.current_cluster:
            raise click.ClickException("Invalid cluster configuration")

        app = GKELogProcessorApp(config)
        app.run()

    except GKELogProcessorError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()


if __name__ == "__main__":
    main()
