"""Main CLI entry point for GKE Log Processor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ai.analyzer import LogAnalysisEngine
from .ai.client import GeminiConfig
from .ai.summarizer import LogSummaryReport, SummarizerConfig
from .core.config import Config
from .core.exceptions import GKELogProcessorError
from .core.models import AIAnalysisResult, LogEntry, LogLevel, SeverityLevel
from .core.service import LogProcessingService
from .gke.client import GKEClient
from .ui.app import GKELogProcessorApp

console = Console()


@dataclass
class SummaryArtifacts:
    """Container for AI summary results."""

    analysis: Optional[AIAnalysisResult]
    summary: Optional[LogSummaryReport]
    log_entries: List[LogEntry]
    container_name: str
    pod_namespace: str


def _load_config_from_options(
    *,
    config_file: Optional[str],
    cluster: Optional[str],
    project: Optional[str],
    zone: Optional[str],
    region: Optional[str],
    namespace: Optional[str],
    gemini_api_key: Optional[str],
    verbose: bool,
) -> Config:
    """Create a Config instance applying CLI overrides."""

    if config_file:
        config = Config.load_from_file(config_file)
    else:
        config = Config()

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
    if namespace:
        config.kubernetes.default_namespace = namespace
    if gemini_api_key:
        config.ai.gemini_api_key = gemini_api_key
    config.verbose = verbose

    return config


def _validate_cluster_config(config: Config) -> None:
    """Ensure configuration contains enough data to connect to a cluster."""

    if not config.gke.cluster_name:
        raise click.ClickException(
            "Cluster name is required (use --cluster or config file)"
        )
    if not config.gke.project_id:
        raise click.ClickException(
            "Project ID is required (use --project or config file)"
        )
    if not config.current_cluster:
        raise click.ClickException("Invalid cluster configuration")


def _build_analysis_engine(config: Config) -> LogAnalysisEngine:
    """Construct a log analysis engine respecting configuration."""

    gemini_config: Optional[GeminiConfig] = None
    if config.ai.analysis_enabled:
        api_key = config.effective_gemini_api_key
        if api_key:
            max_tokens = max(1, min(config.ai.max_tokens, 32768))
            gemini_config = GeminiConfig(
                api_key=api_key,
                model=config.ai.model_name,
                temperature=config.ai.temperature,
                max_output_tokens=max_tokens,
            )

    summary_length = max(100, min(config.ai.max_tokens, 2000))
    summarizer_config = SummarizerConfig(
        max_summary_length=summary_length,
        enable_ai_summarization=bool(gemini_config),
    )

    return LogAnalysisEngine(
        gemini_config=gemini_config,
        summarizer_config=summarizer_config,
    )


def _parse_log_line(line: str) -> tuple[datetime, str]:
    """Parse a Kubernetes log line into timestamp and message."""

    default_timestamp = datetime.now(timezone.utc)
    if not line:
        return default_timestamp, ""

    ts_candidate, _, remainder = line.partition(" ")
    parsed_ts = _parse_timestamp(ts_candidate)
    if parsed_ts and remainder:
        return parsed_ts, remainder

    parts = line.split(" ", 3)
    if len(parts) >= 4 and parts[1] in {"stdout", "stderr"}:
        parsed_ts = _parse_timestamp(parts[0])
        if parsed_ts:
            return parsed_ts, parts[3]

    return default_timestamp, line


def _parse_timestamp(value: str) -> Optional[datetime]:
    """Best-effort RFC3339 timestamp parsing."""

    if not value:
        return None

    normalised = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalised)
    except ValueError:
        return None


def _detect_log_level(message: str) -> Optional[LogLevel]:
    """Derive a log level from the message prefix if available."""

    if not message:
        return None

    token = message.split(" ", 1)[0].strip("[]:").upper()
    level_map: Dict[str, LogLevel] = {
        "TRACE": LogLevel.TRACE,
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARN": LogLevel.WARNING,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "ERR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
        "FATAL": LogLevel.CRITICAL,
    }

    return level_map.get(token)


def _build_log_entry(
    raw_line: str,
    *,
    pod_name: str,
    namespace: str,
    cluster: str,
    container_name: str,
) -> LogEntry:
    """Convert a raw log line to a structured LogEntry."""

    timestamp, message = _parse_log_line(raw_line)
    level = _detect_log_level(message)

    return LogEntry(
        timestamp=timestamp,
        message=message,
        level=level.value if level else None,
        source=container_name,
        pod_name=pod_name,
        namespace=namespace,
        cluster=cluster,
        container_name=container,
        raw_message=raw_line,
    )


def _confidence_bar(confidence: float, width: int = 5) -> str:
    """Render a simple ASCII confidence bar."""

    filled = max(0, min(width, int(round(confidence * width))))
    return "#" * filled + "-" * (width - filled)


def _render_summary(
    pod_name: str,
    namespace: str,
    artifacts: SummaryArtifacts,
) -> None:
    """Render analysis and summary data to the console."""

    console.print(
        f"[bold cyan]AI Summary for {namespace}/{pod_name}[/bold cyan]"
        f" (container: [magenta]{artifacts.container_name}[/magenta])"
    )
    console.print(f"Log entries analyzed: {len(artifacts.log_entries):,}")

    severity_icons = {
        SeverityLevel.LOW: "✅",
        SeverityLevel.MEDIUM: "⚠️",
        SeverityLevel.HIGH: "🔥",
        SeverityLevel.CRITICAL: "🚨",
    }

    if artifacts.analysis:
        severity = artifacts.analysis.overall_severity
        icon = severity_icons.get(severity, "❔")
        console.print(
            f"Overall severity: {icon} {severity.value.title()}"
            f" (confidence {artifacts.analysis.confidence_score:.0%})"
        )
        if artifacts.analysis.top_error_messages:
            console.print("\n[bold]Top Issues:[/bold]")
            for idx, error in enumerate(artifacts.analysis.top_error_messages[:3], 1):
                console.print(f"{idx}. {error}")
        if artifacts.analysis.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in artifacts.analysis.recommendations[:5]:
                console.print(f"- {rec}")

    if artifacts.summary:
        summary = artifacts.summary
        start = summary.time_range_start.strftime("%Y-%m-%d %H:%M:%S")
        end = summary.time_range_end.strftime("%Y-%m-%d %H:%M:%S")
        console.print(
            f"\nTime range: {start} - {end}"
            f" · windows: {len(summary.window_summaries)}"
            f" · total logs: {summary.total_log_entries:,}"
        )

        if summary.executive_summary:
            console.print()
            console.print(
                Panel.fit(
                    summary.executive_summary,
                    title="Executive Summary",
                    border_style="cyan",
                )
            )

        if summary.key_insights:
            console.print("\n[bold]Key Insights:[/bold]")
            for insight in summary.key_insights[:5]:
                console.print(
                    f"- [bold]{insight.title}[/bold]"
                    f" ({insight.severity.value.title()}, {insight.confidence:.0%})"
                )
                console.print(f"  {insight.description}")
                console.print(f"  Confidence: {_confidence_bar(insight.confidence)}")

        if summary.trend_analyses:
            console.print("\n[bold]Trend Analysis:[/bold]")
            for trend in summary.trend_analyses[:5]:
                direction = trend.direction.value.title()
                change = (
                    f" · change {trend.change_percentage:+.1f}%"
                    if trend.change_percentage is not None
                    else ""
                )
                console.print(f"- {trend.metric_name}: {direction}{change}")
                if trend.recommendation:
                    console.print(f"  Action: {trend.recommendation}")

        if summary.recommendations:
            console.print("\n[bold]Summary Recommendations:[/bold]")
            for rec in summary.recommendations[:5]:
                console.print(f"- {rec}")

    console.print()


async def _collect_ai_summary(
    config: Config,
    *,
    pod_name: str,
    namespace: str,
    container_override: Optional[str],
    tail_lines: int,
) -> SummaryArtifacts:
    """Fetch pod logs and generate AI analysis and summary."""

    service = LogProcessingService(config)
    
    # We need to get the container name if it's not provided, similar to original logic
    # Service doesn't return container name directly with get_pod_logs currently
    # But get_pod_logs handles the fetching.
    
    # To keep this clean, let's use the service but we might need to query pod details 
    # to get the container name for the artifact if not provided.
    
    # For now, let's stick to using the service for fetching and analysis
    
    log_entries = await service.get_pod_logs(
        namespace=namespace,
        pod_name=pod_name,
        container=container_override,
        tail_lines=tail_lines
    )

    if not log_entries:
        return SummaryArtifacts(None, None, [], container_override or "unknown", namespace)

    # Determine container name from first log entry if available
    container_name = log_entries[0].container_name if log_entries else (container_override or "unknown")

    analysis = await service.analyze_logs(
        log_entries,
        analysis_type="summary",
    )
    summary = await service.summarize_logs(log_entries)

    return SummaryArtifacts(analysis, summary, log_entries, container_name, namespace)


def _list_pods(config: Config) -> None:
    """List pods using the provided configuration."""

    namespace = config.kubernetes.default_namespace or "default"
    client = GKEClient(config)

    try:
        console.print(
            f"[cyan]Listing pods in namespace '{namespace}' for cluster '{config.gke.cluster_name}'...[/cyan]"
        )

        k8s_client = client.get_kubernetes_client()
        pods = asyncio.run(
            k8s_client.list_pods(namespace=namespace, force_refresh=True)
        )

        if not pods:
            console.print(f"[yellow]No pods found in namespace '{namespace}'.[/yellow]")
            return

        table = Table(title=f"Pods in namespace '{namespace}'")
        table.add_column("Pod", style="bold")
        table.add_column("Status")
        table.add_column("Ready")
        table.add_column("Restarts", justify="right")
        table.add_column("Age", justify="right")

        for pod in pods:
            ready = "✅" if pod.is_ready else "➖"
            table.add_row(
                pod.name,
                pod.status_summary,
                ready,
                str(pod.restart_count),
                pod.age,
            )

        console.print(table)

    finally:
        client.close()


def _ui_options(func):
    """Decorator applying shared UI CLI options."""

    options = [
        click.option("--cluster", "-c", help="GKE cluster name"),
        click.option("--project", "-p", help="GCP project ID"),
        click.option("--zone", "-z", help="GKE cluster zone"),
        click.option("--region", "-r", help="GKE cluster region"),
        click.option("--namespace", "-n", default="default", help="Kubernetes namespace"),
        click.option("--config-file", help="Path to configuration file"),
        click.option("--gemini-api-key", help="Gemini AI API key"),
        click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging"),
        click.option("--list-pods", is_flag=True, help="List pods in the target namespace and exit"),
    ]

    for option in reversed(options):
        func = option(func)
    return func


@click.group(invoke_without_command=True)
@click.pass_context
@_ui_options
def cli(ctx, **kwargs):
    """GKE Log Processor CLI."""

    if ctx.invoked_subcommand is None:
        ctx.invoke(run_ui, **kwargs)
    else:
        ctx.obj = {"base_options": kwargs}


@cli.command("run-ui")
@_ui_options
def run_ui(
    cluster: Optional[str],
    project: Optional[str],
    zone: Optional[str],
    region: Optional[str],
    namespace: str,
    config_file: Optional[str],
    gemini_api_key: Optional[str],
    verbose: bool,
    list_pods: bool,
) -> None:
    """Launch the Textual UI for interactive log exploration."""

    try:
        config = _load_config_from_options(
            config_file=config_file,
            cluster=cluster,
            project=project,
            zone=zone,
            region=region,
            namespace=namespace,
            gemini_api_key=gemini_api_key,
            verbose=verbose,
        )
        _validate_cluster_config(config)

        if list_pods:
            _list_pods(config)
            return

        app = GKELogProcessorApp(config)
        app.run()

    except GKELogProcessorError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise click.Abort()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except click.ClickException:
        raise
    except Exception as error:  # pragma: no cover - defensive logging
        console.print(f"[red]Unexpected error: {error}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()


def _summary_options(func):
    """Decorator applying shared connection options for AI summary command."""

    options = [
        click.option("--cluster", "-c", help="GKE cluster name"),
        click.option("--project", "-p", help="GCP project ID"),
        click.option("--zone", "-z", help="GKE cluster zone"),
        click.option("--region", "-r", help="GKE cluster region"),
        click.option("--namespace", "-n", help="Kubernetes namespace"),
        click.option("--config-file", help="Path to configuration file"),
        click.option("--gemini-api-key", help="Gemini AI API key"),
        click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging"),
    ]

    for option in reversed(options):
        func = option(func)
    return func


@cli.command("ai-summary")
@click.pass_context
@_summary_options
@click.option("--pod-name", "-P", help="Name of the pod to analyze")
@click.option("--pod-regex", "-R", help="Regex pattern to match pod names")
@click.option("--container", "-C", help="Specific container within the pod")
@click.option(
    "--tail-lines",
    "-t",
    type=int,
    help="Number of recent log lines to analyze (defaults to streaming.tail_lines)",
)
def ai_summary(
    ctx: click.Context,
    pod_name: Optional[str],
    pod_regex: Optional[str],
    container: Optional[str],
    tail_lines: Optional[int],
    cluster: Optional[str],
    project: Optional[str],
    zone: Optional[str],
    region: Optional[str],
    namespace: Optional[str],
    config_file: Optional[str],
    gemini_api_key: Optional[str],
    verbose: bool,
) -> None:
    """Generate an AI-powered summary for a pod's recent logs."""

    base_options = (ctx.obj or {}).get("base_options", {})

    cluster = cluster or base_options.get("cluster")
    project = project or base_options.get("project")
    zone = zone or base_options.get("zone")
    region = region or base_options.get("region")
    namespace = namespace or base_options.get("namespace") or "default"
    config_file = config_file or base_options.get("config_file")
    gemini_api_key = gemini_api_key or base_options.get("gemini_api_key")
    verbose = verbose or base_options.get("verbose", False)

    if not pod_name and not pod_regex:
        raise click.ClickException("Either pod name (use --pod-name) or pod regex (use --pod-regex) is required")

    try:
        config = _load_config_from_options(
            config_file=config_file,
            cluster=cluster,
            project=project,
            zone=zone,
            region=region,
            namespace=namespace,
            gemini_api_key=gemini_api_key,
            verbose=verbose,
        )
        _validate_cluster_config(config)

        if not config.ai.analysis_enabled:
            console.print("[yellow]AI analysis disabled via configuration.[/yellow]")
        elif not config.effective_gemini_api_key:
            console.print(
                "[yellow]No Gemini API key configured; falling back to heuristic summaries.[/yellow]"
            )

        effective_tail = tail_lines or config.streaming.tail_lines or 200
        if effective_tail <= 0:
            raise click.ClickException("tail-lines must be greater than zero")

        if pod_regex:
            import re
            try:
                re.compile(pod_regex)
            except re.error as e:
                raise click.ClickException(f"Invalid regex pattern: {e}")

            console.print(f"[cyan]Fetching logs for pods matching '{pod_regex}'...[/cyan]")
            service = LogProcessingService(config)
            log_entries = asyncio.run(
                service.get_matching_pods_logs(
                    namespace=namespace,
                    pod_regex=pod_regex,
                    container=container,
                    tail_lines=effective_tail
                )
            )
            if not log_entries:
                console.print(f"[yellow]No logs found for pods matching '{pod_regex}' in namespace '{namespace}'.[/yellow]")
                return

            pod_name_display = f"regex('{pod_regex}')"
            container_name = container or "mixed"
        else:
            # pod_name is guaranteed to be set if pod_regex is not, due to prior check
            if not pod_name:
                 raise click.ClickException("Pod name is required if regex is not provided")

            artifacts = asyncio.run(
                _collect_ai_summary(
                    config,
                    pod_name=pod_name,
                    namespace=namespace,
                    container_override=container,
                    tail_lines=effective_tail,
                )
            )
            if not artifacts.log_entries:
                console.print(
                    f"[yellow]No logs retrieved for pod '{namespace}/{pod_name}'.[/yellow]"
                )
                return
            
            # Use simple render for single pod path
            _render_summary(pod_name, artifacts.pod_namespace, artifacts)
            return

        # Process aggregated logs
        # This part handles the 'if pod_regex' branch continuation
        service = LogProcessingService(config)
        analysis = asyncio.run(service.analyze_logs(log_entries, analysis_type="summary"))
        summary = asyncio.run(service.summarize_logs(log_entries))
        
        artifacts = SummaryArtifacts(
            analysis=analysis,
            summary=summary,
            log_entries=log_entries,
            container_name=container,
            pod_namespace=namespace
        )
        
        _render_summary(pod_name_display, namespace, artifacts)

    except GKELogProcessorError as error:
        console.print(f"[red]Cluster error: {error}[/red]")
        raise click.Abort()
    except click.ClickException:
        raise
    except Exception as error:  # pragma: no cover - defensive logging
        console.print(f"[red]Unexpected error: {error}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()



@cli.command("logs")
@click.pass_context
@_summary_options
@click.option("--pod-name", "-P", help="Name of the pod to analyze")
@click.option("--pod-regex", "-R", help="Regex pattern to match pod names")
@click.option("--container", "-C", help="Specific container within the pod")
@click.option(
    "--tail-lines",
    "-t",
    type=int,
    help="Number of recent log lines to analyze (defaults to streaming.tail_lines)",
)
@click.option("--filter", "-f", "filter_pattern", help="Regex pattern to filter log messages")
@click.option("--ai", "ai_enabled", is_flag=True, default=False, help="Enable AI analysis")
def logs(
    ctx: click.Context,
    pod_name: Optional[str],
    pod_regex: Optional[str],
    container: Optional[str],
    tail_lines: Optional[int],
    filter_pattern: Optional[str],
    cluster: Optional[str],
    project: Optional[str],
    zone: Optional[str],
    region: Optional[str],
    namespace: Optional[str],
    config_file: Optional[str],
    gemini_api_key: Optional[str],
    verbose: bool,
    ai_enabled: bool,
) -> None:
    """View logs from a pod or multiple pods with regex support."""
    import re

    base_options = (ctx.obj or {}).get("base_options", {})

    cluster = cluster or base_options.get("cluster")
    project = project or base_options.get("project")
    zone = zone or base_options.get("zone")
    region = region or base_options.get("region")
    namespace = namespace or base_options.get("namespace") or "default"
    config_file = config_file or base_options.get("config_file")
    gemini_api_key = gemini_api_key or base_options.get("gemini_api_key")
    verbose = verbose or base_options.get("verbose", False)

    if not pod_name and not pod_regex:
        raise click.ClickException(
            "Either pod name (use --pod-name) or pod regex (use --pod-regex) is required"
        )

    try:
        config = _load_config_from_options(
            config_file=config_file,
            cluster=cluster,
            project=project,
            zone=zone,
            region=region,
            namespace=namespace,
            gemini_api_key=gemini_api_key,
            verbose=verbose,
        )
        _validate_cluster_config(config)

        effective_tail = tail_lines or config.streaming.tail_lines or 200
        if effective_tail <= 0:
            raise click.ClickException("tail-lines must be greater than zero")

        service = LogProcessingService(config)
        log_entries: List[LogEntry] = []
        source_desc = ""

        if pod_regex:
            try:
                re.compile(pod_regex)
            except re.error as e:
                raise click.ClickException(f"Invalid pod regex pattern: {e}")

            console.print(f"[cyan]Fetching logs for pods matching '{pod_regex}'...[/cyan]")
            log_entries = asyncio.run(
                service.get_matching_pods_logs(
                    namespace=namespace,
                    pod_regex=pod_regex,
                    container=container,
                    tail_lines=effective_tail,
                )
            )
            if not log_entries:
                console.print(
                    f"[yellow]No logs found for pods matching '{pod_regex}' in namespace '{namespace}'.[/yellow]"
                )
                return
            source_desc = f"regex('{pod_regex}')"
        else:
            # pod_name is guaranteed to be set
            if not pod_name:
                raise click.ClickException("Pod name is required")
                
            console.print(f"[cyan]Fetching logs for pod '{pod_name}'...[/cyan]")
            log_entries = asyncio.run(
                service.get_pod_logs(
                    namespace=namespace,
                    pod_name=pod_name,
                    container=container,
                    tail_lines=effective_tail,
                )
            )
            if not log_entries:
                console.print(
                    f"[yellow]No logs retrieved for pod '{namespace}/{pod_name}'.[/yellow]"
                )
                return
            source_desc = f"{namespace}/{pod_name}"

        # Apply message filter if provided
        if filter_pattern:
            try:
                pattern = re.compile(filter_pattern, re.IGNORECASE)
                original_count = len(log_entries)
                log_entries = [e for e in log_entries if pattern.search(e.message)]
                console.print(
                    f"[dim]Filtered {original_count} logs down to {len(log_entries)} matching '{filter_pattern}'[/dim]"
                )
            except re.error as e:
                raise click.ClickException(f"Invalid filter regex pattern: {e}")

        if not log_entries:
            console.print("[yellow]No logs to display after filtering.[/yellow]")
            return

        # Render logs table
        table = Table(
            title=f"Logs for {source_desc}",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            # box=click.exceptions # This was weird in my thought, box=None or box.SIMPLE
        )
        table.add_column("Timestamp", style="dim", width=24)
        if pod_regex:  # specific pod provided? might be multiple if regex
            table.add_column("Pod", style="cyan")
            table.add_column("Container", style="blue")
        table.add_column("Level", width=8)
        table.add_column("Message")

        level_styles = {
            "TRACE": "dim",
            "DEBUG": "cyan",
            "INFO": "green",
            "WARN": "yellow",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
            "FATAL": "bold red reversed",
        }

        for entry in log_entries:
            level_str = (entry.level or "INFO").upper()
            style = level_styles.get(level_str, "white")
            
            row = [
                entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            if pod_regex:
                row.append(entry.pod_name)
                row.append(entry.container_name)
            
            row.append(f"[{style}]{level_str}[/{style}]")
            row.append(entry.message)
            
            table.add_row(*row)

        console.print(table)

        # Process aggregated logs
        # This part handles the 'if pod_regex' branch continuation
        if ai_enabled:
            service = LogProcessingService(config)
            analysis = asyncio.run(service.analyze_logs(log_entries, analysis_type="comprehensive"))
            summary = asyncio.run(service.summarize_logs(log_entries, ai_summary=analysis.summary))
            
            artifacts = SummaryArtifacts(
                analysis=analysis,
                summary=summary,
                log_entries=log_entries,
                container_name=container,
                pod_namespace=namespace
            )
            _render_summary(source_desc, namespace, artifacts)

    except GKELogProcessorError as error:
        console.print(f"[red]Cluster error: {error}[/red]")
        raise click.Abort()
    except click.ClickException:
        raise
    except Exception as error:  # pragma: no cover - defensive logging
        console.print(f"[red]Unexpected error: {error}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()


main = cli


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the GKE Log Processor API server."""
    import uvicorn
    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    uvicorn.run(
        "gke_log_processor.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
