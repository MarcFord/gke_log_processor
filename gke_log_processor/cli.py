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
        container_name=container_name,
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

    client = GKEClient(config)
    try:
        k8s_client = client.get_kubernetes_client()
        pod = await k8s_client.get_pod(pod_name, namespace)

        container_name = container_override or (pod.containers[0] if pod.containers else None)
        if not container_name:
            raise click.ClickException(
                f"Pod '{namespace}/{pod_name}' has no containers to analyze"
            )

        raw_logs = await k8s_client.get_pod_logs(
            pod.name,
            namespace=pod.namespace,
            container=container_name,
            tail_lines=tail_lines,
            timestamps=True,
        )

        cluster_name = (
            config.gke.cluster_name
            or (config.current_cluster.name if config.current_cluster else None)
            or "unknown"
        )

        log_entries = [
            _build_log_entry(
                line,
                pod_name=pod.name,
                namespace=pod.namespace,
                cluster=cluster_name,
                container_name=container_name,
            )
            for line in raw_logs
        ]

        if not log_entries:
            return SummaryArtifacts(None, None, [], container_name, pod.namespace)

        engine = _build_analysis_engine(config)
        use_ai = config.ai.analysis_enabled and engine.gemini_client is not None

        analysis = await engine.analyze_logs_comprehensive(
            log_entries,
            use_ai=use_ai,
            analysis_type="summary",
        )
        summary = await engine.summarizer.summarize_logs(log_entries)

        return SummaryArtifacts(analysis, summary, log_entries, container_name, pod.namespace)

    finally:
        client.close()


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

    if not pod_name:
        raise click.ClickException("Pod name is required (use --pod-name)")

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

        _render_summary(pod_name, artifacts.pod_namespace, artifacts)

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


if __name__ == "__main__":  # pragma: no cover
    cli()
