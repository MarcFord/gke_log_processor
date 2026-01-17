# GKE Log Processor

A powerful CLI application for monitoring and analyzing Google Kubernetes Engine (GKE) pod logs with AI-powered insights using Gemini AI.

## Features

🚀 **Real-time Log Monitoring**
- Watch logs from running pods in GKE workloads
- Support for multiple namespaces and pod selection
- Real-time streaming with beautiful terminal UI

🤖 **AI-Powered Analysis**
- Gemini AI integration for intelligent log analysis
- Automatic highlighting of severe log messages
- Log summarization for specific time windows
- Pattern detection and anomaly identification

📊 **Rich Terminal Interface**
- Built with Textual for an interactive TUI experience
- Multiple panes for different views (pods, logs, AI insights)
- Keyboard shortcuts for efficient navigation
- Customizable themes and layouts

🔧 **Enterprise Ready**
- Support for both zonal and regional GKE clusters
- Authentication via Google Cloud credentials
- Configuration file support for team workflows
- Comprehensive error handling and logging

## Installation

### Prerequisites

- Python 3.14+
- Google Cloud SDK (gcloud) configured
- kubectl configured for your GKE cluster
- Gemini AI API key

### Install from PyPI (Coming Soon)

```bash
pip install gke-log-processor
```

### Install from Source

```bash
git clone https://github.com/MarcFord/gke_log_processor.git
cd gke_log_processor
uv sync
uv run gke-logs --help
```

### Development Installation

```bash
git clone https://github.com/MarcFord/gke_log_processor.git
cd gke_log_processor
uv sync --group dev
uv run pip install -e .
```

## Quick Start

### Basic Usage

Monitor logs from a GKE cluster:

```bash
gke-logs --cluster my-cluster --project my-project --zone us-central1-a
```

### With AI Analysis

Enable Gemini AI for log analysis:

```bash
export GEMINI_API_KEY="your-api-key"
gke-logs --cluster my-cluster --project my-project --region us-central1
```

### Advanced Usage

```bash
# Monitor specific namespace
gke-logs -c my-cluster -p my-project -z us-central1-a -n production

# Use configuration file
gke-logs --config-file ~/.gke-logs.yaml

# Verbose logging for debugging
gke-logs -c my-cluster -p my-project -z us-central1-a --verbose
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini AI API key | No* |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account key | No** |

\* Required for AI features  
** Required if not using gcloud auth

### Configuration File

Create a configuration file at `~/.gke-logs.yaml`:

```yaml
clusters:
  - name: production
    project_id: my-prod-project
    region: us-central1
    namespace: default
    
  - name: staging
    project_id: my-staging-project
    zone: us-west1-b
    namespace: staging

gemini:
  api_key: ${GEMINI_API_KEY}
  
ui:
  theme: dark
  refresh_rate: 1000
```

## Features in Detail

### Log Monitoring

- **Multi-pod Selection**: Choose specific pods or monitor entire deployments
- **Namespace Filtering**: Focus on specific Kubernetes namespaces
- **Real-time Streaming**: Live log updates with minimal latency
- **Search and Filter**: Find specific log entries quickly
- **Export Capabilities**: Save logs for further analysis

### AI Analysis with Gemini

- **Severity Detection**: Automatically identify ERROR, WARN, and FATAL messages
- **Pattern Recognition**: Detect recurring issues and anomalies
- **Smart Summaries**: Get concise summaries of log activity over time windows
- **Trend Analysis**: Identify patterns in application behavior
- **Custom Queries**: Ask specific questions about your logs

### User Interface

- **Split Panes**: Multiple views showing pods, logs, and AI insights
- **Keyboard Navigation**: Efficient shortcuts for power users
- **Theme Support**: Dark and light themes
- **Responsive Design**: Works on various terminal sizes
- **Status Indicators**: Clear visual feedback on connection and processing status

## CLI Reference

The `gke-logs` CLI provides several commands for interactive and batch log processing.

### Global Options

All commands support these base connection options:

| Option | Shorthand | Description | Required |
| --- | --- | --- | --- |
| `--cluster` | `-c` | GKE cluster name | Yes |
| `--project` | `-p` | GCP project ID | Yes |
| `--namespace` | `-n` | Kubernetes namespace | No (default: `default`) |
| `--zone` | `-z` | GKE cluster zone (for zonal clusters) | No |
| `--region` | `-r` | GKE cluster region (for regional clusters) | No |
| `--config-file` | | Path to configuration file | No |
| `--gemini-api-key` | | Gemini AI API key | No |
| `--verbose` | `-v` | Enable verbose logging | No |

### Commands

#### 1. `ui` (Interactive TUI)

Launch the interactive terminal user interface for exploring pods and logs.

```bash
gke-logs ui [OPTIONS]
```

#### 2. `ai-summary`

Generate a comprehensive AI-powered summary for a specific pod's recent logs.

```bash
gke-logs ai-summary --pod-name <pod-name> [OPTIONS]
```

#### 3. `logs`

Stream or view logs from a pod with optional AI analysis.

```bash
gke-logs logs --pod-name <pod-name> --ai [OPTIONS]
```

- Use `--pod-regex` to match multiple pods.
- Use `--ai` to enable consolidated analysis and summary in one pass.
- Use `--filter` to apply a local regex filter to log messages.

#### 4. `serve`

Start the GKE Log Processor backend API server.

```bash
gke-logs serve --port 8080
```

---

## Backend API

The `serve` command launches a FastAPI-based backend that exposes log analysis and streaming capabilities over HTTP and WebSockets.

### REST Endpoints

#### `GET /health`

Returns the API status and version.

#### `POST /analysis/summary`

Trigger a consolidated AI analysis and summary for a specific pod.

- **Request Body**:

  ```json
  {
    "namespace": "default",
    "pod_name": "my-pod-123",
    "container": "main",
    "tail_lines": 500
  }
  ```

- **Response**: Returns a structured `AIAnalysisResult` containing patterns, recommendations, and an executive summary.

### WebSocket Endpoints

#### `WS /ws/logs/{namespace}/{pod_name}`

Stream log entries in real-time over a WebSocket connection.

- **Query Parameters**:
  - `container`: Specific container name
  - `tail_lines`: Initial lines to fetch (default: 50)

## Development

### Setup Development Environment

```bash
git clone https://github.com/MarcFord/gke_log_processor.git
cd gke_log_processor
make dev-install  # Install in development mode with all dependencies
```

### Available Make Commands

The project includes a comprehensive Makefile with 30+ commands for development workflow automation:

#### **Quick Reference**
```bash
make help         # Show all available commands with descriptions
make dev          # Alias for dev-install
make check        # Run all checks (lint + test)
make ci           # Full CI pipeline (deps + check + build)
```

#### **Development Setup**
| Command | Description |
|---------|-------------|
| `make dev-install` | Install package in development mode with all dependencies |
| `make deps` | Install/update all dependencies |
| `make deps-update` | Update all dependencies to latest versions |
| `make env` | Show environment information |

#### **Code Quality & Testing**
| Command | Description |
|---------|-------------|
| `make lint` | Run all linting tools (flake8, pylint, mypy) |
| `make lint-fix` | Fix linting issues automatically where possible |
| `make format` | Format code with autopep8 and isort |
| `make test` | Run full test suite with coverage |
| `make test-fast` | Run tests quickly (no coverage) |
| `make test-cov` | Run tests with detailed coverage report |
| `make test-watch` | Run tests in watch mode (requires pytest-watch) |
| `make security` | Run security checks with safety |

#### **Build & Package**
| Command | Description |
|---------|-------------|
| `make build` | Build Python package (wheel + source distribution) |
| `make clean` | Clean up build artifacts and cache files |
| `make install` | Install the package locally |
| `make publish` | Publish to PyPI (placeholder) |

#### **Development Utilities**
| Command | Description |
|---------|-------------|
| `make run` | Run the CLI application with example args |
| `make demo` | Run the GKE client demo |
| `make tree` | Show project structure |
| `make info` | Show project information |
| `make version` | Show current version |

#### **Version Management**
| Command | Description |
|---------|-------------|
| `make version-bump-patch` | Bump patch version (0.1.0 → 0.1.1) |
| `make version-bump-minor` | Bump minor version (0.1.0 → 0.2.0) |
| `make version-bump-major` | Bump major version (0.1.0 → 1.0.0) |

#### **Docker (Planned)**
| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image (placeholder) |
| `make docker-run` | Run Docker container (placeholder) |

### Development Workflow

#### **Daily Development**
```bash
# Start development
make dev-install

# Code, then check quality
make check          # Runs lint + test

# Fix any issues
make lint-fix       # Auto-fix what's possible
make format         # Format code

# Build and test package
make build
```

#### **Continuous Integration**
```bash
make ci             # Full pipeline: deps + check + build
```

#### **Quick Testing**
```bash
make test-fast      # Quick tests without coverage
make test-watch     # Continuous testing while developing
```

### Code Quality Standards

The project maintains high code quality with:

- **Flake8**: Code style and syntax checking
- **Pylint**: Advanced code analysis (target: 10.0/10)
- **Mypy**: Static type checking
- **Autopep8**: Automatic code formatting
- **Isort**: Import statement organization
- **Pytest**: Comprehensive test coverage

All tools are pre-configured and run automatically with `make lint`.

## Architecture

```
gke_log_processor/
├── core/           # Configuration and core utilities
├── gke/            # GKE and Kubernetes integration  
├── ai/             # Gemini AI integration
├── ui/             # Textual-based user interface
└── cli.py          # Command-line interface
```

### Key Components

- **Core**: Configuration management, error handling, utilities
- **GKE Module**: Google Cloud and Kubernetes API integration
- **AI Module**: Gemini AI integration for log analysis
- **UI Module**: Textual-based terminal user interface

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 **Documentation**: [GitHub Wiki](https://github.com/MarcFord/gke_log_processor/wiki)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/MarcFord/gke_log_processor/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/MarcFord/gke_log_processor/discussions)

## Roadmap

- [ ] Support for multiple clusters simultaneously
- [ ] Log aggregation and historical analysis
- [ ] Custom AI prompts and analysis templates
- [ ] Integration with monitoring tools (Prometheus, Grafana)
- [ ] Export to various formats (JSON, CSV, PDF reports)
- [ ] Plugin system for custom log processors
- [ ] Web-based dashboard (optional)

---

**Note**: This project is in active development. APIs and features may change between versions.
