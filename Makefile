.PHONY: help clean lint format test test-cov build install dev-install check all
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project configuration
PROJECT_NAME := gke-log-processor
PYTHON := python
UV := uv

help: ## Show this help message
	@echo "$(BLUE)$(PROJECT_NAME) - Makefile Commands$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

clean: ## Clean up build artifacts and cache files
	@echo "$(YELLOW)Cleaning up build artifacts...$(RESET)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type f -name "coverage.xml" -delete
	@echo "$(GREEN)✓ Cleanup completed$(RESET)"

format: ## Format code with autopep8 and isort
	@echo "$(YELLOW)Formatting code with autopep8...$(RESET)"
	@$(UV) run autopep8 --in-place --aggressive --recursive gke_log_processor/
	@$(UV) run autopep8 --in-place --aggressive --recursive tests/
	@echo "$(YELLOW)Organizing imports with isort...$(RESET)"
	@$(UV) run isort gke_log_processor/ tests/
	@echo "$(GREEN)✓ Code formatting completed$(RESET)"

lint: ## Run all linting tools (flake8, pylint, mypy)
	@echo "$(YELLOW)Running flake8...$(RESET)"
	@$(UV) run flake8 gke_log_processor/ tests/ || (echo "$(RED)✗ Flake8 failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Flake8 passed$(RESET)"
	
	@echo "$(YELLOW)Running pylint...$(RESET)"
	@$(UV) run pylint gke_log_processor/ || (echo "$(RED)✗ Pylint failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Pylint passed$(RESET)"
	
	@echo "$(YELLOW)Running mypy...$(RESET)"
	@$(UV) run mypy gke_log_processor/ || (echo "$(RED)✗ Mypy failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Mypy passed$(RESET)"
	
	@echo "$(GREEN)✓ All linting checks passed$(RESET)"

lint-fix: format ## Fix linting issues automatically where possible
	@echo "$(GREEN)✓ Automatic linting fixes applied$(RESET)"

test: ## Run tests with pytest
	@echo "$(YELLOW)Running tests...$(RESET)"
	@$(UV) run pytest --ignore=tests/ui/ tests/ -v || (echo "$(RED)✗ Tests failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ All tests passed$(RESET)"

test-fast: ## Run tests quickly (no coverage)
	@echo "$(YELLOW)Running fast tests...$(RESET)"
	@$(UV) run pytest tests/ -x -q || (echo "$(RED)✗ Tests failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Fast tests completed$(RESET)"

test-cov: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(RESET)"
	@$(UV) run pytest tests/ --cov=gke_log_processor --cov-report=html --cov-report=xml --cov-report=term || (echo "$(RED)✗ Tests failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Tests with coverage completed$(RESET)"
	@echo "$(BLUE)Coverage report: htmlcov/index.html$(RESET)"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(YELLOW)Starting test watch mode...$(RESET)"
	@$(UV) add --group dev pytest-watch
	@$(UV) run ptw tests/ gke_log_processor/

build: clean ## Build the Python package
	@echo "$(YELLOW)Building Python package...$(RESET)"
	@$(UV) build || (echo "$(RED)✗ Build failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Package built successfully$(RESET)"
	@echo "$(BLUE)Built files:$(RESET)"
	@ls -la dist/

install: build ## Install the package locally
	@echo "$(YELLOW)Installing package locally...$(RESET)"
	@$(UV) sync || (echo "$(RED)✗ Installation failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Package installed successfully$(RESET)"

dev-install: ## Install package in development mode with all dependencies
	@echo "$(YELLOW)Installing in development mode...$(RESET)"
	@$(UV) sync --group dev || (echo "$(RED)✗ Development installation failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

deps: ## Install/update all dependencies
	@echo "$(YELLOW)Installing dependencies...$(RESET)"
	@$(UV) sync --group dev || (echo "$(RED)✗ Dependency installation failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

deps-update: ## Update all dependencies to latest versions
	@echo "$(YELLOW)Updating dependencies...$(RESET)"
	@$(UV) lock --upgrade || (echo "$(RED)✗ Dependency update failed$(RESET)" && exit 1)
	@$(UV) sync --group dev || (echo "$(RED)✗ Dependency sync failed$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)✓ All checks passed$(RESET)"

ci: deps check build ## Run full CI pipeline (deps + check + build)
	@echo "$(GREEN)✓ CI pipeline completed successfully$(RESET)"

security: ## Run security checks with safety
	@echo "$(YELLOW)Running security checks...$(RESET)"
	@$(UV) add --group dev safety
	@$(UV) run safety check || (echo "$(RED)✗ Security issues found$(RESET)" && exit 1)
	@echo "$(GREEN)✓ No security issues found$(RESET)"

docs: ## Generate documentation (placeholder)
	@echo "$(YELLOW)Documentation generation not yet implemented$(RESET)"
	@echo "$(BLUE)Available docs:$(RESET)"
	@find docs/ -name "*.md" -exec echo "  - {}" \;
	@echo "  - README.md"

run: ## Run the CLI application with example args
	@echo "$(YELLOW)Running CLI application (--help)...$(RESET)"
	@$(UV) run gke-logs --help

demo: ## Run the GKE client demo
	@echo "$(YELLOW)Running GKE client demo...$(RESET)"
	@$(UV) run python examples/gke_client_demo.py

tree: ## Show project structure
	@echo "$(BLUE)Project Structure:$(RESET)"
	@tree -I '__pycache__|*.pyc|.git|.venv|.pytest_cache|*.egg-info|htmlcov' || ls -la

all: clean deps check build ## Run everything (clean + deps + check + build)
	@echo "$(GREEN)✓ All tasks completed successfully$(RESET)"

# Development shortcuts
dev: dev-install ## Alias for dev-install
fix: lint-fix ## Alias for lint-fix  
coverage: test-cov ## Alias for test-cov

# Docker commands (if needed later)
docker-build: ## Build Docker image (placeholder)
	@echo "$(YELLOW)Docker support not yet implemented$(RESET)"

docker-run: ## Run Docker container (placeholder)
	@echo "$(YELLOW)Docker support not yet implemented$(RESET)"

# Release commands
version: ## Show current version
	@echo "$(BLUE)Current version:$(RESET)"
	@$(UV) run python -c "import gke_log_processor; print(gke_log_processor.__version__)"

version-bump-patch: ## Bump patch version (0.1.0 -> 0.1.1)
	@echo "$(YELLOW)Version bumping not yet implemented$(RESET)"

version-bump-minor: ## Bump minor version (0.1.0 -> 0.2.0)
	@echo "$(YELLOW)Version bumping not yet implemented$(RESET)"

version-bump-major: ## Bump major version (0.1.0 -> 1.0.0)
	@echo "$(YELLOW)Version bumping not yet implemented$(RESET)"

publish: build ## Publish to PyPI (placeholder)
	@echo "$(YELLOW)Publishing not yet implemented$(RESET)"
	@echo "$(BLUE)To publish manually:$(RESET)"
	@echo "  1. uv build"
	@echo "  2. uv publish --token <your-pypi-token>"

# Information commands
info: ## Show project info
	@echo "$(BLUE)Project Information:$(RESET)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Python: $$($(UV) run python --version)"
	@echo "  UV: $$($(UV) --version)"
	@echo "  Directory: $$(pwd)"
	@echo ""
	@echo "$(BLUE)Dependencies:$(RESET)"
	@$(UV) tree | head -20

env: ## Show environment information
	@echo "$(BLUE)Environment Information:$(RESET)"
	@echo "  Virtual Environment: $$(which python)"
	@echo "  UV Location: $$(which uv)"
	@echo "  Project Root: $$(pwd)"
	@echo "  Python Path: $$($(UV) run python -c 'import sys; print(sys.path[0])')"