.PHONY: help install install-dev hooks-install hooks test test-cov lint format type-check clean build publish bump-patch bump-minor bump-major doctor

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with uv
	uv pip install -e .

install-dev: ## Install the package with development dependencies using uv
	uv sync --extra dev

hooks-install: ## Install git pre-commit hooks
	uv run --extra dev pre-commit install

hooks: ## Run pre-commit hooks across the repository
	uv run --extra dev pre-commit run --all-files

test: ## Run tests
	uv run --extra dev pytest

doctor: ## Validate GridGulp Gemma configuration
	uv run --extra dev gridgulp doctor

test-cov: ## Run tests with coverage
	uv run --extra dev pytest --cov=gridgulp --cov-report=html --cov-report=term

lint: ## Run linting with ruff
	uv run --extra dev ruff check src/ tests/

format: ## Format code with black
	uv run --extra dev ruff format src/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage

build: clean ## Build distribution packages with uv
	uv build

publish: ## Publish to PyPI using uv
	uv publish

publish-test: ## Publish to TestPyPI using uv
	uv publish --index-url https://test.pypi.org/legacy/

bump-patch: ## Bump patch version
	python scripts/bump_version.py patch

bump-minor: ## Bump minor version
	python scripts/bump_version.py minor

bump-major: ## Bump major version
	python scripts/bump_version.py major

dev: install-dev ## Set up development environment
	uv run --extra dev pre-commit install

check: lint type-check test ## Run all checks (lint, type-check, test)
