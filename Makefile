.PHONY: help install install-dev test test-cov lint format type-check clean build publish bump-patch bump-minor bump-major

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with uv
	uv pip install -e .

install-dev: ## Install the package with development dependencies using uv
	uv pip install -e ".[dev]"

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=gridgulp --cov-report=html --cov-report=term

lint: ## Run linting with ruff
	ruff check src/ tests/

format: ## Format code with black
	black src/ tests/

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
	pre-commit install

check: lint type-check test ## Run all checks (lint, type-check, test)
