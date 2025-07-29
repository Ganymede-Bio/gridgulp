# Contributing to GridGulp

Thank you for your interest in contributing to GridGulp! This document provides guidelines and instructions for contributing to the project.

## Getting Started

GridGulp uses the **fork and pull request** model for contributions. This means you don't need direct write access to the repository - you'll work on your own fork and submit pull requests for review.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for fast Python package management
- Git

### Setting Up Your Development Environment

1. **Fork the repository**
   - Go to https://github.com/Ganymede-Bio/gridgulp
   - Click the "Fork" button in the top-right corner
   - This creates your own copy at `https://github.com/yourusername/gridgulp`

2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/gridgulp.git
   cd gridgulp
   ```

3. **Add the upstream remote**
   ```bash
   git remote add upstream https://github.com/Ganymede-Bio/gridgulp.git
   git remote -v  # Verify you have both origin and upstream
   ```

4. **Create a virtual environment with uv**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

5. **Install the package in development mode**
   ```bash
   make install-dev
   # Or directly: uv pip install -e ".[dev]"
   ```

6. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Standards

### Style Guide

- We use [Ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are required for all new code
- Follow PEP 8 naming conventions

### Pre-commit Hooks

Pre-commit hooks will automatically run when you commit. They include:
- Ruff linting
- Trailing whitespace removal
- File size checks

You can manually run all hooks:
```bash
pre-commit run --all-files
```

### Type Annotations

All code must be fully typed. We use Pydantic 2 for data models:

```python
from pydantic import BaseModel, Field

class TableInfo(BaseModel):
    range: str = Field(..., description="Excel-style range")
    confidence: float = Field(..., ge=0.0, le=1.0)
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_detectors.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_should_detect_single_table_when_no_gaps`
- Include both positive and negative test cases
- Test edge cases and error conditions
- Aim for >80% code coverage

Example test:
```python
import pytest
from gridgulp.detectors import SimpleCaseDetector

def test_simple_case_detection():
    detector = SimpleCaseDetector()
    sheet_data = create_test_sheet_data()
    result = detector.detect_simple_table(sheet_data)
    assert result.is_simple_table
    assert result.confidence > 0.9
```

## Making Changes

### Keeping Your Fork Updated

Before starting new work, always sync your fork with the upstream repository:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

### Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guide
   - Add/update tests
   - Update documentation if needed

3. **Run checks locally**
   ```bash
   make check  # Runs lint, type-check, and tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add support for merged cell detection"
   ```

   Follow conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for refactoring
   - `chore:` for maintenance

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request**
   - Go to your fork on GitHub
   - Click "Contribute" → "Open pull request"
   - Ensure the base repository is `Ganymede-Bio/gridgulp` and base branch is `main`
   - Fill out the PR template with a clear description of your changes
   - Submit the pull request for review

### Pull Request Guidelines

- Fill out the PR description template
- Ensure all CI checks pass
- Include tests for new functionality
- Update CHANGELOG.md if applicable
- Be patient - maintainers will review your PR as soon as possible
- Respond to review feedback promptly
- Your PR must be approved before it can be merged

## Project Structure

```
gridgulp/
├── src/gridgulp/        # Main package
│   ├── core/             # Core functionality
│   ├── detectors/        # Detection strategies
│   ├── models/           # Pydantic models
│   ├── readers/          # File readers
│   └── utils/            # Utilities
├── tests/                # Test suite
├── examples/             # Example scripts
└── docs/                 # Documentation
```

## Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings:

```python
def detect_tables(file_path: str) -> DetectionResult:
    """Detect tables in a spreadsheet file.

    Args:
        file_path: Path to the spreadsheet file

    Returns:
        DetectionResult containing all detected tables

    Raises:
        FileNotFoundError: If file doesn't exist
        UnsupportedFormatError: If file format is not supported
    """
```

## Questions?

If you have questions or need help:

1. Check existing issues and discussions
2. Open a new issue with your question
3. Join our community discussions

Thank you for contributing!
