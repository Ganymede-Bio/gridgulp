# Contributing to GridGulp

Thank you for your interest in contributing to GridGulp! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Setting Up Development Environment

1. **Fork and clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/gridgulp.git
cd gridgulp
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**:
```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks**:
```bash
pre-commit install
```

5. **Run tests to verify setup**:
```bash
pytest
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

Follow these guidelines:
- Write clear, concise code
- Add type hints to all functions
- Include docstrings (Google style)
- Keep functions focused and small
- Follow existing patterns

### 3. Write Tests

All new code should have tests:

```python
# tests/test_your_feature.py
import pytest
from gridgulp import YourFeature

def test_your_feature():
    """Test that your feature works correctly."""
    result = YourFeature().do_something()
    assert result == expected_value

@pytest.mark.asyncio
async def test_async_feature():
    """Test async functionality."""
    result = await YourFeature().do_async()
    assert result is not None
```

### 4. Run Quality Checks

Before committing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gridgulp

# Run linting
ruff check .

# Run type checking
mypy src/gridgulp/

# Format code
ruff format .
```

### 5. Commit Your Changes

Write clear commit messages:

```bash
# Good commit messages
git commit -m "feat: add support for Excel 365 tables"
git commit -m "fix: handle empty cells in CSV detection"
git commit -m "docs: update configuration examples"

# Follow conventional commits:
# feat: new feature
# fix: bug fix
# docs: documentation
# test: testing
# refactor: code refactoring
# style: formatting
# perf: performance improvement
# chore: maintenance
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guide

### Python Style

We use Ruff for linting and formatting:

```python
# Good
def detect_tables(
    file_path: Path,
    confidence_threshold: float = 0.7,
) -> list[TableInfo]:
    """Detect tables in a spreadsheet file.

    Args:
        file_path: Path to the spreadsheet file
        confidence_threshold: Minimum confidence score (0.0-1.0)

    Returns:
        List of detected tables

    Raises:
        FileNotFoundError: If file doesn't exist
        ReaderError: If file cannot be read
    """
    tables = []
    # Implementation
    return tables
```

### Type Hints

Always use type hints:

```python
from typing import Optional, Union, Any
from pathlib import Path

# Use Union types for multiple options
FilePath = Union[str, Path]

# Use Optional for nullable values
def process(value: Optional[str] = None) -> str:
    return value or "default"

# Use specific types over Any when possible
def parse_data(data: dict[str, Any]) -> TableInfo:
    # Implementation
    pass
```

### Async/Await

Follow async patterns consistently:

```python
# Async function names should be descriptive
async def detect_tables_async(file_path: Path) -> list[TableInfo]:
    # Always use async context managers
    async with aiofiles.open(file_path) as f:
        content = await f.read()

    # Use asyncio.gather for parallel operations
    results = await asyncio.gather(
        process_sheet_async(sheet1),
        process_sheet_async(sheet2),
    )
    return results
```

## Testing Guidelines

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── fixtures/       # Test data files
└── conftest.py     # Shared fixtures
```

### Writing Tests

```python
import pytest
from gridgulp import GridGulp

class TestTableDetection:
    """Test table detection functionality."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample file for testing."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("col1,col2\n1,2\n3,4")
        return file_path

    def test_detect_simple_csv(self, sample_file):
        """Test detection of simple CSV file."""
        gg = GridGulp()
        result = gg.detect_tables_sync(sample_file)

        assert result.total_tables == 1
        assert result.sheets[0].tables[0].shape == (3, 2)

    @pytest.mark.parametrize("confidence", [0.5, 0.7, 0.9])
    def test_confidence_threshold(self, sample_file, confidence):
        """Test different confidence thresholds."""
        config = Config(confidence_threshold=confidence)
        gg = GridGulp(config=config)
        result = gg.detect_tables_sync(sample_file)
        # Assertions
```

### Performance Tests

```python
import pytest
import time

@pytest.mark.performance
def test_large_file_performance(large_excel_file):
    """Ensure large file processing is reasonably fast."""
    gg = GridGulp()

    start = time.time()
    result = gg.detect_tables_sync(large_excel_file)
    elapsed = time.time() - start

    assert elapsed < 5.0  # Should process in under 5 seconds
    assert result.total_tables > 0
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: int,
    param3: Optional[float] = None,
) -> tuple[str, int]:
    """Short description of function.

    Longer description if needed. Can span multiple lines and
    include examples or important notes.

    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Optional parameter description. Defaults to None.

    Returns:
        A tuple containing:
        - Processed string value
        - Count of operations

    Raises:
        ValueError: If param2 is negative
        TypeError: If param1 is not a string

    Examples:
        >>> result = complex_function("test", 5)
        >>> print(result)
        ('processed_test', 5)
    """
```

### Updating Documentation

1. Update relevant .md files in `docs_src/`
2. Add examples for new features
3. Update the changelog
4. Run docs locally to verify:
```bash
mkdocs serve
```

## Submitting Pull Requests

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No linting errors (`ruff check`)
- [ ] Type checks pass (`mypy`)
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] PR description is clear

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] My code follows the project style
- [ ] I've added tests for my changes
- [ ] I've updated documentation
- [ ] I've added a changelog entry
```

## Getting Help

### Resources

- [GitHub Issues](https://github.com/Ganymede-Bio/gridgulp/issues)
- [Discussions](https://github.com/Ganymede-Bio/gridgulp/discussions)
- [Documentation](https://ganymede-bio.github.io/gridgulp/)

### Communication

- **Bug Reports**: Use GitHub Issues with the bug template
- **Feature Requests**: Use GitHub Issues with the feature template
- **Questions**: Use GitHub Discussions
- **Security Issues**: Email security@ganymede.bio

## Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please read and follow it in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
