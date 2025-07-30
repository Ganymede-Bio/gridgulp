# Installation

## Requirements

- Python 3.10 or higher
- pip (Python package installer)

## Install from PyPI

The recommended way to install GridGulp is from PyPI:

```bash
pip install gridgulp
```

## Install from Source

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/Ganymede-Bio/gridgulp.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/Ganymede-Bio/gridgulp.git
cd gridgulp
pip install -e .
```

## Optional Dependencies

### Development Tools

To contribute to GridGulp or run tests, install the development dependencies:

```bash
pip install gridgulp[dev]
```

This includes:
- pytest for testing
- ruff for linting
- mypy for type checking
- pre-commit hooks

### Documentation

To build the documentation locally:

```bash
pip install gridgulp[docs]
mkdocs serve
```

## Verify Installation

After installation, verify that GridGulp is working:

```python
import gridgulp
print(gridgulp.__version__)
```

Or from the command line:

```bash
python -c "import gridgulp; print(gridgulp.__version__)"
```

## Troubleshooting

### ImportError: No module named 'magic'

GridGulp uses `python-magic` for file type detection. On some systems, you may need to install additional system dependencies:

**macOS:**
```bash
brew install libmagic
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libmagic1
```

**Windows:**
The Python package includes the necessary DLLs, but if you encounter issues, see the [python-magic documentation](https://github.com/ahupp/python-magic#installation).


## Next Steps

- Follow the [Quick Start](quickstart.md) guide to process your first spreadsheet
- Learn about [Jupyter notebook usage](jupyter.md) for interactive data exploration
- See the [User Guide](../user-guide/basic-usage.md) for detailed usage instructions
