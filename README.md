# GridPorter

An intelligent spreadsheet ingestion framework with automatic multi-table detection using AI agents.

## Overview

GridPorter is a Python framework that automatically detects and extracts multiple tables from complex spreadsheets (Excel and CSV files). It uses a hierarchical detection strategy combined with LLM-powered naming suggestions to provide a robust solution for spreadsheet data extraction.

### What's New in v0.2.1

- **Semantic Understanding**: GridPorter now understands the meaning and structure of complex tables, not just their layout
- **Multi-Row Headers**: Automatic detection of hierarchical headers with merged cells
- **Financial Reports**: Built-in support for financial statements, pivot tables, and reports with subtotals
- **Feature Collection**: Optional telemetry system to help improve detection accuracy
- **100% Test Coverage**: Comprehensive test suite ensuring reliability

## Features

- **Multi-format Support**: Handles Excel files (.xlsx, .xls, .xlsm, .xlsb) and CSV files
- **Intelligent Table Detection**: Multiple detection strategies including:
  - Single table fast check
  - Excel ListObjects detection
  - Vision-based detection with bitmap analysis
  - Complex table agent for semantic understanding
  - Multi-row header detection with merged cells
  - Format and semantic structure analysis
- **Semantic Understanding** (New in v0.2.1):
  - Multi-row header detection with column hierarchy
  - Merged cell analysis and mapping
  - Section and subtotal detection
  - Format pattern recognition
  - Preservation of semantic blank rows
- **AI-Powered Features**:
  - Table naming suggestions using LLMs
  - Vision model integration for complex layouts
  - Confidence scoring with multi-factor analysis
- **Feature Collection System**:
  - Telemetry for continuous improvement
  - 40+ detection metrics tracked
  - Privacy-preserving local storage
  - Export capabilities for analysis
- **Performance & Reliability**:
  - Async processing for better performance
  - High-speed CalamineReader for Excel files
  - Memory-efficient processing of large files
  - Comprehensive error handling
- **Developer Experience**:
  - Type-safe with Pydantic 2 models
  - Extensive configuration options
  - Easy to extend with new detectors
  - Well-documented API

## Repository Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/gridporter.git
cd gridporter

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in development mode (recommended method)
make dev
# This installs dependencies, sets up pre-commit hooks, and configures the environment

# 4. Verify installation
python -c "import gridporter; print(f'GridPorter v{gridporter.__version__} installed successfully!')"
```

### Alternative Setup Methods

#### Using uv (Fastest)
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

#### Using pip
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install from requirements files
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### Environment Configuration

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` to add your API keys (optional for basic functionality):**
   ```bash
   # Required only if using LLM features
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

### Development Commands

```bash
# Run tests
make test

# Run linting and formatting
make check

# Build the package
make build

# Clean build artifacts
make clean
```

## Installation (PyPI - Coming Soon)

```bash
# Once published to PyPI
pip install gridporter

# For development
pip install gridporter[dev]

# With visualization support
pip install gridporter[visualization]
```

### Troubleshooting

**Dependency resolution issues:**
- Make sure you're using Python 3.10+
- Try updating pip: `pip install --upgrade pip`
- Use uv for faster dependency resolution: `pip install uv`

**Virtual environment issues:**
- Delete `.venv` and recreate: `rm -rf .venv && python -m venv .venv`
- Make sure you're activating the correct environment

## Quick Start

```python
import asyncio
from gridporter import GridPorter

async def main():
    # Initialize GridPorter
    porter = GridPorter()

    # Detect tables in an Excel file
    result = await porter.detect_tables("data/sales_report.xlsx")

    # Print file information
    print(f"File: {result.file_info.path}")
    print(f"Type: {result.file_info.type}")
    print(f"Size: {result.file_info.size_mb:.1f} MB")
    print(f"Processing time: {result.metadata['detection_time']:.2f}s")

    # Show detected tables
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for table in sheet.tables:
            print(f"  Table: {table.suggested_name or table.range}")
            print(f"  Range: {table.range}")
            print(f"  Confidence: {table.confidence:.2%}")
            print(f"  Method: {table.detection_method}")
            if table.multi_row_headers:
                print(f"  Header rows: {table.multi_row_headers}")

# Run the async function
asyncio.run(main())
```

### Advanced Features (v0.2.1)

#### Multi-Row Header Detection

```python
from gridporter import GridPorter, Config

# Enable all advanced features
config = Config(
    use_vision=True,  # Use vision-based detection
    suggest_names=True,  # Use LLM for table naming
    confidence_threshold=0.7
)

porter = GridPorter(config=config)

# Detect complex financial report
result = await porter.detect_tables("financial_report.xlsx")

# The system automatically detects:
# - Multi-row headers with merged cells
# - Hierarchical column structures
# - Subtotals and grand totals
# - Section boundaries
```

#### Semantic Structure Analysis

```python
# GridPorter understands semantic meaning
for table in result.sheets[0].tables:
    if table.semantic_features:
        print(f"Table has {table.semantic_features['section_count']} sections")
        print(f"Contains subtotals: {table.semantic_features['has_subtotals']}")
        print(f"Has grand total: {table.semantic_features['has_grand_total']}")
```

#### Feature Collection for Continuous Improvement

```python
# Enable feature collection to help improve detection
config = Config(
    enable_feature_collection=True,
    feature_db_path="~/.gridporter/features.db"
)

porter = GridPorter(config=config)

# Features are automatically collected during detection
result = await porter.detect_tables("complex_spreadsheet.xlsx")

# Export features for analysis
from gridporter.telemetry import get_feature_collector
collector = get_feature_collector()
collector.export_features("detection_features.csv")
```

### Basic Configuration

```python
from gridporter import GridPorter, Config

# Create configuration
config = Config(
    suggest_names=False,  # No LLM calls
    max_file_size_mb=10,
    confidence_threshold=0.8
)

# Initialize with config
porter = GridPorter(config=config)

# Basic file analysis (currently implemented)
result = await porter.detect_tables("your_file.xlsx")
print(f"File analyzed: {result.file_info.type}")
print(f"Ready for detection implementation!")
```

## CLI Usage

```bash
# Detect tables in a file
gridporter detect data/report.xlsx

# With verbose output
gridporter detect data/report.xlsx --verbose

# Export results to JSON
gridporter detect data/report.xlsx --output results.json

# Process multiple files
gridporter detect data/*.xlsx --output-dir results/
```

## Architecture

GridPorter uses a multi-stage detection pipeline:

1. **File Type Detection**: Validates file type using magic bytes
2. **Single Table Check**: Fast check for simple single-table sheets
3. **ListObjects Detection**: Checks for Excel's native table objects
4. **Island Detection**: Uses image processing to find disconnected regions
5. **Heuristic Enhancement**: Applies formatting and header analysis
6. **AI Naming**: Suggests meaningful names using LLMs

## Configuration

GridPorter supports both cloud-based (OpenAI) and local (Ollama) LLM providers for intelligent table naming and analysis.

### Option 1: OpenAI (Cloud-based)

Create a `.env` file for API keys:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Option 2: Ollama (Local, Private, Free)

**Recommended Models:**
- **Text/Reasoning**: `deepseek-r1:7b` - Excellent at logical reasoning and tool usage
- **Vision/Analysis**: `qwen2.5vl:7b` - Advanced vision model for spreadsheet analysis

**Quick Setup:**

```bash
# 1. Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull the recommended models
ollama pull deepseek-r1:7b
ollama pull qwen2.5vl:7b

# 3. GridPorter will automatically use Ollama if no OpenAI key is configured
```

**Environment Configuration:**

```env
# Ollama Configuration (optional - these are the defaults)
OLLAMA_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=deepseek-r1:7b
OLLAMA_VISION_MODEL=qwen2.5vl:7b

# Force local LLM usage even if OpenAI key is available
GRIDPORTER_USE_LOCAL_LLM=true
```

### Programmatic Configuration

```python
from gridporter import GridPorter, Config

# OpenAI Configuration
config_openai = Config(
    openai_api_key="your_api_key",
    openai_model="gpt-4o-mini",
    use_local_llm=False
)

# Ollama Configuration
config_ollama = Config(
    use_local_llm=True,
    ollama_text_model="deepseek-r1:7b",
    ollama_vision_model="qwen2.5vl:7b",
    ollama_url="http://localhost:11434"
)

# Auto-detection (uses Ollama if no OpenAI key)
config_auto = Config.from_env()

porter = GridPorter(config=config_auto)
```

### Model Capabilities & Hardware Requirements

**DeepSeek-R1 (Text/Reasoning)**
- **Strengths**: Mathematical reasoning, code generation, logical inference, tool usage
- **Use Cases**: Table naming, data type inference, pattern recognition
- **Hardware**:
  - `1.5b`: 2GB RAM, CPU inference
  - `7b`: 6GB RAM, optimal for most users
  - `32b`: 21GB RAM, high-performance tasks

**Qwen2.5-VL (Vision/Analysis)**
- **Strengths**: Document parsing, chart analysis, visual table detection, multilingual support
- **Use Cases**: Spreadsheet layout analysis, merged cell detection, table boundary identification
- **Hardware**:
  - `7b`: 6GB RAM, recommended for most users
  - `32b`: 21GB RAM, enhanced accuracy
  - `72b`: 71GB RAM, maximum performance

**Performance Comparison**:
- **Privacy**: Ollama keeps all data local, no cloud API calls
- **Cost**: Free after initial setup, no per-request charges
- **Speed**: Local inference, no network latency
- **Accuracy**: DeepSeek-R1 matches GPT-4 performance on many reasoning tasks

## Output Format

GridPorter returns structured data using Pydantic models:

```python
{
    "file_info": {
        "path": "data/report.xlsx",
        "type": "xlsx",
        "size": 1048576,
        "detected_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    },
    "sheets": [
        {
            "name": "Sheet1",
            "tables": [
                {
                    "range": "A1:E100",
                    "suggested_name": "sales_data",
                    "confidence": 0.95,
                    "detection_method": "complex_detection",
                    "headers": ["Date", "Product", "Quantity", "Price", "Total"],
                    "multi_row_headers": 2,  // Number of header rows
                    "column_hierarchy": {  // Hierarchical header mapping
                        "2": ["Sales", "Quantity"],
                        "3": ["Sales", "Price"],
                        "4": ["Sales", "Total"]
                    },
                    "semantic_features": {
                        "has_subtotals": true,
                        "has_grand_total": true,
                        "section_count": 3,
                        "has_merged_cells": true
                    },
                    "data_preview": [...]
                }
            ]
        }
    ],
    "metadata": {
        "detection_time": 1.23,
        "methods_used": ["single_table", "complex_detection", "semantic_analysis"],
        "llm_calls": 2,
        "llm_tokens": 450,
        "feature_collection_enabled": true
    }
}
```

## Advanced Usage

### Custom Detection Strategies

```python
from gridporter.detectors import BaseDetector

class MyCustomDetector(BaseDetector):
    async def detect(self, sheet_data):
        # Your detection logic here
        pass

# Register the detector
porter = GridPorter()
porter.register_detector(MyCustomDetector())
```

### Batch Processing

```python
from gridporter import GridPorter

async def process_batch(file_paths):
    porter = GridPorter()

    # Process files concurrently
    tasks = [porter.detect_tables(path) for path in file_paths]
    results = await asyncio.gather(*tasks)

    return results
```

### Complex Table Detection

```python
from gridporter.agents import ComplexTableAgent
from gridporter.config import Config

# Direct usage of ComplexTableAgent for advanced control
agent = ComplexTableAgent(Config())

# Detect complex tables with semantic understanding
tables = await agent.detect_complex_tables(sheet_data)

for table in tables:
    print(f"Table: {table.range}")
    print(f"Header rows: {table.multi_row_headers}")
    print(f"Has sections: {table.semantic_features.get('section_count', 0) > 0}")
    print(f"Confidence: {table.confidence:.2%}")
```

### Feature Analysis

```python
from gridporter.telemetry import get_feature_collector
import pandas as pd

# Analyze collected features
collector = get_feature_collector()
if collector.enabled:
    # Get summary statistics
    stats = collector.get_summary_statistics()
    print(f"Total detections: {stats['total_records']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")

    # Export for detailed analysis
    collector.export_features("features.csv")

    # Load in pandas for analysis
    df = pd.read_csv("features.csv")
    print(df.groupby('pattern_type')['confidence'].mean())
```

### Visualization

```python
from gridporter.utils.visualization import visualize_detection

# Visualize detection results
result = await porter.detect_tables("data/complex_sheet.xlsx")
visualize_detection(result, sheet_index=0, output_path="detection_viz.png")
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/gridporter.git
cd gridporter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gridporter

# Run specific test file
pytest tests/test_detectors.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [TableSense](https://github.com/microsoft/TableSense) and [table-transformer](https://github.com/microsoft/table-transformer)
- Built with [openai-agents-python](https://github.com/openai/openai-agents-python)
- Uses [openpyxl](https://openpyxl.readthedocs.io/) and [xlrd](https://xlrd.readthedocs.io/) for Excel handling

## Roadmap

### Completed (v0.2.1)
- [x] Multi-row header detection
- [x] Merged cell analysis
- [x] Semantic structure understanding
- [x] Feature collection system
- [x] Complex table detection agent

### Upcoming (v0.3.0)
- [ ] Support for Google Sheets API
- [ ] WebSocket support for real-time updates
- [ ] Cloud storage integration
- [ ] Real-time collaboration features

### Future (v0.4.0+)
- [ ] ML-based table detection using table-transformer
- [ ] Automatic data type inference
- [ ] Table relationship detection
- [ ] Export to Parquet/Arrow formats
- [ ] Web UI for confirmation workflow
- [ ] Plugin system for custom processors
