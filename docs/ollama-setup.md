# Ollama Setup Guide for GridPorter

This guide provides detailed instructions for setting up Ollama with GridPorter's recommended models for local, private spreadsheet analysis.

## Overview

GridPorter uses two specialized models when running with Ollama:
- **DeepSeek-R1** (`deepseek-r1:7b`): Advanced reasoning model for table naming and data analysis
- **Qwen2.5-VL** (`qwen2.5vl:7b`): Vision-language model for spreadsheet layout and visual analysis

## Installation

### 1. Install Ollama

#### macOS/Linux (Recommended)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download the installer from [ollama.ai](https://ollama.ai/download) and run it.

#### Alternative: Manual Installation
```bash
# macOS with Homebrew
brew install ollama

# Linux (manual)
wget https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64
sudo mv ollama-linux-amd64 /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama
```

### 2. Start Ollama Server

```bash
# Start Ollama service (runs in background)
ollama serve
```

The server will start on `http://localhost:11434` by default.

### 3. Pull Required Models

#### DeepSeek-R1 (Text/Reasoning)
```bash
# Recommended for most users (6GB RAM)
ollama pull deepseek-r1:7b

# For lower-end hardware (2GB RAM)
ollama pull deepseek-r1:1.5b

# For high-performance tasks (21GB RAM)
ollama pull deepseek-r1:32b
```

#### Qwen2.5-VL (Vision/Analysis)
```bash
# Recommended for most users (6GB RAM)
ollama pull qwen2.5vl:7b

# For enhanced accuracy (21GB RAM)
ollama pull qwen2.5vl:32b

# Maximum performance (71GB RAM)
ollama pull qwen2.5vl:72b
```

### 4. Verify Installation

```bash
# List installed models
ollama list

# Test text model
ollama run deepseek-r1:7b "What are the key components of a financial balance sheet?"

# Test vision model (if you have an image)
ollama run qwen2.5vl:7b "Describe the structure of this spreadsheet" < path/to/spreadsheet_screenshot.png
```

## Configuration

### Environment Variables

Create a `.env` file in your GridPorter project:

```env
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=deepseek-r1:7b
OLLAMA_VISION_MODEL=qwen2.5vl:7b

# Optional: Force Ollama usage even if OpenAI key is present
GRIDPORTER_USE_LOCAL_LLM=true

# Optional: Disable LLM features entirely
GRIDPORTER_SUGGEST_NAMES=false
```

### Programmatic Configuration

```python
from gridporter import GridPorter, Config

# Basic Ollama configuration
config = Config(
    use_local_llm=True,
    ollama_text_model="deepseek-r1:7b",
    ollama_vision_model="qwen2.5vl:7b",
    ollama_url="http://localhost:11434"
)

porter = GridPorter(config=config)
```

### Advanced Configuration

```python
# Custom Ollama server (e.g., remote instance)
config = Config(
    use_local_llm=True,
    ollama_url="http://remote-server:11434",
    ollama_text_model="deepseek-r1:32b",  # High-performance model
    ollama_vision_model="qwen2.5vl:32b",
    llm_temperature=0.1,  # More deterministic responses
    max_tokens_per_table=100  # Longer table descriptions
)

# Multiple model sizes for different tasks
config_lightweight = Config(
    use_local_llm=True,
    ollama_text_model="deepseek-r1:1.5b",  # For resource-constrained environments
    ollama_vision_model="qwen2.5vl:7b"
)
```

## Model Selection Guide

### DeepSeek-R1 Variants

| Model | RAM Required | Use Case | Performance |
|-------|-------------|----------|-------------|
| `deepseek-r1:1.5b` | 2GB | Basic naming, low-end hardware | Good |
| `deepseek-r1:7b` | 6GB | **Recommended** for most users | Excellent |
| `deepseek-r1:14b` | 10GB | Enhanced reasoning | Very Good |
| `deepseek-r1:32b` | 21GB | High-performance tasks | Outstanding |
| `deepseek-r1:70b` | 45GB | Maximum accuracy | Exceptional |

### Qwen2.5-VL Variants

| Model | RAM Required | Use Case | Performance |
|-------|-------------|----------|-------------|
| `qwen2.5vl:7b` | 6GB | **Recommended** for most users | Excellent |
| `qwen2.5vl:32b` | 21GB | Complex visual analysis | Outstanding |
| `qwen2.5vl:72b` | 71GB | Maximum visual understanding | Exceptional |

## Performance Optimization

### Hardware Recommendations

**Minimum Requirements:**
- 8GB RAM
- 4 CPU cores
- 10GB free disk space

**Recommended Setup:**
- 16GB RAM
- 8 CPU cores
- SSD storage
- GPU (optional, for faster inference)

**High-Performance Setup:**
- 32GB+ RAM
- 16+ CPU cores
- NVMe SSD
- RTX 4080/4090 or similar GPU

### GPU Acceleration

If you have a compatible GPU, Ollama will automatically use it for faster inference:

```bash
# Check if GPU is being used
ollama ps

# Force CPU-only mode (if needed)
OLLAMA_NUM_GPU=0 ollama serve
```

### Memory Management

```bash
# Set memory limits for models
export OLLAMA_MAX_MEMORY=8GB

# Limit concurrent model loading
export OLLAMA_MAX_MODELS=2
```

## Usage Examples

### Basic Table Detection with Ollama

```python
import asyncio
from gridporter import GridPorter

async def analyze_spreadsheet():
    # GridPorter will auto-detect Ollama if no OpenAI key is configured
    porter = GridPorter()

    # Process a financial spreadsheet
    result = await porter.detect_tables("examples/financial/balance_sheet.xlsx")

    print(f"Detected {result.total_tables} tables")
    print(f"LLM calls made: {result.llm_calls}")
    print(f"Processing time: {result.detection_time:.2f}s")

    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for table in sheet.tables:
            print(f"  - {table.suggested_name}: {table.range.excel_range}")

asyncio.run(analyze_spreadsheet())
```

### Vision-Enhanced Analysis

```python
async def analyze_complex_layout():
    config = Config(
        use_local_llm=True,
        ollama_vision_model="qwen2.5vl:32b",  # Enhanced vision model
        suggest_names=True
    )

    porter = GridPorter(config=config)

    # Process spreadsheet with complex layouts, merged cells, charts
    result = await porter.detect_tables("examples/complex/quarterly_report.xlsx")

    # The vision model will help identify table boundaries and structures
    return result
```

### Batch Processing with Ollama

```python
async def batch_process_files():
    porter = GridPorter(use_local_llm=True)

    files = [
        "examples/financial/income_statement.xlsx",
        "examples/sales/monthly_sales.csv",
        "examples/complex/pivot_table_report.xlsx"
    ]

    results = await porter.batch_detect(files)

    for result in results:
        print(f"File: {result.file_info.path}")
        print(f"Tables found: {result.total_tables}")
        print(f"Success rate: {result.success_rate:.1%}")
```

## Troubleshooting

### Common Issues

**1. "Connection refused" error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

**2. "Model not found" error**
```bash
# Verify models are installed
ollama list

# Pull missing models
ollama pull deepseek-r1:7b
ollama pull qwen2.5vl:7b
```

**3. Out of memory errors**
```bash
# Use smaller models
export OLLAMA_TEXT_MODEL=deepseek-r1:1.5b
export OLLAMA_VISION_MODEL=qwen2.5vl:7b

# Or increase system memory/swap
```

**4. Slow inference**
```bash
# Check system resources
htop
nvidia-smi  # For GPU usage

# Use smaller models for faster inference
ollama pull deepseek-r1:1.5b
```

### Performance Tuning

**Concurrent Requests:**
```python
# Limit concurrent LLM calls to prevent memory issues
config = Config(
    use_local_llm=True,
    max_concurrent_requests=2  # Adjust based on available memory
)
```

**Model Switching:**
```python
# Use different models for different tasks
config = Config(
    use_local_llm=True,
    ollama_text_model="deepseek-r1:1.5b",  # Fast for simple naming
    ollama_vision_model="qwen2.5vl:32b"    # Accurate for complex layouts
)
```

## Security Considerations

### Data Privacy
- All processing happens locally - no data leaves your system
- Models run entirely offline after initial download
- No API keys or external connections required

### Network Security
```bash
# Bind Ollama to localhost only (default)
OLLAMA_HOST=127.0.0.1:11434 ollama serve

# For remote access (use with caution)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Model Verification
```bash
# Verify model checksums (if available)
ollama list --verbose

# Check model sources
ollama show deepseek-r1:7b
ollama show qwen2.5vl:7b
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test GridPorter with Ollama
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Ollama
        run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama serve &
          sleep 10

      - name: Pull models
        run: |
          ollama pull deepseek-r1:1.5b  # Smaller model for CI
          ollama pull qwen2.5vl:7b

      - name: Run tests
        run: |
          export GRIDPORTER_USE_LOCAL_LLM=true
          export OLLAMA_TEXT_MODEL=deepseek-r1:1.5b
          pytest tests/
```

### Docker Integration

```dockerfile
FROM ollama/ollama:latest

# Install Python and GridPorter
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Pre-pull models
RUN ollama serve & sleep 10 && \
    ollama pull deepseek-r1:7b && \
    ollama pull qwen2.5vl:7b

# Set environment
ENV GRIDPORTER_USE_LOCAL_LLM=true
ENV OLLAMA_TEXT_MODEL=deepseek-r1:7b
ENV OLLAMA_VISION_MODEL=qwen2.5vl:7b

COPY . /app
WORKDIR /app

CMD ["python3", "-m", "gridporter"]
```

## Next Steps

1. **Install Ollama and models** following the instructions above
2. **Test the setup** with the provided examples
3. **Explore model variants** to find the best performance/resource balance
4. **Integrate with your workflows** using the Python API or CLI
5. **Monitor performance** and adjust configuration as needed

For additional help, see the main [README](../README.md) or open an issue on GitHub.
