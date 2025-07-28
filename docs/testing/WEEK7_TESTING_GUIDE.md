# Week 7 Testing Guide - VisionOrchestratorAgent

## Overview
This guide provides comprehensive testing procedures for Week 7's **VisionOrchestratorAgent**, the central AI coordination system that intelligently routes table detection based on sheet complexity, cost constraints, and confidence requirements.

## Test Environment Setup

### 1. Python Environment
```bash
# Ensure you're in the gridporter directory
cd ~/dev/gridporter

# Install development dependencies
pip install -e .

# Install additional testing dependencies
pip install pytest pytest-asyncio pytest-cov

# Install vision model dependencies
pip install openai httpx  # For OpenAI integration
```

### 2. API Keys Configuration

#### Using .env File (Recommended)
Create a `.env` file in the project root by copying the example:
```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:
```
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

#### Alternative: Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENAI_MODEL="gpt-4o-mini"
```

#### Ollama Setup (for local vision models)
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull vision model
ollama pull qwen2.5-vl:7b

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### 3. Test Configuration Files

Create test configuration files in the project root:

#### `test_config_basic.json` - Basic Testing
```json
{
  "use_vision": false,
  "confidence_threshold": 0.7,
  "max_cost_per_session": 1.0,
  "max_cost_per_file": 0.1,
  "enable_cache": true,
  "log_level": "INFO"
}
```

#### `test_config_vision.json` - Vision Testing
```json
{
  "use_vision": true,
  "openai_api_key": "${OPENAI_API_KEY}",
  "openai_model": "gpt-4o",
  "confidence_threshold": 0.8,
  "max_cost_per_session": 2.0,
  "max_cost_per_file": 0.2,
  "enable_region_verification": true,
  "verification_strict_mode": false,
  "log_level": "DEBUG"
}
```

#### `test_config_local.json` - Local Ollama Testing
```json
{
  "use_vision": true,
  "use_local_llm": true,
  "ollama_url": "http://localhost:11434",
  "ollama_vision_model": "qwen2.5-vl:7b",
  "confidence_threshold": 0.7,
  "max_cost_per_session": 0.0,
  "max_cost_per_file": 0.0,
  "log_level": "DEBUG"
}
```

#### `test_config_cost_conscious.json` - Budget Testing
```json
{
  "use_vision": true,
  "openai_api_key": "${OPENAI_API_KEY}",
  "max_cost_per_session": 0.20,
  "max_cost_per_file": 0.02,
  "confidence_threshold": 0.6,
  "enable_cache": true,
  "log_level": "INFO"
}
```

## Test Data Preparation

### 1. Simple Test Files

#### **simple_table.csv**
```csv
Product,Category,Price,Stock
Widget A,Electronics,29.99,150
Widget B,Electronics,39.99,75
Gadget X,Home,19.99,200
Gadget Y,Home,24.99,100
```

#### **simple_table.xlsx**
- Single table starting at A1
- Headers in row 1 (bold formatting)
- 5 rows of data
- Expected: Simple case detection, no vision needed

### 2. Complex Test Files

#### **complex_multi_table.xlsx**
- **Sheet1**: Two separate tables with gaps
  - Table 1: A1:D10 (Sales data)
  - Table 2: F1:I15 (Product inventory)
- **Sheet2**: Hierarchical data with merged headers
- Expected: Complex assessment, vision recommended

#### **sparse_data.xlsx**
- Large sheet (100x50) with scattered data islands
- Multiple small tables in different regions
- Lots of empty space between data
- Expected: High sparsity score, vision required

### 3. Edge Case Files

#### **merged_headers.xlsx**
- Complex multi-row headers with merged cells
- Hierarchical column structure
- Expected: High format complexity

#### **large_sheet.csv**
- 1000+ rows, 20+ columns
- Single large table
- Expected: Size complexity factor

## Core Functionality Tests

### 1. Complexity Assessment Testing

#### Test 1.1: Simple Sheet Assessment
```python
import asyncio
from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers import create_reader

async def test_simple_complexity():
    """Test complexity assessment for simple sheets."""
    config = Config(use_vision=False, confidence_threshold=0.7)
    orchestrator = VisionOrchestratorAgent(config)

    # Load simple test file
    reader = create_reader("simple_table.csv")
    sheet_data = list(reader.read_sheets())[0]

    # Assess complexity
    assessment = await orchestrator._assess_complexity(sheet_data)

    # Validate results
    print(f"Complexity Score: {assessment.complexity_score:.3f}")
    print(f"Requires Vision: {assessment.requires_vision}")
    print(f"Reasoning: {assessment.reasoning}")

    # Expected: Low complexity (< 0.6), no vision required
    assert assessment.complexity_score < 0.6
    assert not assessment.requires_vision
    assert "Simple" in assessment.reasoning

# Run test
asyncio.run(test_simple_complexity())
```

**Expected Output:**
```
Complexity Score: 0.235
Requires Vision: False
Reasoning: Simple, well-structured sheet suitable for traditional detection
```

#### Test 1.2: Complex Sheet Assessment
```python
async def test_complex_complexity():
    """Test complexity assessment for complex sheets."""
    config = Config(use_vision=False, confidence_threshold=0.7)
    orchestrator = VisionOrchestratorAgent(config)

    # Load complex test file
    reader = create_reader("complex_multi_table.xlsx")
    sheet_data = list(reader.read_sheets())[0]

    # Assess complexity
    assessment = await orchestrator._assess_complexity(sheet_data)

    print(f"Complexity Score: {assessment.complexity_score:.3f}")
    print(f"Assessment Factors:")
    for factor, value in assessment.assessment_factors.items():
        print(f"  {factor}: {value:.3f}")

    # Expected: High complexity (> 0.6), vision recommended
    assert assessment.complexity_score > 0.6
    assert assessment.requires_vision

asyncio.run(test_complex_complexity())
```

### 2. Strategy Selection Testing

#### Test 2.1: Cost-Conscious Strategy Selection
```python
async def test_cost_conscious_strategy():
    """Test strategy selection with tight budget constraints."""
    config = Config(
        use_vision=True,
        openai_api_key="test-key",  # Mock key
        max_cost_per_session=0.10,
        max_cost_per_file=0.02,
        confidence_threshold=0.6
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Create mock complexity assessment
    from gridporter.agents.vision_orchestrator_agent import ComplexityAssessment
    complexity = ComplexityAssessment(
        complexity_score=0.8,
        requires_vision=True,
        assessment_factors={"sparsity": 0.7, "pattern_complexity": 0.9},
        confidence=0.9,
        reasoning="Complex sheet with multiple regions"
    )

    # Make orchestration decision
    sheet_data = None  # Mock sheet data
    decision = await orchestrator._make_orchestration_decision(
        sheet_data, complexity, excel_metadata=None
    )

    print(f"Strategy: {decision.detection_strategy}")
    print(f"Use Vision: {decision.use_vision}")
    print(f"Cost Estimate: ${decision.cost_estimate:.4f}")
    print(f"Fallback Strategies: {decision.fallback_strategies}")

    # Expected: Conservative strategy due to budget constraints
    assert decision.cost_estimate <= 0.02
    assert len(decision.fallback_strategies) > 0

asyncio.run(test_cost_conscious_strategy())
```

#### Test 2.2: Quality-Focused Strategy Selection
```python
async def test_quality_focused_strategy():
    """Test strategy selection with higher budget for quality."""
    config = Config(
        use_vision=True,
        openai_api_key="test-key",
        max_cost_per_session=5.0,
        max_cost_per_file=0.5,
        confidence_threshold=0.9
    )

    orchestrator = VisionOrchestratorAgent(config)

    # High complexity assessment
    complexity = ComplexityAssessment(
        complexity_score=0.9,
        requires_vision=True,
        assessment_factors={"sparsity": 0.8, "merged_cell_ratio": 0.2},
        confidence=0.95,
        reasoning="Very complex sheet requiring vision analysis"
    )

    decision = await orchestrator._make_orchestration_decision(
        None, complexity, excel_metadata=None
    )

    print(f"Strategy: {decision.detection_strategy}")
    print(f"Vision Model: {decision.vision_model}")
    print(f"Decision Confidence: {decision.confidence:.2f}")

    # Expected: High-quality strategy with vision
    assert decision.use_vision
    assert decision.detection_strategy in ["full_vision", "hybrid_selective_vision"]
    assert decision.confidence >= 0.75

asyncio.run(test_quality_focused_strategy())
```

### 3. Multi-Model Integration Testing

#### Test 3.1: OpenAI Model Status Check
```python
import os

async def test_openai_model_status():
    """Test OpenAI vision model integration."""
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model="gpt-4o"
    )

    orchestrator = VisionOrchestratorAgent(config)
    status = await orchestrator.get_model_status()

    print("OpenAI Model Status:")
    print(f"  Available: {status['vision_model_available']}")
    print(f"  Model: {status['vision_model_name']}")
    print(f"  Reachable: {status.get('vision_model_reachable', 'N/A')}")
    print(f"  Cost Limits: ${status['cost_limits']['per_file']:.2f}")

    # Validate status
    if os.getenv("OPENAI_API_KEY"):
        assert status['vision_model_available']
        assert status['vision_model_name'] == "gpt-4o"
    else:
        print("⚠️  OPENAI_API_KEY not set - skipping API tests")

asyncio.run(test_openai_model_status())
```

#### Test 3.2: Ollama Model Status Check
```python
async def test_ollama_model_status():
    """Test Ollama local vision model integration."""
    config = Config(
        use_vision=True,
        use_local_llm=True,
        ollama_url="http://localhost:11434",
        ollama_vision_model="qwen2.5-vl:7b"
    )

    orchestrator = VisionOrchestratorAgent(config)
    status = await orchestrator.get_model_status()

    print("Ollama Model Status:")
    print(f"  Available: {status['vision_model_available']}")
    print(f"  Model: {status['vision_model_name']}")
    print(f"  Reachable: {status.get('vision_model_reachable', False)}")

    if not status.get('vision_model_reachable', False):
        print("⚠️  Ollama not running or model not available")
        print("   Run: ollama pull qwen2.5-vl:7b")
    else:
        assert status['vision_model_name'] == "qwen2.5-vl:7b"

asyncio.run(test_ollama_model_status())
```

### 4. Full Pipeline Testing

#### Test 4.1: End-to-End Simple File Processing
```python
async def test_e2e_simple_file():
    """Test complete orchestration pipeline with simple file."""
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_file=0.05,
        confidence_threshold=0.8
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Process simple file
    reader = create_reader("simple_table.csv")
    sheet_data = list(reader.read_sheets())[0]

    result = await orchestrator.orchestrate_detection(sheet_data)

    print("End-to-End Results:")
    print(f"  Tables Detected: {len(result.tables)}")
    print(f"  Complexity Score: {result.complexity_assessment.complexity_score:.3f}")
    print(f"  Strategy Used: {result.orchestrator_decision.detection_strategy}")
    print(f"  Vision Used: {result.orchestrator_decision.use_vision}")
    print(f"  Processing Time: {result.processing_metadata['processing_time_seconds']:.2f}s")
    print(f"  Cost: ${result.cost_report.get('total_cost_usd', 0):.4f}")

    # Validate results
    assert len(result.tables) >= 1
    assert result.complexity_assessment.complexity_score < 0.6  # Simple file
    assert not result.orchestrator_decision.use_vision  # Should use free methods
    assert result.processing_metadata['processing_time_seconds'] < 5.0

asyncio.run(test_e2e_simple_file())
```

#### Test 4.2: Budget Exhaustion Testing
```python
async def test_budget_exhaustion():
    """Test behavior when budget is exhausted."""
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_session=0.01,  # Very low budget
        max_cost_per_file=0.005,
        confidence_threshold=0.6
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Process multiple files to exhaust budget
    test_files = ["simple_table.csv", "complex_multi_table.xlsx"]

    for i, filename in enumerate(test_files):
        try:
            reader = create_reader(filename)
            sheet_data = list(reader.read_sheets())[0]

            result = await orchestrator.orchestrate_detection(sheet_data)

            print(f"File {i+1} ({filename}):")
            print(f"  Strategy: {result.orchestrator_decision.detection_strategy}")
            print(f"  Cost: ${result.cost_report.get('total_cost_usd', 0):.4f}")

            status = await orchestrator.get_model_status()
            print(f"  Session Cost: ${status['session_cost']:.4f}")

            # Later files should use traditional methods due to budget
            if i > 0:
                assert not result.orchestrator_decision.use_vision

        except FileNotFoundError:
            print(f"  ⚠️  Test file {filename} not found - skipping")

asyncio.run(test_budget_exhaustion())
```

## Performance Testing

### 1. Processing Time Benchmarks

#### Performance Test Script
```python
import time
import statistics
from typing import List

async def benchmark_orchestrator_performance():
    """Benchmark VisionOrchestratorAgent performance across different scenarios."""

    test_scenarios = [
        {
            "name": "Simple CSV (no vision)",
            "config": Config(use_vision=False, confidence_threshold=0.7),
            "file": "simple_table.csv",
            "expected_time": 0.1  # seconds
        },
        {
            "name": "Complex Excel (vision)",
            "config": Config(
                use_vision=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                confidence_threshold=0.8
            ),
            "file": "complex_multi_table.xlsx",
            "expected_time": 3.0  # seconds (with vision)
        },
        {
            "name": "Local Ollama",
            "config": Config(
                use_vision=True,
                use_local_llm=True,
                ollama_url="http://localhost:11434",
                ollama_vision_model="qwen2.5-vl:7b"
            ),
            "file": "simple_table.csv",
            "expected_time": 2.0  # seconds
        }
    ]

    print("Performance Benchmark Results:")
    print("=" * 60)

    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")

        try:
            orchestrator = VisionOrchestratorAgent(scenario['config'])
            reader = create_reader(scenario['file'])
            sheet_data = list(reader.read_sheets())[0]

            # Run multiple iterations
            times = []
            for i in range(3):
                start_time = time.time()
                result = await orchestrator.orchestrate_detection(sheet_data)
                elapsed = time.time() - start_time
                times.append(elapsed)

            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)

            print(f"  Average Time: {avg_time:.3f}s")
            print(f"  Min Time: {min_time:.3f}s")
            print(f"  Max Time: {max_time:.3f}s")
            print(f"  Tables Found: {len(result.tables)}")
            print(f"  Strategy: {result.orchestrator_decision.detection_strategy}")

            # Performance validation
            if avg_time > scenario['expected_time'] * 2:
                print(f"  ⚠️  Performance warning: Expected ~{scenario['expected_time']}s")
            else:
                print(f"  ✅ Performance acceptable")

        except Exception as e:
            print(f"  ❌ Test failed: {e}")

asyncio.run(benchmark_orchestrator_performance())
```

### 2. Memory Usage Monitoring

```python
import psutil
import os

async def monitor_memory_usage():
    """Monitor memory usage during orchestration."""
    process = psutil.Process(os.getpid())

    config = Config(use_vision=False, confidence_threshold=0.7)
    orchestrator = VisionOrchestratorAgent(config)

    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial Memory: {initial_memory:.1f} MB")

    # Process file
    reader = create_reader("simple_table.csv")
    sheet_data = list(reader.read_sheets())[0]

    result = await orchestrator.orchestrate_detection(sheet_data)

    # Final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    print(f"Final Memory: {final_memory:.1f} MB")
    print(f"Memory Increase: {memory_increase:.1f} MB")

    # Memory usage should be reasonable
    assert memory_increase < 100  # Less than 100MB increase

asyncio.run(monitor_memory_usage())
```

## Automated Testing

### 1. Running Unit Tests

```bash
# Run all VisionOrchestratorAgent tests
python -m pytest tests/agents/test_vision_orchestrator_agent.py -v

# Run with coverage
python -m pytest tests/agents/test_vision_orchestrator_agent.py --cov=src/gridporter/agents/vision_orchestrator_agent --cov-report=html

# Run specific test categories
python -m pytest tests/agents/test_vision_orchestrator_agent.py::TestVisionOrchestratorAgent::test_assess_complexity_simple_sheet -v

# Run async tests with detailed output
python -m pytest tests/agents/test_vision_orchestrator_agent.py -v -s --tb=short
```

**Expected Output:**
```
tests/agents/test_vision_orchestrator_agent.py::TestVisionOrchestratorAgent::test_init_with_vision PASSED
tests/agents/test_vision_orchestrator_agent.py::TestVisionOrchestratorAgent::test_init_without_vision PASSED
tests/agents/test_vision_orchestrator_agent.py::TestVisionOrchestratorAgent::test_assess_complexity_simple_sheet PASSED
tests/agents/test_vision_orchestrator_agent.py::TestVisionOrchestratorAgent::test_make_orchestration_decision_simple PASSED
...
```

### 2. Integration Tests

```bash
# Run integration tests with real files
python -m pytest tests/integration/ -k "orchestrator" -v

# Test with different configurations
OPENAI_API_KEY=your-key python -m pytest tests/integration/test_orchestrator_integration.py -v
```

### 3. Mock Testing for Vision Models

```python
# Example mock test
from unittest.mock import AsyncMock, MagicMock, patch

@patch('gridporter.agents.vision_orchestrator_agent.create_vision_model')
async def test_vision_model_mocking(mock_create_vision_model):
    """Test orchestrator with mocked vision model."""
    # Setup mock vision model
    mock_vision_model = MagicMock()
    mock_vision_model.name = "gpt-4o"
    mock_vision_model.check_model_available = AsyncMock(return_value=True)
    mock_create_vision_model.return_value = mock_vision_model

    config = Config(use_vision=True, openai_api_key="test-key")
    orchestrator = VisionOrchestratorAgent(config)

    # Test model status
    status = await orchestrator.get_model_status()
    assert status['vision_model_available']
    assert status['vision_model_name'] == "gpt-4o"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: "Vision model not available"
**Symptoms:** `vision_model_available: False` in status check
**Solutions:**
1. Check API key: `echo $OPENAI_API_KEY`
2. Verify Ollama service: `curl http://localhost:11434/api/tags`
3. Check model availability: `ollama list`

#### Issue 2: "Budget exhausted" warnings
**Symptoms:** Strategy falls back to traditional methods
**Solutions:**
1. Increase budget limits in config
2. Check session cost: `status['session_cost']`
3. Clear cost optimizer cache

#### Issue 3: Slow performance
**Symptoms:** Processing takes >10 seconds for simple files
**Solutions:**
1. Disable vision for testing: `use_vision: false`
2. Check network connectivity for API calls
3. Use local Ollama model for faster processing

#### Issue 4: Import errors
**Symptoms:** `ModuleNotFoundError` when running tests
**Solutions:**
1. Install in development mode: `pip install -e .`
2. Check Python path: `python -c "import gridporter; print(gridporter.__file__)"`
3. Verify dependencies: `pip install pytest pytest-asyncio`

### Debug Mode Testing

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

config = Config(
    use_vision=True,
    log_level="DEBUG",
    enable_debug=True
)

# This will show detailed orchestration decision-making
orchestrator = VisionOrchestratorAgent(config)
```

## Test Validation Checklist

### ✅ Core Functionality
- [ ] Complexity assessment works for simple and complex sheets
- [ ] Strategy selection respects budget constraints
- [ ] Multi-model support (OpenAI and Ollama) functions correctly
- [ ] Fallback strategies activate when primary methods fail
- [ ] Cost tracking and reporting is accurate

### ✅ Performance Requirements
- [ ] Simple file processing completes in <1 second (no vision)
- [ ] Complex file processing completes in <10 seconds (with vision)
- [ ] Memory usage increases by <100MB during processing
- [ ] Concurrent processing works without conflicts

### ✅ Integration Testing
- [ ] Works with existing Week 6 hybrid detection
- [ ] Integrates properly with ComplexTableAgent
- [ ] File readers work correctly with orchestrator
- [ ] Configuration system functions as expected

### ✅ Error Handling
- [ ] Graceful degradation when vision models unavailable
- [ ] Budget exhaustion handled properly
- [ ] Invalid file formats handled gracefully
- [ ] Network errors don't crash the system

### ✅ Real-World Usage
- [ ] Processes typical Excel and CSV files correctly
- [ ] Handles edge cases (empty files, malformed data)
- [ ] Provides useful status information and logging
- [ ] Documentation and examples work as intended

## Conclusion

This testing guide provides comprehensive coverage of the VisionOrchestratorAgent functionality. Use it to:

1. **Validate new installations** and ensure proper setup
2. **Test configuration changes** and custom scenarios
3. **Benchmark performance** across different sheet types
4. **Troubleshoot issues** and verify fixes
5. **Develop extensions** and new features

For additional support, refer to the [Week 7 Summary](../WEEK7_SUMMARY.md) and [example code](../../examples/week7_orchestrator_examples.py).

---

**Next Steps:** After validating Week 7 functionality, proceed to Week 8 testing for Rich Metadata & Output Generation features.
