# GridPorter Manual Testing Guide

## Overview

This guide provides comprehensive manual testing procedures for the GridPorter framework, focusing on the refactored architecture with the VisionOrchestratorAgent at its core.

## Prerequisites

### 1. Environment Setup

```bash
# Install GridPorter in development mode
cd ~/dev/gridporter
pip install -e ".[dev]"

# Set up API keys
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENAI_MODEL="gpt-4o-mini"

# Optional: Set up Ollama for local models
ollama pull qwen2.5-vl:7b
```

### 2. Test Data Requirements

Ensure you have the following test files in `tests/manual/`:
- Level 0: Basic files (single tables, various formats)
- Level 1: Medium complexity (multiple tables, large files)
- Level 2: Complex files (creative layouts, weird structures)

## Testing Scenarios

### 1. Basic Functionality Tests

#### Test 1.1: Simple CSV Detection
```python
import asyncio
from gridporter import GridPorter
from gridporter.config import Config

async def test_simple_csv():
    """Test simple CSV file detection."""
    config = Config(
        use_vision=False,
        confidence_threshold=0.7
    )

    gp = GridPorter(config)
    result = await gp.extract_from_file("tests/manual/level0/test_comma.csv")

    # Verify:
    # - Single table detected
    # - Headers correctly identified
    # - Data types preserved
    print(f"Tables found: {len(result.tables)}")
    for table in result.tables:
        print(f"  Range: {table.range.excel_range}")
        print(f"  Confidence: {table.confidence}")
        print(f"  Headers: {table.headers[:5] if table.headers else 'None'}")

asyncio.run(test_simple_csv())
```

**Expected Results:**
- ✅ Single table detected
- ✅ Confidence > 0.9
- ✅ Headers match file content

#### Test 1.2: Multi-Sheet Excel
```python
async def test_multi_sheet_excel():
    """Test Excel file with multiple sheets."""
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_file=0.10
    )

    gp = GridPorter(config)
    result = await gp.extract_from_file("tests/manual/level0/test_multi_sheet.xlsx")

    print(f"Sheets processed: {len(result.sheets)}")
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        print(f"  Tables: {len(sheet.tables)}")
        for table in sheet.tables:
            print(f"    {table.range.excel_range} (confidence: {table.confidence:.2f})")

asyncio.run(test_multi_sheet_excel())
```

### 2. Complex Detection Tests

#### Test 2.1: Multiple Tables with Gaps
```python
async def test_multiple_tables():
    """Test detection of multiple tables with gaps."""
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        confidence_threshold=0.8,
        enable_region_verification=True
    )

    gp = GridPorter(config)

    # Use VisionOrchestratorAgent directly for detailed analysis
    from gridporter.agents import VisionOrchestratorAgent
    from gridporter.readers import create_reader

    orchestrator = VisionOrchestratorAgent(config)

    reader = create_reader("tests/manual/level2/creative_tables.xlsx")
    sheets = list(reader.read_sheets())

    for sheet_data in sheets:
        result = await orchestrator.orchestrate_detection(sheet_data)

        print(f"\nSheet: {sheet_data.name}")
        print(f"Complexity Score: {result.complexity_assessment.complexity_score:.3f}")
        print(f"Strategy Used: {result.orchestrator_decision.detection_strategy}")
        print(f"Vision Used: {result.orchestrator_decision.use_vision}")
        print(f"Tables Detected: {len(result.tables)}")

        for i, table in enumerate(result.tables):
            print(f"\n  Table {i+1}:")
            print(f"    Range: {table.range.excel_range}")
            print(f"    Method: {table.detection_method}")
            print(f"    Confidence: {table.confidence:.3f}")

asyncio.run(test_multiple_tables())
```

#### Test 2.2: Sparse Data Patterns
```python
async def test_sparse_patterns():
    """Test detection in files with sparse data patterns."""
    from gridporter.vision.pattern_detector import SparsePatternDetector
    from gridporter.readers import create_reader

    reader = create_reader("tests/manual/level1/large_table.xlsx")
    sheet_data = list(reader.read_sheets())[0]

    detector = SparsePatternDetector()
    patterns = detector.detect_patterns(sheet_data)

    print(f"Patterns detected: {len(patterns)}")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  Bounds: {pattern.bounds.min_row},{pattern.bounds.min_col} to "
              f"{pattern.bounds.max_row},{pattern.bounds.max_col}")
        print(f"  Density: {pattern.density:.3f}")
        print(f"  Cell count: {pattern.cell_count}")

asyncio.run(test_sparse_patterns())
```

### 3. Cost Optimization Tests

#### Test 3.1: Budget Constraints
```python
async def test_budget_constraints():
    """Test behavior under tight budget constraints."""
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_session=0.05,  # Very low budget
        max_cost_per_file=0.01
    )

    gp = GridPorter(config)

    test_files = [
        "tests/manual/level0/test_basic.xlsx",
        "tests/manual/level1/complex_table.xlsx",
        "tests/manual/level2/weird_tables.xlsx"
    ]

    total_cost = 0
    for file_path in test_files:
        try:
            result = await gp.extract_from_file(file_path)

            # Check cost tracking
            cost_report = gp.get_cost_report()
            file_cost = cost_report.get("last_file_cost", 0)
            total_cost = cost_report.get("total_cost_usd", 0)

            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"  Tables found: {len(result.tables)}")
            print(f"  File cost: ${file_cost:.4f}")
            print(f"  Total cost: ${total_cost:.4f}")
            print(f"  Strategy: {result.metadata.get('detection_strategy', 'unknown')}")

            if total_cost >= 0.05:
                print("  ⚠️ Budget exhausted - remaining files will use free methods")

        except Exception as e:
            print(f"  ❌ Error: {e}")

asyncio.run(test_budget_constraints())
```

### 4. Performance Tests

#### Test 4.1: Large File Processing
```python
import time

async def test_large_file_performance():
    """Test performance on large files."""
    config = Config(
        use_vision=False,  # Test traditional methods for speed
        confidence_threshold=0.7
    )

    gp = GridPorter(config)

    # Create or use a large test file
    large_file = "tests/manual/level1/large_table.csv"

    start_time = time.time()
    result = await gp.extract_from_file(large_file)
    end_time = time.time()

    processing_time = end_time - start_time

    print(f"File: {large_file}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Tables detected: {len(result.tables)}")

    if result.tables:
        table = result.tables[0]
        print(f"Table size: {table.range.row_count} x {table.range.col_count}")
        print(f"Cells per second: {(table.range.row_count * table.range.col_count) / processing_time:.0f}")

asyncio.run(test_large_file_performance())
```

#### Test 4.2: Multi-Scale Vision Performance
```python
async def test_vision_performance():
    """Test multi-scale vision processing performance."""
    from gridporter.vision.vision_request_builder import VisionRequestBuilder
    from gridporter.readers import create_reader
    import time

    reader = create_reader("tests/manual/level1/complex_table.xlsx")
    sheet_data = list(reader.read_sheets())[0]

    builder = VisionRequestBuilder()

    start_time = time.time()
    request = builder.build_request(sheet_data, sheet_data.name)
    build_time = time.time() - start_time

    print(f"Vision Request Build Time: {build_time:.3f}s")
    print(f"Images generated: {request.total_images}")
    print(f"Total size: {request.total_size_mb:.2f} MB")
    print(f"Strategy: {request.prompt_template}")

    for img in request.images:
        print(f"\n  Image: {img.image_id}")
        print(f"    Compression: {img.compression_level}")
        print(f"    Coverage: {img.covers_cells}")
        print(f"    Size: {img.size_mb:.3f} MB")

asyncio.run(test_vision_performance())
```

### 5. Edge Case Tests

#### Test 5.1: Empty Files
```python
async def test_empty_files():
    """Test handling of empty or nearly empty files."""
    config = Config(use_vision=False)
    gp = GridPorter(config)

    # Create an empty CSV
    with open("test_empty.csv", "w") as f:
        f.write("")

    try:
        result = await gp.extract_from_file("test_empty.csv")
        print(f"Empty file - Tables found: {len(result.tables)}")
        assert len(result.tables) == 0
    finally:
        os.remove("test_empty.csv")

asyncio.run(test_empty_files())
```

#### Test 5.2: Malformed Data
```python
async def test_malformed_data():
    """Test handling of malformed data."""
    config = Config(use_vision=False)
    gp = GridPorter(config)

    # Create malformed CSV
    with open("test_malformed.csv", "w") as f:
        f.write("Col1,Col2,Col3\n")
        f.write("Data1,Data2\n")  # Missing column
        f.write("Data3,Data4,Data5,ExtraData\n")  # Extra column

    try:
        result = await gp.extract_from_file("test_malformed.csv")
        print(f"Malformed file - Tables found: {len(result.tables)}")
        if result.tables:
            table = result.tables[0]
            print(f"  Columns detected: {table.range.col_count}")
    finally:
        os.remove("test_malformed.csv")

asyncio.run(test_malformed_data())
```

### 6. Model Comparison Tests

#### Test 6.1: OpenAI vs Ollama
```python
async def test_model_comparison():
    """Compare results between OpenAI and Ollama models."""
    test_file = "tests/manual/level1/complex_table.xlsx"

    # Test with OpenAI
    config_openai = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model="gpt-4o"
    )

    gp_openai = GridPorter(config_openai)
    result_openai = await gp_openai.extract_from_file(test_file)

    # Test with Ollama
    config_ollama = Config(
        use_vision=True,
        use_local_llm=True,
        ollama_url="http://localhost:11434",
        ollama_vision_model="qwen2.5-vl:7b"
    )

    gp_ollama = GridPorter(config_ollama)
    result_ollama = await gp_ollama.extract_from_file(test_file)

    print("Model Comparison Results:")
    print(f"\nOpenAI GPT-4o:")
    print(f"  Tables found: {len(result_openai.tables)}")
    print(f"  Avg confidence: {sum(t.confidence for t in result_openai.tables) / len(result_openai.tables):.3f}")

    print(f"\nOllama qwen2.5-vl:")
    print(f"  Tables found: {len(result_ollama.tables)}")
    print(f"  Avg confidence: {sum(t.confidence for t in result_ollama.tables) / len(result_ollama.tables):.3f}")

asyncio.run(test_model_comparison())
```

## Validation Checklist

### ✅ Basic Functionality
- [ ] Single table CSV files detected correctly
- [ ] Multi-sheet Excel files processed
- [ ] Headers identified accurately
- [ ] Data types preserved
- [ ] Empty cells handled properly

### ✅ Complex Detection
- [ ] Multiple tables with gaps detected
- [ ] Sparse data patterns handled
- [ ] Merged cells processed correctly
- [ ] Hierarchical headers preserved
- [ ] Format-based detection working

### ✅ Cost Management
- [ ] Budget constraints respected
- [ ] Fallback to free methods works
- [ ] Cost tracking accurate
- [ ] Session limits enforced
- [ ] Per-file limits working

### ✅ Performance
- [ ] Large files process in reasonable time
- [ ] Memory usage stays within limits
- [ ] Multi-scale images generated efficiently
- [ ] Compression reduces file sizes
- [ ] No performance regressions

### ✅ Edge Cases
- [ ] Empty files handled gracefully
- [ ] Malformed data doesn't crash
- [ ] Unicode characters preserved
- [ ] Large numbers/dates formatted correctly
- [ ] Formula cells detected

### ✅ Integration
- [ ] All detection methods accessible
- [ ] Vision models integrate properly
- [ ] Cost optimizer makes correct decisions
- [ ] Telemetry captures metrics
- [ ] Error messages helpful

## Output Verification

### 1. Check Detection Accuracy
```python
def verify_detection_accuracy(result, expected_tables):
    """Verify detection matches expectations."""
    assert len(result.tables) == expected_tables

    for table in result.tables:
        # Check minimum size
        assert table.range.row_count >= 2  # At least header + 1 data row
        assert table.range.col_count >= 1

        # Check confidence
        assert table.confidence >= 0.5

        # Check detection method is valid
        valid_methods = [
            "simple_case", "island_detection", "excel_metadata",
            "vision_basic", "vision_full", "hybrid"
        ]
        assert any(method in table.detection_method for method in valid_methods)
```

### 2. Verify Cost Tracking
```python
def verify_cost_tracking(gp, max_expected_cost):
    """Verify cost tracking is working."""
    cost_report = gp.get_cost_report()

    assert "total_cost_usd" in cost_report
    assert "total_api_calls" in cost_report
    assert "method_usage" in cost_report

    total_cost = cost_report["total_cost_usd"]
    assert 0 <= total_cost <= max_expected_cost

    print(f"Total cost: ${total_cost:.4f}")
    print(f"API calls: {cost_report['total_api_calls']}")
    print(f"Methods used: {cost_report['method_usage']}")
```

## Troubleshooting

### Common Issues

1. **"Vision model not available"**
   - Check API key: `echo $OPENAI_API_KEY`
   - Verify network connectivity
   - Check Ollama is running: `curl http://localhost:11434/api/tags`

2. **"Budget exhausted" errors**
   - Increase budget limits in config
   - Use `use_vision=False` for testing
   - Clear session with new GridPorter instance

3. **Slow performance**
   - Disable vision for large files
   - Use compression for vision requests
   - Check network latency to API endpoints

4. **Incorrect table detection**
   - Verify file format matches content
   - Check for hidden rows/columns
   - Enable debug logging: `log_level="DEBUG"`

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

config = Config(
    log_level="DEBUG",
    enable_debug=True
)
```

## Reporting Issues

When reporting issues, include:

1. **Test file** (or minimal reproduction)
2. **Configuration used**
3. **Expected vs actual results**
4. **Error messages/stack traces**
5. **Debug logs** (if available)

## Conclusion

This manual testing guide covers the essential scenarios for validating GridPorter functionality. Regular testing ensures the framework maintains high quality and reliability across diverse spreadsheet formats and use cases.

For automated testing, see the [pytest test suite](../../tests/) and [CI/CD configuration](../../.github/workflows/).
