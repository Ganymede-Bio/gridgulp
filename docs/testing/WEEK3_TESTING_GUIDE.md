# Week 3 Testing Guide: Vision Infrastructure Foundation

This guide covers manual testing for the Week 3 vision-based table detection infrastructure implemented in GridPorter.

## Overview

Week 3 introduces vision-based table detection using bitmap representations of spreadsheet data. The system converts spreadsheet cells to lossless PNG images and uses AI vision models to identify table regions.

## Prerequisites

1. **Set up Vision Models** (choose one):
   - **OpenAI**: Set `OPENAI_API_KEY` environment variable
   - **Ollama**: Install Ollama and pull a vision model like `qwen2.5vl:7b`

2. **Test Files**: Use files from `tests/manual/` directory

## Test Cases

### Section 1: Bitmap Generation

#### Test 1.1: Basic Bitmap Generation
```python
from gridporter.vision import BitmapGenerator
from gridporter.readers import get_reader
from gridporter.models.sheet_data import SheetData, CellData

# Option 1: Load from file (if available)
try:
    reader = get_reader("tests/manual/simple_table.xlsx")
    sheets = reader.read_all()
    sheet = sheets[0]
except FileNotFoundError:
    # Option 2: Create test data programmatically
    sheet = SheetData(name="TestSheet")
    sheet.cells["A1"] = CellData(value="Name", data_type="text", is_bold=True)
    sheet.cells["B1"] = CellData(value="Age", data_type="text", is_bold=True)
    sheet.cells["A2"] = CellData(value="Alice", data_type="text")
    sheet.cells["B2"] = CellData(value=25, data_type="number")
    sheet.max_row = 1
    sheet.max_column = 1

# Generate bitmap
generator = BitmapGenerator()
image_bytes, metadata = generator.generate(sheet)

print(f"Generated bitmap: {metadata.width}x{metadata.height} pixels")
print(f"Cell size: {metadata.cell_width}x{metadata.cell_height}")
print(f"Sheet size: {metadata.total_rows}x{metadata.total_cols} cells")
print(f"Image size: {len(image_bytes)} bytes")
```

**Expected**: Bitmap generated successfully with correct dimensions

#### Test 1.2: Different Bitmap Modes
```python
# Use the sheet from Test 1.1
from gridporter.vision import BitmapGenerator

# Test binary mode (default)
generator_binary = BitmapGenerator(mode="binary")
img_binary, meta_binary = generator_binary.generate(sheet)

# Test grayscale mode
generator_gray = BitmapGenerator(mode="grayscale")
img_gray, meta_gray = generator_gray.generate(sheet)

# Test color mode
generator_color = BitmapGenerator(mode="color")
img_color, meta_color = generator_color.generate(sheet)

print("Binary mode:", len(img_binary), "bytes")
print("Grayscale mode:", len(img_gray), "bytes")
print("Color mode:", len(img_color), "bytes")
```

**Expected**: All three modes generate different bitmap representations

#### Test 1.3: Save Debug Bitmap
```python
from pathlib import Path

# Save bitmap to file for visual inspection
debug_path = Path("tests/manual/debug_bitmap.png")
with open(debug_path, "wb") as f:
    f.write(image_bytes)

print(f"Debug bitmap saved to {debug_path}")
# Open the file in an image viewer to verify visual representation
```

**Expected**: PNG file created that can be opened in image viewer

#### Test 1.4: Large Sheet Scaling
```python
from gridporter.models.sheet_data import SheetData, CellData

# Create a large sheet that should trigger scaling
large_sheet = SheetData(name="LargeSheet")
# Simulate a 200x200 sheet
for row in range(200):
    for col in range(200):
        if (row + col) % 10 == 0:  # Sparse data
            cell_addr = f"{chr(65 + col % 26)}{row + 1}"
            large_sheet.cells[cell_addr] = CellData(value=f"R{row}C{col}", data_type="text")
large_sheet.max_row = 199
large_sheet.max_column = 199

generator = BitmapGenerator(cell_width=20, cell_height=20)
image_bytes, metadata = generator.generate(large_sheet)

print(f"Scale factor applied: {metadata.scale_factor}")
print(f"Original cell size: 20x20, actual: {metadata.cell_width}x{metadata.cell_height}")
```

**Expected**: Scale factor < 1.0 if sheet is large, proper scaling applied

### Section 2: Vision Model Configuration

#### Test 2.1: OpenAI Vision Model (if API key available)
```python
from gridporter.config import Config
from gridporter.vision.vision_models import create_vision_model, VisionModelError

# Configure for OpenAI
config = Config(
    openai_api_key="your-api-key-here",  # Use your actual key
    use_local_llm=False
)

try:
    model = create_vision_model(config)
    print(f"Created vision model: {model.name}")
    print(f"Supports batch: {model.supports_batch}")
except VisionModelError as e:
    print(f"Vision model error: {e}")
    # Common issues: missing API key, missing openai package
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install gridporter[vision]")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Expected**: OpenAI model created if API key is valid

#### Test 2.2: Ollama Vision Model (if Ollama available)
```python
import asyncio
from gridporter.config import Config
from gridporter.vision.vision_models import create_vision_model

async def test_ollama_model():
    """Test Ollama vision model creation and availability."""
    from gridporter.vision.vision_models import VisionModelError

    # Configure for Ollama
    config = Config(
        use_local_llm=True,
        ollama_url="http://localhost:11434",
        ollama_vision_model="qwen2.5vl:7b"
    )

    try:
        model = create_vision_model(config)
        print(f"Created vision model: {model.name}")

        # Check if model is available
        if hasattr(model, 'check_model_available'):
            available = await model.check_model_available()
            print(f"Model available: {available}")
            if not available:
                print("Model not found. Try: ollama pull qwen2.5vl:7b")

    except VisionModelError as e:
        print(f"Vision model error: {e}")
        # Common issues: Ollama not running, model not pulled
    except ConnectionError as e:
        print(f"Connection error: {e}")
        print("Is Ollama running? Start with: ollama serve")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Run the async test
asyncio.run(test_ollama_model())
```

**Expected**: Ollama model created if service is running and model is available

### Section 3: Region Proposal Parsing

#### Test 3.1: JSON Response Parsing
```python
from gridporter.vision.region_proposer import RegionProposer
from gridporter.vision.bitmap_generator import BitmapMetadata

proposer = RegionProposer()
metadata = BitmapMetadata(
    width=200, height=150, cell_width=10, cell_height=10,
    total_rows=15, total_cols=20, scale_factor=1.0, mode="binary"
)

# Test structured JSON response
json_response = '''
{
    "tables": [
        {
            "bounds": {"x1": 10, "y1": 10, "x2": 100, "y2": 50},
            "confidence": 0.9,
            "characteristics": {"has_headers": true, "type": "data_table"}
        },
        {
            "bounds": {"x1": 120, "y1": 10, "x2": 180, "y2": 40},
            "confidence": 0.75,
            "characteristics": {"has_headers": false}
        }
    ]
}
'''

proposals = proposer.parse_response(json_response, metadata)
print(f"Parsed {len(proposals)} proposals from JSON")
for i, p in enumerate(proposals):
    print(f"  Table {i+1}: {p.excel_range}, confidence: {p.confidence}")
    print(f"    Characteristics: {p.characteristics}")
```

**Expected**: 2 proposals parsed with correct coordinates and metadata

#### Test 3.2: Text Response Parsing
```python
# Test unstructured text response
text_response = '''
I can see two tables in this spreadsheet:

Table 1: Located at pixel coordinates (20, 20, 90, 60) with confidence 0.85.
This appears to be a header table with bold formatting in the first row.

Table 2: Found another table at (110, 20, 170, 50) with confidence 0.7.
This one contains numeric data without clear headers.
'''

proposals = proposer.parse_response(text_response, metadata)
print(f"Parsed {len(proposals)} proposals from text")
for i, p in enumerate(proposals):
    print(f"  Table {i+1}: {p.excel_range}, confidence: {p.confidence}")
    print(f"    Has headers: {p.characteristics.get('has_headers', 'unknown')}")
```

**Expected**: 2 proposals parsed from text with extracted characteristics

#### Test 3.3: Proposal Filtering
```python
# Test filtering by confidence and size
all_proposals = proposer.parse_response(json_response, metadata)
print(f"Before filtering: {len(all_proposals)} proposals")

# Filter with different thresholds
filtered_high = proposer.filter_proposals(all_proposals, min_confidence=0.8)
print(f"High confidence (>0.8): {len(filtered_high)} proposals")

filtered_low = proposer.filter_proposals(all_proposals, min_confidence=0.5)
print(f"Low confidence (>0.5): {len(filtered_low)} proposals")

# Filter by size
filtered_size = proposer.filter_proposals(all_proposals, min_size=(3, 3))
print(f"Minimum size 3x3: {len(filtered_size)} proposals")
```

**Expected**: Different numbers of proposals based on filter criteria

### Section 4: Vision Pipeline Integration

#### Test 4.1: Complete Vision Analysis (Mock)
```python
from gridporter.vision import VisionPipeline
from gridporter.config import Config
from gridporter.models.sheet_data import SheetData, CellData

# Create test sheet data
sheet = SheetData(name="TestSheet")
sheet.cells["A1"] = CellData(value="Product", data_type="text", is_bold=True)
sheet.cells["B1"] = CellData(value="Price", data_type="text", is_bold=True)
sheet.cells["A2"] = CellData(value="Apple", data_type="text")
sheet.cells["B2"] = CellData(value=1.25, data_type="number")
sheet.max_row = 1
sheet.max_column = 1

# Configure pipeline
config = Config(
    vision_cell_width=12,
    vision_cell_height=10,
    vision_mode="grayscale",
    use_local_llm=True,  # Set False for OpenAI
    confidence_threshold=0.6
)

pipeline = VisionPipeline(config)

# Test bitmap generation only (without calling LLM)
image_bytes, bitmap_metadata = pipeline.bitmap_generator.generate(sheet)
print(f"Pipeline generated bitmap: {bitmap_metadata.width}x{bitmap_metadata.height}")

# Create mock response for testing
mock_response = '''
{
    "tables": [
        {
            "bounds": {"x1": 0, "y1": 0, "x2": 60, "y2": 30},
            "confidence": 0.9,
            "characteristics": {"has_headers": true}
        }
    ]
}
'''

proposals = pipeline.region_proposer.parse_response(mock_response, bitmap_metadata)
print(f"Mock analysis found {len(proposals)} tables")
```

**Expected**: Pipeline components work together correctly

#### Test 4.2: Debug Bitmap Saving
```python
from pathlib import Path

# Save debug bitmap using pipeline
debug_path = Path("pipeline_debug.png")
metadata = pipeline.save_debug_bitmap(sheet, debug_path)

print(f"Debug bitmap saved: {debug_path}")
print(f"Metadata: {metadata.total_rows}x{metadata.total_cols} cells")
```

**Expected**: Debug bitmap saved successfully

#### Test 4.3: Cache Operations
```python
# Test cache functionality
cache_stats = pipeline.get_cache_stats()
print(f"Cache stats: {cache_stats}")

# Clear cache
pipeline.clear_cache()
new_stats = pipeline.get_cache_stats()
print(f"After clear: {new_stats}")
```

**Expected**: Cache operations work correctly

### Section 5: Vision Models Integration

#### Test 5.1: Vision Model Factory
```python
from gridporter.vision.vision_models import create_vision_model

# Test with different configurations
configs = [
    Config(use_local_llm=True, ollama_vision_model="qwen2.5vl:7b"),
    Config(use_local_llm=False, openai_api_key="test-key"),
]

for i, config in enumerate(configs):
    try:
        model = create_vision_model(config)
        print(f"Config {i+1}: Created {model.name}")
    except Exception as e:
        print(f"Config {i+1}: Failed - {e}")
```

**Expected**: Factory creates appropriate models based on configuration

#### Test 5.2: Model Response Structure
```python
# Test response structure (if you have a working model)
from gridporter.vision.vision_models import VisionModelResponse

# Mock response for testing
response = VisionModelResponse(
    content='{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}, "confidence": 0.9}]}',
    model="test-model",
    usage={"total_tokens": 150}
)

print(f"Model: {response.model}")
print(f"Content length: {len(response.content)}")
print(f"Tokens used: {response.usage.get('total_tokens', 0)}")
```

**Expected**: Response structure contains expected fields

### Section 6: Configuration and Environment

#### Test 6.1: Vision Configuration
```python
from gridporter.config import Config

# Test vision-specific configuration
config = Config.from_env()
print(f"Vision cell width: {config.vision_cell_width}")
print(f"Vision cell height: {config.vision_cell_height}")
print(f"Vision mode: {config.vision_mode}")

# Test custom configuration
custom_config = Config(
    vision_cell_width=15,
    vision_cell_height=15,
    vision_mode="color"
)
print(f"Custom config: {custom_config.vision_cell_width}x{custom_config.vision_cell_height}")
```

**Expected**: Configuration loads with proper defaults and custom values

#### Test 6.2: Environment Variables
```bash
# Test environment variable configuration
export GRIDPORTER_VISION_CELL_WIDTH=20
export GRIDPORTER_VISION_CELL_HEIGHT=16
export GRIDPORTER_VISION_MODE=grayscale
```

```python
# Then in Python
config = Config.from_env()
print(f"From env - Width: {config.vision_cell_width}")
print(f"From env - Height: {config.vision_cell_height}")
print(f"From env - Mode: {config.vision_mode}")
```

**Expected**: Configuration picks up environment variables

### Section 7: Model Integration

#### Test 7.1: Vision Result Models
```python
from gridporter.models.vision_result import VisionRegion, VisionAnalysisResult

# Test creating vision region
region = VisionRegion(
    pixel_bounds={"x1": 0, "y1": 0, "x2": 100, "y2": 50},
    cell_bounds={"start_row": 0, "start_col": 0, "end_row": 4, "end_col": 9},
    range="A1:J5",
    confidence=0.85,
    characteristics={"has_headers": True, "type": "data_table"}
)

print(f"Region: {region.range}, confidence: {region.confidence}")
print(f"Characteristics: {region.characteristics}")

# Convert to TableRange
table_range = region.to_table_range()
print(f"TableRange: {table_range.range}")
```

**Expected**: Vision models work correctly and integrate with existing models

#### Test 7.2: Analysis Result Operations
```python
# Test analysis result
result = VisionAnalysisResult(
    regions=[region],
    bitmap_info={"width": 100, "height": 50, "mode": "binary"}
)

# Test high confidence filtering
high_conf = result.high_confidence_regions(threshold=0.8)
print(f"High confidence regions: {len(high_conf)}")

# Convert to table ranges
table_ranges = result.to_table_ranges()
print(f"Table ranges: {[tr.range for tr in table_ranges]}")
```

**Expected**: Result operations work correctly

## Troubleshooting

### Common Issues

1. **ImportError for PIL/Pillow**:
   ```bash
   pip install Pillow>=10.0.0
   ```

2. **OpenAI API Key Not Set**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Ollama Connection Failed**:
   - Check if Ollama is running: `ollama list`
   - Pull vision model: `ollama pull qwen2.5vl:7b`

4. **Memory Issues with Large Files**:
   - Vision components automatically scale down large images
   - Check scale_factor in bitmap metadata

5. **Model Response Parsing Errors**:
   - Vision models may return varying response formats
   - RegionProposer handles both JSON and text responses

### Test Results Validation

After running tests, verify:

1. **Bitmap Generation**: PNG files can be opened in image viewers
2. **Model Integration**: No import errors for vision components
3. **Configuration**: Environment variables are properly loaded
4. **Error Handling**: Graceful failures when models unavailable

## Performance Notes

- **Bitmap Generation**: Typically < 100ms for normal-sized sheets
- **Vision Analysis**: Depends on model (OpenAI: ~1-3s, Ollama: ~5-15s)
- **Memory Usage**: Images are generated in-memory and not cached by default
- **Scaling**: Large sheets automatically scaled to avoid memory issues

## Next Steps

After successful Week 3 testing:
1. Week 4 will integrate vision detection with the main detection pipeline
2. Add caching for vision analysis results
3. Implement batch processing for multiple sheets
4. Add more sophisticated region filtering algorithms
