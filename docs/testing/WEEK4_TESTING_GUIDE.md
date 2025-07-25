# Week 4 Testing Guide: Region Verification & Geometry Analysis

This guide covers comprehensive testing for the Week 4 region verification system implemented in GridPorter. The system validates AI-proposed table regions using geometric analysis and provides feedback for refinement.

## Overview

Week 4 introduces a region verification system that:
- Validates AI vision model proposals using local geometry analysis
- Computes metrics like rectangularness, filledness, density, and contiguity
- Provides specific feedback for failed regions
- Enables iterative refinement through a feedback loop

## Prerequisites

1. **Completed Week 3 Setup**:
   - Vision infrastructure (bitmap generation, vision models)
   - Either OpenAI API key or Ollama with vision model

2. **Install Dependencies**:
   ```bash
   pip install numpy pillow
   ```

3. **Test Files**: Use files from `tests/manual/` or create test data

## Test Cases

### Section 1: Basic Region Verification

#### Test 1.1: Verify Valid Region
```python
from gridporter.vision.region_verifier import RegionVerifier, GeometryMetrics
from gridporter.models.sheet_data import SheetData, CellData
from gridporter.models.table import TableRange

# Create a simple valid table
sheet = SheetData(name="ValidTable")
# Add headers
sheet.cells["A1"] = CellData(value="Name", data_type="text", is_bold=True)
sheet.cells["B1"] = CellData(value="Age", data_type="text", is_bold=True)
sheet.cells["C1"] = CellData(value="City", data_type="text", is_bold=True)
# Add data rows
sheet.cells["A2"] = CellData(value="Alice", data_type="text")
sheet.cells["B2"] = CellData(value=25, data_type="number")
sheet.cells["C2"] = CellData(value="NYC", data_type="text")
sheet.cells["A3"] = CellData(value="Bob", data_type="text")
sheet.cells["B3"] = CellData(value=30, data_type="number")
sheet.cells["C3"] = CellData(value="LA", data_type="text")
sheet.max_row = 2
sheet.max_column = 2

# Define region covering the table
region = TableRange(
    range="A1:C3",
    start_row=0,
    start_col=0,
    end_row=2,
    end_col=2
)

# Verify the region
verifier = RegionVerifier()
result = verifier.verify_region(sheet, region)

print(f"Valid: {result.valid}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reason: {result.reason}")
print("Metrics:")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value:.2f}")
```

**Expected Output**:
```
Valid: True
Confidence: 0.95
Reason: All checks passed
Metrics:
  rectangularness: 1.00
  filledness: 1.00
  density: 1.00
  contiguity: 1.00
  edge_quality: 1.00
  aspect_ratio: 1.00
  overall_score: 1.00
```

#### Test 1.2: Verify Sparse Region
```python
# Create a sparse table
sparse_sheet = SheetData(name="SparseTable")
# Scattered data points
sparse_sheet.cells["A1"] = CellData(value="A", data_type="text")
sparse_sheet.cells["E1"] = CellData(value="E", data_type="text")
sparse_sheet.cells["C3"] = CellData(value="X", data_type="text")
sparse_sheet.cells["A5"] = CellData(value="F", data_type="text")
sparse_sheet.cells["E5"] = CellData(value="J", data_type="text")
sparse_sheet.max_row = 4
sparse_sheet.max_column = 4

# Define region covering sparse area
sparse_region = TableRange(
    range="A1:E5",
    start_row=0,
    start_col=0,
    end_row=4,
    end_col=4
)

# Verify with custom thresholds
verifier = RegionVerifier(min_filledness=0.3)
result = verifier.verify_region(sparse_sheet, sparse_region)

print(f"Valid: {result.valid}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reason: {result.reason}")
if result.feedback:
    print(f"Feedback: {result.feedback}")
```

**Expected Output**:
```
Valid: False
Confidence: 0.08
Reason: Low filledness (0.20 < 0.3); Low rectangularness (0.20 < 0.7); Low contiguity (0.04 < 0.5)
Feedback: Region is too sparse (20.0% filled); Data distribution is not rectangular enough; Data is too fragmented or disconnected
```

#### Test 1.3: Verify Invalid Bounds
```python
# Test with bounds exceeding sheet dimensions
oversized_region = TableRange(
    range="A1:Z100",
    start_row=0,
    start_col=0,
    end_row=99,
    end_col=25
)

result = verifier.verify_region(sheet, oversized_region)
print(f"Valid: {result.valid}")
print(f"Reason: {result.reason}")
print(f"Feedback: {result.feedback}")
```

**Expected Output**:
```
Valid: False
Reason: Invalid bounds
Feedback: Region bounds exceed sheet dimensions
```

### Section 2: Geometry Metrics Analysis

#### Test 2.1: Rectangularness Testing
```python
# Create L-shaped data (non-rectangular)
l_shape_sheet = SheetData(name="LShape")
# Top horizontal part
for col in range(5):
    addr = f"{chr(65 + col)}1"
    l_shape_sheet.cells[addr] = CellData(value=f"H{col}", data_type="text")
# Left vertical part
for row in range(1, 5):
    l_shape_sheet.cells[f"A{row+1}"] = CellData(value=f"V{row}", data_type="text")
l_shape_sheet.max_row = 4
l_shape_sheet.max_column = 4

# Test full region
l_region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)

verifier = RegionVerifier(min_rectangularness=0.8)
result = verifier.verify_region(l_shape_sheet, l_region)

print(f"Valid: {result.valid}")
print(f"Rectangularness: {result.metrics['rectangularness']:.2f}")
print(f"Reason: {result.reason}")
```

**Expected Output**:
```
Valid: False
Rectangularness: 0.56
Reason: Low rectangularness (0.56 < 0.8)
```

#### Test 2.2: Contiguity Testing
```python
# Create fragmented data (multiple islands)
fragmented_sheet = SheetData(name="Fragmented")
# Island 1 (top-left)
fragmented_sheet.cells["A1"] = CellData(value="I1A", data_type="text")
fragmented_sheet.cells["B1"] = CellData(value="I1B", data_type="text")
fragmented_sheet.cells["A2"] = CellData(value="I1C", data_type="text")
fragmented_sheet.cells["B2"] = CellData(value="I1D", data_type="text")
# Island 2 (bottom-right)
fragmented_sheet.cells["D4"] = CellData(value="I2A", data_type="text")
fragmented_sheet.cells["E4"] = CellData(value="I2B", data_type="text")
fragmented_sheet.cells["D5"] = CellData(value="I2C", data_type="text")
fragmented_sheet.cells["E5"] = CellData(value="I2D", data_type="text")
fragmented_sheet.max_row = 4
fragmented_sheet.max_column = 4

region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)
verifier = RegionVerifier(min_contiguity=0.8)
result = verifier.verify_region(fragmented_sheet, region)

print(f"Valid: {result.valid}")
print(f"Contiguity: {result.metrics['contiguity']:.2f}")
print(f"Filledness: {result.metrics['filledness']:.2f}")
```

**Expected Output**:
```
Valid: False
Contiguity: 0.50
Filledness: 0.32
```

#### Test 2.3: Edge Quality Testing
```python
# Create table with ragged edges
ragged_sheet = SheetData(name="Ragged")
# Row 1: Full
for col in range(5):
    ragged_sheet.cells[f"{chr(65+col)}1"] = CellData(value=f"R1C{col}", data_type="text")
# Row 2: Missing middle
ragged_sheet.cells["A2"] = CellData(value="R2C0", data_type="text")
ragged_sheet.cells["E2"] = CellData(value="R2C4", data_type="text")
# Row 3: Full
for col in range(5):
    ragged_sheet.cells[f"{chr(65+col)}3"] = CellData(value=f"R3C{col}", data_type="text")
ragged_sheet.max_row = 2
ragged_sheet.max_column = 4

region = TableRange(range="A1:E3", start_row=0, start_col=0, end_row=2, end_col=4)
verifier = RegionVerifier()
result = verifier.verify_region(ragged_sheet, region)

print(f"Edge quality: {result.metrics['edge_quality']:.2f}")
print(f"Overall score: {result.metrics['overall_score']:.2f}")
```

### Section 3: Pattern-Specific Verification

#### Test 3.1: Header-Data Pattern
```python
from gridporter.vision.pattern_detector import TablePattern, TableBounds, PatternType

# Create table with clear headers
header_sheet = SheetData(name="HeaderData")
# Headers (bold)
headers = ["Product", "Price", "Stock", "Category"]
for i, header in enumerate(headers):
    addr = f"{chr(65+i)}1"
    header_sheet.cells[addr] = CellData(value=header, data_type="text", is_bold=True)
# Data rows
data = [
    ["Apple", 1.25, 100, "Fruit"],
    ["Banana", 0.75, 150, "Fruit"],
    ["Carrot", 0.50, 200, "Vegetable"]
]
for row_idx, row_data in enumerate(data):
    for col_idx, value in enumerate(row_data):
        addr = f"{chr(65+col_idx)}{row_idx+2}"
        header_sheet.cells[addr] = CellData(
            value=value,
            data_type="number" if isinstance(value, (int, float)) else "text"
        )
header_sheet.max_row = 3
header_sheet.max_column = 3

# Create pattern object
pattern = TablePattern(
    pattern_type=PatternType.HEADER_DATA,
    bounds=TableBounds(start_row=0, start_col=0, end_row=3, end_col=3),
    confidence=0.9,
    characteristics={}
)
# Add required properties
pattern.start_row = 0
pattern.start_col = 0
pattern.end_row = 3
pattern.end_col = 3
pattern.range = "A1:D4"

verifier = RegionVerifier()
result = verifier.verify_pattern(header_sheet, pattern)

print(f"Valid: {result.valid}")
print(f"Pattern verification: {result.reason}")
```

**Expected Output**:
```
Valid: True
Pattern verification: All checks passed
```

#### Test 3.2: Matrix Pattern
```python
# Create cross-reference matrix
matrix_sheet = SheetData(name="Matrix")
# Top-left empty
matrix_sheet.cells["A1"] = CellData(value="", data_type="text")
# Column headers
quarters = ["Q1", "Q2", "Q3", "Q4"]
for i, q in enumerate(quarters):
    matrix_sheet.cells[f"{chr(66+i)}1"] = CellData(value=q, data_type="text", is_bold=True)
# Row headers
products = ["Product A", "Product B", "Product C"]
for i, p in enumerate(products):
    matrix_sheet.cells[f"A{i+2}"] = CellData(value=p, data_type="text", is_bold=True)
# Data cells
for row in range(3):
    for col in range(4):
        value = (row + 1) * (col + 1) * 100
        addr = f"{chr(66+col)}{row+2}"
        matrix_sheet.cells[addr] = CellData(value=value, data_type="number")
matrix_sheet.max_row = 3
matrix_sheet.max_column = 4

# Test matrix pattern verification
from gridporter.vision.region_verifier import RegionVerifier

verifier = RegionVerifier()
# Direct call to matrix verification
result = verifier._verify_matrix_pattern(
    matrix_sheet,
    TablePattern(
        pattern_type=PatternType.MATRIX,
        bounds=TableBounds(start_row=0, start_col=0, end_row=3, end_col=4),
        confidence=0.9,
        characteristics={}
    )
)

print(f"Valid: {result.valid}")
print(f"Has row headers: {result.metrics.get('has_row_headers', False)}")
print(f"Has column headers: {result.metrics.get('has_col_headers', False)}")
```

### Section 4: Feedback Loop Testing

#### Test 4.1: Generate Feedback for Failed Regions
```python
# Create extremely wide sparse region
wide_sheet = SheetData(name="WideSheet")
# Only first and last columns have data
wide_sheet.cells["A1"] = CellData(value="Start", data_type="text")
wide_sheet.cells["CV1"] = CellData(value="End", data_type="text")
wide_sheet.max_row = 0
wide_sheet.max_column = 99  # 100 columns

wide_region = TableRange(range="A1:CV1", start_row=0, start_col=0, end_row=0, end_col=99)
verifier = RegionVerifier()
result = verifier.verify_region(wide_sheet, wide_region)

print(f"Valid: {result.valid}")
print(f"Feedback: {result.feedback}")
print("\nDetailed metrics:")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value}")
```

**Expected Output**:
```
Valid: False
Feedback: Region contains too little data to be a meaningful table

Detailed metrics:
  filled_cells: 2.0
```

#### Test 4.2: Test Feedback Loop Integration
```python
import asyncio
from gridporter.vision.pipeline import VisionPipeline
from gridporter.models.vision_result import VisionRegion
from gridporter.config import Config

async def test_feedback_loop():
    # Configure pipeline
    config = Config(
        enable_region_verification=True,
        enable_verification_feedback=True,
        min_region_filledness=0.5,
        min_region_rectangularness=0.7
    )

    pipeline = VisionPipeline(config)

    # Create a failed region (sparse)
    failed_region = VisionRegion(
        pixel_bounds={"x1": 0, "y1": 0, "x2": 500, "y2": 100},
        cell_bounds={"start_row": 0, "start_col": 0, "end_row": 9, "end_col": 49},
        range="A1:AX10",
        confidence=0.8,
        characteristics={"sparse": True}
    )

    # Mock sheet data
    sheet = SheetData(name="TestSheet")
    # Add some scattered data
    sheet.cells["A1"] = CellData(value="Data1", data_type="text")
    sheet.cells["AX10"] = CellData(value="Data2", data_type="text")
    sheet.max_row = 9
    sheet.max_column = 49

    # Test refinement (will use mock if no real vision model)
    try:
        refined_result = await pipeline.refine_with_feedback(
            sheet, [failed_region], max_iterations=1
        )
        print(f"Refined regions: {len(refined_result.regions)}")
        for region in refined_result.regions:
            print(f"  - {region.range}, confidence: {region.confidence:.2f}")
    except Exception as e:
        print(f"Feedback loop test (expected without real vision model): {type(e).__name__}")

# Run async test
asyncio.run(test_feedback_loop())
```

### Section 5: Integration with Vision Pipeline

#### Test 5.1: Integrated Pipeline with Verification
```python
from gridporter.vision.integrated_pipeline import IntegratedVisionPipeline
from gridporter.config import Config

# Configure with verification enabled
config = Config(
    enable_region_verification=True,
    verification_strict_mode=False,
    min_region_filledness=0.2,
    min_region_rectangularness=0.6,
    min_region_contiguity=0.4
)

# Create pipeline from config
pipeline = IntegratedVisionPipeline.from_config(config)

# Create test sheet with multiple patterns
mixed_sheet = SheetData(name="MixedPatterns")
# Valid table region
for row in range(3):
    for col in range(3):
        addr = f"{chr(65+col)}{row+1}"
        mixed_sheet.cells[addr] = CellData(value=f"T1_{row}_{col}", data_type="text")
# Sparse region (should be filtered)
mixed_sheet.cells["F1"] = CellData(value="Sparse1", data_type="text")
mixed_sheet.cells["J5"] = CellData(value="Sparse2", data_type="text")
mixed_sheet.max_row = 4
mixed_sheet.max_column = 9

# Process sheet
result = pipeline.process_sheet(mixed_sheet)

print(f"Detected tables: {len(result.detected_tables)}")
print(f"Verification enabled: {pipeline.enable_verification}")
if result.verification_results:
    print("\nVerification results:")
    for pattern_id, verification in result.verification_results.items():
        print(f"  Pattern {pattern_id}: valid={verification.valid}, "
              f"confidence={verification.confidence:.2f}")
```

#### Test 5.2: Verification Metrics in Pattern
```python
# Check that verification metrics are added to patterns
for pattern in result.detected_tables:
    print(f"\nPattern at {pattern.range}:")
    print(f"  Type: {pattern.pattern_type}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    if "verification_score" in pattern.characteristics:
        print(f"  Verification score: {pattern.characteristics['verification_score']:.2f}")
    if "verification_metrics" in pattern.characteristics:
        metrics = pattern.characteristics['verification_metrics']
        print("  Geometry metrics:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.2f}")
```

### Section 6: Configuration Testing

#### Test 6.1: Custom Verification Thresholds
```python
# Test with different threshold configurations
configs = [
    {
        "name": "Strict",
        "min_filledness": 0.5,
        "min_rectangularness": 0.9,
        "min_contiguity": 0.8
    },
    {
        "name": "Moderate",
        "min_filledness": 0.3,
        "min_rectangularness": 0.7,
        "min_contiguity": 0.5
    },
    {
        "name": "Lenient",
        "min_filledness": 0.1,
        "min_rectangularness": 0.5,
        "min_contiguity": 0.3
    }
]

# Test sheet with moderate sparsity
test_sheet = SheetData(name="TestConfig")
# Create a 5x5 grid with 60% filled
for row in range(5):
    for col in range(5):
        if (row + col) % 2 == 0:  # Checkerboard pattern
            addr = f"{chr(65+col)}{row+1}"
            test_sheet.cells[addr] = CellData(value=f"D{row}{col}", data_type="text")
test_sheet.max_row = 4
test_sheet.max_column = 4

region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)

for config in configs:
    verifier = RegionVerifier(
        min_filledness=config["min_filledness"],
        min_rectangularness=config["min_rectangularness"],
        min_contiguity=config["min_contiguity"]
    )
    result = verifier.verify_region(test_sheet, region)
    print(f"\n{config['name']} configuration:")
    print(f"  Valid: {result.valid}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Filledness: {result.metrics['filledness']:.2f}")
```

#### Test 6.2: Strict Mode Testing
```python
# Test strict vs non-strict mode
slightly_irregular = SheetData(name="SlightlyIrregular")
# Mostly filled table with one missing corner
for row in range(4):
    for col in range(4):
        if not (row == 3 and col == 3):  # Missing bottom-right
            addr = f"{chr(65+col)}{row+1}"
            slightly_irregular.cells[addr] = CellData(value=f"D{row}{col}", data_type="text")
slightly_irregular.max_row = 3
slightly_irregular.max_column = 3

region = TableRange(range="A1:D4", start_row=0, start_col=0, end_row=3, end_col=3)

# Non-strict mode
verifier_lenient = RegionVerifier()
result_lenient = verifier.verify_region(slightly_irregular, region, strict=False)
print(f"Non-strict mode: valid={result_lenient.valid}, confidence={result_lenient.confidence:.2f}")

# Strict mode
result_strict = verifier.verify_region(slightly_irregular, region, strict=True)
print(f"Strict mode: valid={result_strict.valid}, confidence={result_strict.confidence:.2f}")
```

### Section 7: Performance and Edge Cases

#### Test 7.1: Large Region Performance
```python
import time

# Create a large sheet
large_sheet = SheetData(name="LargeSheet")
# Fill 100x100 region
print("Creating large sheet...")
start_time = time.time()
for row in range(100):
    for col in range(100):
        # Create address (handle columns > 26)
        col_letter = ""
        c = col
        while c >= 0:
            col_letter = chr(c % 26 + ord('A')) + col_letter
            c = c // 26 - 1
            if c < 0:
                break
        addr = f"{col_letter}{row+1}"
        large_sheet.cells[addr] = CellData(value=f"V{row},{col}", data_type="text")
large_sheet.max_row = 99
large_sheet.max_column = 99
creation_time = time.time() - start_time
print(f"Sheet creation time: {creation_time:.2f}s")

# Time verification
large_region = TableRange(range="A1:CV100", start_row=0, start_col=0, end_row=99, end_col=99)
verifier = RegionVerifier()

start_time = time.time()
result = verifier.verify_region(large_sheet, large_region)
verification_time = time.time() - start_time

print(f"Verification time: {verification_time:.3f}s")
print(f"Valid: {result.valid}")
print(f"Cells processed: 10,000")
print(f"Processing rate: {10000/verification_time:.0f} cells/second")
```

#### Test 7.2: Edge Cases
```python
# Test various edge cases

# 1. Single cell
single_cell_sheet = SheetData(name="SingleCell")
single_cell_sheet.cells["A1"] = CellData(value="Only", data_type="text")
single_cell_sheet.max_row = 0
single_cell_sheet.max_column = 0

result = verifier.verify_region(
    single_cell_sheet,
    TableRange(range="A1", start_row=0, start_col=0, end_row=0, end_col=0)
)
print(f"Single cell: valid={result.valid}, reason='{result.reason}'")

# 2. Empty region
empty_sheet = SheetData(name="Empty")
empty_sheet.max_row = 0
empty_sheet.max_column = 0

result = verifier.verify_region(
    empty_sheet,
    TableRange(range="A1:C3", start_row=0, start_col=0, end_row=2, end_col=2)
)
print(f"Empty region: valid={result.valid}, reason='{result.reason}'")

# 3. Single row
single_row_sheet = SheetData(name="SingleRow")
for col in range(10):
    single_row_sheet.cells[f"{chr(65+col)}1"] = CellData(value=f"C{col}", data_type="text")
single_row_sheet.max_row = 0
single_row_sheet.max_column = 9

result = verifier.verify_region(
    single_row_sheet,
    TableRange(range="A1:J1", start_row=0, start_col=0, end_row=0, end_col=9)
)
print(f"Single row: valid={result.valid}, aspect_ratio={result.metrics['aspect_ratio']:.1f}")

# 4. Single column
single_col_sheet = SheetData(name="SingleCol")
for row in range(10):
    single_col_sheet.cells[f"A{row+1}"] = CellData(value=f"R{row}", data_type="text")
single_col_sheet.max_row = 9
single_col_sheet.max_column = 0

result = verifier.verify_region(
    single_col_sheet,
    TableRange(range="A1:A10", start_row=0, start_col=0, end_row=9, end_col=0)
)
print(f"Single column: valid={result.valid}, aspect_ratio={result.metrics['aspect_ratio']:.1f}")
```

### Section 8: Visual Debugging

#### Test 8.1: Visualize Verification Results
```python
def visualize_verification(sheet: SheetData, region: TableRange, result: VerificationResult):
    """Create a simple text visualization of verification results."""
    print(f"\nRegion {region.range} Verification:")
    print("=" * 50)

    # Create grid visualization
    for row in range(region.start_row, region.end_row + 1):
        row_str = f"Row {row+1:2d}: "
        for col in range(region.start_col, region.end_col + 1):
            cell = sheet.get_cell(row, col)
            if cell and cell.value is not None:
                row_str += "[X] "
            else:
                row_str += "[ ] "
        print(row_str)

    print("\nMetrics Summary:")
    print(f"  Valid: {'✓' if result.valid else '✗'} ({result.confidence:.2%} confidence)")
    print(f"  Rectangularness: {'▮' * int(result.metrics['rectangularness'] * 10)}")
    print(f"  Filledness:      {'▮' * int(result.metrics['filledness'] * 10)}")
    print(f"  Contiguity:      {'▮' * int(result.metrics['contiguity'] * 10)}")
    print(f"  Edge Quality:    {'▮' * int(result.metrics['edge_quality'] * 10)}")

    if result.feedback:
        print(f"\nFeedback: {result.feedback}")

# Test with sample data
viz_sheet = SheetData(name="Visualization")
# Create an L-shape
for col in range(5):
    viz_sheet.cells[f"{chr(65+col)}1"] = CellData(value="H", data_type="text")
for row in range(1, 5):
    viz_sheet.cells[f"A{row+1}"] = CellData(value="V", data_type="text")
viz_sheet.max_row = 4
viz_sheet.max_column = 4

region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)
result = verifier.verify_region(viz_sheet, region)
visualize_verification(viz_sheet, region, result)
```

#### Test 8.2: Compare Before/After Verification
```python
# Show how verification filters proposals
def compare_proposals(original_regions, verified_regions):
    """Compare original vs verified regions."""
    print("\nRegion Verification Comparison:")
    print("=" * 60)
    print(f"Original regions: {len(original_regions)}")
    print(f"Verified regions: {len(verified_regions)}")
    print(f"Filtered out: {len(original_regions) - len(verified_regions)}")

    # Show details
    print("\nOriginal Regions:")
    for i, region in enumerate(original_regions):
        print(f"  {i+1}. {region.range} (confidence: {region.confidence:.2f})")

    print("\nVerified Regions:")
    for i, region in enumerate(verified_regions):
        print(f"  {i+1}. {region.range} (confidence: {region.confidence:.2f})")

# Mock example
from gridporter.models.vision_result import VisionRegion

original = [
    VisionRegion(
        pixel_bounds={"x1": 0, "y1": 0, "x2": 100, "y2": 50},
        cell_bounds={"start_row": 0, "start_col": 0, "end_row": 4, "end_col": 9},
        range="A1:J5",
        confidence=0.9,
        characteristics={"type": "valid_table"}
    ),
    VisionRegion(
        pixel_bounds={"x1": 0, "y1": 100, "x2": 500, "y2": 150},
        cell_bounds={"start_row": 10, "start_col": 0, "end_row": 14, "end_col": 49},
        range="A11:AX15",
        confidence=0.8,
        characteristics={"type": "sparse_region"}
    )
]

# Simulate verification filtering
verified = [original[0]]  # Only first region passes
compare_proposals(original, verified)
```

## Troubleshooting

### Common Issues

1. **NumPy Type Errors**:
   - Ensure all metrics are converted to Python types (not numpy)
   - Use `float()` and `bool()` conversions

2. **Sheet Data Access**:
   - Use `sheet.get_cell(row, col)` instead of direct array access
   - Check `sheet.max_row` and `sheet.max_column` for bounds

3. **Pattern Verification**:
   - TablePattern requires proper initialization with bounds
   - Add required properties (start_row, end_row, etc.) manually

4. **Performance Issues**:
   - Large regions may take time due to contiguity calculations
   - Consider sampling for very large sheets

5. **Feedback Generation**:
   - Feedback depends on failure reasons
   - Check multiple threshold violations

### Validation Checklist

✓ All geometry metrics compute correctly (0.0 to 1.0 range)
✓ Valid regions pass verification with high confidence
✓ Invalid regions generate appropriate feedback
✓ Pattern-specific verification works for different types
✓ Configuration thresholds affect results as expected
✓ Integration with pipeline filters regions correctly
✓ Performance is acceptable for typical use cases

## Performance Benchmarks

Expected performance for region verification:
- Small regions (< 100 cells): < 10ms
- Medium regions (100-1000 cells): 10-50ms
- Large regions (1000-10000 cells): 50-200ms
- Very large regions (> 10000 cells): 200ms-1s

Memory usage is minimal as verification works with numpy arrays.

## Next Steps

After completing Week 4 testing:
1. Week 5 will add semantic understanding for complex tables
2. Multi-row header detection using vision
3. Hierarchical data relationship analysis
4. Format preservation logic improvements

## Summary

The Week 4 region verification system provides crucial validation for AI-proposed regions, significantly improving the accuracy of table detection by filtering out false positives and providing actionable feedback for refinement. The geometry-based approach is fast, deterministic, and provides interpretable metrics for debugging and optimization.
