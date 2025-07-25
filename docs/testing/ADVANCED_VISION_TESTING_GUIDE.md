# Advanced Vision Testing Guide: Large Spreadsheet Handling

This guide covers testing for the advanced vision features implemented for handling large and sparse spreadsheets, including sparse pattern detection, quadtree analysis, and enhanced bitmap generation.

## Overview

The advanced vision features enable GridPorter to handle massive spreadsheets (up to Excel's limits of 1M×16K cells) by:
- Detecting table patterns in sparse data
- Using quadtree spatial analysis for efficient region selection
- Generating compressed bitmaps that fit within GPT-4o's 20MB limit
- Preserving table structure integrity during visualization

## Prerequisites

1. **Environment Setup**:
   ```bash
   pip install gridporter
   pip install pytest pytest-asyncio pytest-benchmark  # For performance tests
   ```

2. **Test Data Creation Tools**:
   ```python
   # Install for generating large test files
   pip install faker openpyxl xlsxwriter
   ```

3. **Memory Requirements**:
   - At least 8GB RAM for testing large spreadsheets
   - 16GB+ recommended for stress testing Excel limits

## Test Utilities

### Create Test Data Helper
```python
# tests/manual/test_data_generator.py
import random
from gridporter.models.sheet_data import SheetData, CellData
from typing import Tuple, List

class TestDataGenerator:
    """Generate various test spreadsheets for advanced features."""

    @staticmethod
    def create_sparse_table(rows: int, cols: int, density: float = 0.1) -> SheetData:
        """Create a sparse table with given density."""
        sheet = SheetData(name="SparseTable")

        # Add headers
        headers = ["ID", "Name", "Value", "Status", "Date"]
        for i, header in enumerate(headers[:cols]):
            sheet.set_cell(0, i, CellData(value=header, is_bold=True))

        # Add sparse data
        for row in range(1, rows):
            for col in range(min(cols, len(headers))):
                if random.random() < density:
                    if col == 0:  # ID column
                        value = f"ID{row:04d}"
                    elif col == 1:  # Name column
                        value = f"Item_{row}"
                    elif col == 2:  # Value column
                        value = round(random.random() * 1000, 2)
                    elif col == 3:  # Status column
                        value = random.choice(["Active", "Pending", "Complete"])
                    else:  # Date column
                        value = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

                    sheet.set_cell(row, col, CellData(value=value))

        sheet.max_row = rows - 1
        sheet.max_column = min(cols, len(headers)) - 1
        return sheet

    @staticmethod
    def create_matrix_pattern(size: int = 10) -> SheetData:
        """Create a cross-reference matrix pattern."""
        sheet = SheetData(name="Matrix")

        # Row and column headers
        for i in range(1, size):
            sheet.set_cell(i, 0, CellData(value=f"Row{i}", is_bold=True))
            sheet.set_cell(0, i, CellData(value=f"Col{i}", is_bold=True))

        # Sparse matrix data
        for row in range(1, size):
            for col in range(1, size):
                if random.random() < 0.3:  # 30% filled
                    sheet.set_cell(row, col, CellData(value="X"))

        sheet.max_row = size - 1
        sheet.max_column = size - 1
        return sheet

    @staticmethod
    def create_form_pattern(items: int = 20) -> SheetData:
        """Create a form/checklist pattern."""
        sheet = SheetData(name="Form")

        # Headers
        sheet.set_cell(0, 0, CellData(value="Task", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Status", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Date", is_bold=True))
        sheet.set_cell(0, 3, CellData(value="Notes", is_bold=True))

        # Dense first column, sparse others
        for i in range(1, items + 1):
            sheet.set_cell(i, 0, CellData(value=f"Task {i}"))

            if random.random() < 0.4:  # 40% completed
                sheet.set_cell(i, 1, CellData(value="✓"))
                sheet.set_cell(i, 2, CellData(value=f"2024-01-{i:02d}"))

            if random.random() < 0.2:  # 20% have notes
                sheet.set_cell(i, 3, CellData(value="See details"))

        sheet.max_row = items
        sheet.max_column = 3
        return sheet

    @staticmethod
    def create_multiple_tables(table_count: int = 3) -> SheetData:
        """Create sheet with multiple sparse tables."""
        sheet = SheetData(name="MultipleTables")
        current_row = 0

        for table_idx in range(table_count):
            # Table header
            headers = [f"T{table_idx+1}_Col{i}" for i in range(1, 5)]
            for col_idx, header in enumerate(headers):
                sheet.set_cell(current_row, col_idx + table_idx * 6,
                             CellData(value=header, is_bold=True))

            # Table data (sparse)
            for row_offset in range(1, 10):
                for col_idx in range(4):
                    if random.random() < 0.3:
                        value = f"T{table_idx+1}_R{row_offset}_C{col_idx+1}"
                        sheet.set_cell(current_row + row_offset,
                                     col_idx + table_idx * 6,
                                     CellData(value=value))

            current_row += 12  # Space between tables

        sheet.max_row = current_row - 1
        sheet.max_column = table_count * 6 - 1
        return sheet
```

## Section 1: Sparse Pattern Detection Testing

### Test 1.1: Basic Pattern Detection
```python
from gridporter.vision.pattern_detector import SparsePatternDetector, PatternType

# Create test data
generator = TestDataGenerator()
sparse_sheet = generator.create_sparse_table(100, 5, density=0.2)

# Initialize detector
detector = SparsePatternDetector(
    min_filled_ratio=0.1,
    min_table_size=(2, 2),
    header_density_threshold=0.5
)

# Detect patterns
patterns = detector.detect_patterns(sparse_sheet)

print(f"Detected {len(patterns)} patterns")
for i, pattern in enumerate(patterns):
    print(f"Pattern {i+1}:")
    print(f"  Type: {pattern.pattern_type.value}")
    print(f"  Bounds: ({pattern.bounds.start_row},{pattern.bounds.start_col}) to "
          f"({pattern.bounds.end_row},{pattern.bounds.end_col})")
    print(f"  Confidence: {pattern.confidence:.2f}")
    print(f"  Characteristics: {pattern.characteristics}")
```

**Expected**: Should detect at least one HEADER_DATA pattern with confidence > 0.5

### Test 1.2: Matrix Pattern Detection
```python
# Create matrix pattern
matrix_sheet = generator.create_matrix_pattern(15)

# Detect patterns
patterns = detector.detect_patterns(matrix_sheet)

# Check for matrix patterns
matrix_patterns = [p for p in patterns if p.pattern_type == PatternType.MATRIX]
print(f"Found {len(matrix_patterns)} matrix patterns")

for pattern in matrix_patterns:
    print(f"Matrix at ({pattern.bounds.start_row},{pattern.bounds.start_col})")
    print(f"  Header rows: {pattern.header_rows}")
    print(f"  Header cols: {pattern.header_cols}")
```

**Expected**: Should detect one MATRIX pattern with row 0 and column 0 as headers

### Test 1.3: Form Pattern Detection
```python
# Create form pattern
form_sheet = generator.create_form_pattern(25)

# Detect patterns
patterns = detector.detect_patterns(form_sheet)

# Check for form patterns
form_patterns = [p for p in patterns if p.pattern_type == PatternType.FORM]
print(f"Found {len(form_patterns)} form patterns")

for pattern in form_patterns:
    print(f"Form with label column: {pattern.characteristics.get('label_column')}")
    print(f"  Total rows: {pattern.bounds.end_row - pattern.bounds.start_row + 1}")
```

**Expected**: Should detect one FORM pattern with column 0 as label column

### Test 1.4: Multiple Tables Detection
```python
# Create sheet with multiple tables
multi_sheet = generator.create_multiple_tables(3)

# Detect all patterns
patterns = detector.detect_patterns(multi_sheet)

print(f"Detected {len(patterns)} total patterns")

# Group by pattern type
by_type = {}
for pattern in patterns:
    type_name = pattern.pattern_type.value
    by_type[type_name] = by_type.get(type_name, 0) + 1

print("Patterns by type:")
for type_name, count in by_type.items():
    print(f"  {type_name}: {count}")
```

**Expected**: Should detect 3 separate table patterns

### Test 1.5: Pattern Confidence and Merging
```python
# Test confidence scoring
for pattern in patterns[:3]:  # First 3 patterns
    print(f"\nPattern confidence breakdown:")
    print(f"  Type: {pattern.pattern_type.value}")
    print(f"  Base confidence: {pattern.confidence:.2f}")
    print(f"  Fill ratio: {pattern.characteristics.get('fill_ratio', 0):.2f}")
    print(f"  Has headers: {pattern.characteristics.get('has_headers', False)}")

# Test overlapping patterns (detector should merge them)
print(f"\nTotal patterns after merging: {len(patterns)}")
```

**Expected**: Confidence scores between 0.5-1.0, overlapping patterns merged

## Section 2: Quadtree Analysis Testing

### Test 2.1: Basic Quadtree Construction
```python
from gridporter.vision.quadtree import QuadtreeAnalyzer, NodeType

# Create large sparse sheet
large_sheet = generator.create_sparse_table(1000, 100, density=0.05)

# Create analyzer
analyzer = QuadtreeAnalyzer(
    max_depth=6,
    min_node_size=100,
    density_threshold=0.1,
    pattern_aware=True
)

# Build quadtree
quadtree = analyzer.analyze(large_sheet)

# Get statistics
stats = analyzer.get_coverage_stats(quadtree)
print("Quadtree Statistics:")
print(f"  Total nodes: {stats['total_nodes']}")
print(f"  Empty nodes: {stats['empty_nodes']}")
print(f"  Sparse nodes: {stats['sparse_nodes']}")
print(f"  Dense nodes: {stats['dense_nodes']}")
print(f"  Max depth: {stats['max_depth']}")
print(f"  Overall density: {stats['overall_density']:.3f}")
```

**Expected**: Quadtree with multiple levels, mostly sparse/empty nodes

### Test 2.2: Pattern-Aware Splitting
```python
# Detect patterns first
patterns = detector.detect_patterns(large_sheet)
print(f"Detected {len(patterns)} patterns")

# Build pattern-aware quadtree
quadtree_aware = analyzer.analyze(large_sheet, patterns)

# Compare with non-pattern-aware
analyzer_naive = QuadtreeAnalyzer(pattern_aware=False)
quadtree_naive = analyzer_naive.analyze(large_sheet)

stats_aware = analyzer.get_coverage_stats(quadtree_aware)
stats_naive = analyzer_naive.get_coverage_stats(quadtree_naive)

print("\nPattern-aware vs Naive quadtree:")
print(f"  Aware nodes: {stats_aware['total_nodes']}")
print(f"  Naive nodes: {stats_naive['total_nodes']}")
```

**Expected**: Pattern-aware should have fewer but larger nodes

### Test 2.3: Visualization Planning
```python
# Plan visualization regions
regions = analyzer.plan_visualization(
    quadtree_aware,
    max_regions=10,
    max_total_size_mb=20.0
)

print(f"\nPlanned {len(regions)} visualization regions:")
for i, region in enumerate(regions):
    print(f"Region {i+1}:")
    print(f"  Bounds: ({region.bounds.min_row},{region.bounds.min_col}) to "
          f"({region.bounds.max_row},{region.bounds.max_col})")
    print(f"  Priority: {region.priority:.2f}")
    print(f"  Estimated size: {region.estimated_size_mb:.2f}MB")
    print(f"  Patterns: {len(region.patterns)}")
```

**Expected**: Up to 10 regions, total size < 20MB, high-priority regions first

### Test 2.4: Large Sheet Handling
```python
# Test with maximum Excel size (simulated with smaller bounds)
# NOTE: Don't actually create 1M rows - simulate with bounds
from gridporter.vision.quadtree import QuadTreeBounds, QuadTreeNode

# Simulate Excel max bounds
max_bounds = QuadTreeBounds(
    min_row=0, min_col=0,
    max_row=1048575,  # Excel max
    max_col=16383     # Excel max
)

root = QuadTreeNode(bounds=max_bounds, depth=0)
print(f"Root node covers {root.bounds.area:,} cells")

# Check quadrant splits
tl, tr, bl, br = root.bounds.split()
print(f"Top-left quadrant: {tl.area:,} cells")
print(f"Maximum depth needed: {analyzer.max_depth}")
```

**Expected**: Handles Excel's 17 billion cell limit gracefully

## Section 3: Enhanced Bitmap Generation

### Test 3.1: Compression Mode Selection
```python
from gridporter.vision.bitmap_generator import BitmapGenerator, CompressionMode

# Create generator with auto-compression
generator = BitmapGenerator(
    cell_width=10,
    cell_height=10,
    mode="binary",
    compression_level=6,
    auto_compress=True
)

# Test different sheet sizes
test_sizes = [
    (100, 100),      # 10K cells - should use FULL
    (500, 500),      # 250K cells - should use COMPRESSED_4
    (2000, 1000),    # 2M cells - should use COMPRESSED_2
    (10000, 5000),   # 50M cells - should use SAMPLED
]

for rows, cols in test_sizes:
    # Create test sheet
    sheet = SheetData(name=f"Test_{rows}x{cols}")
    sheet.max_row = rows - 1
    sheet.max_column = cols - 1

    # Add some data
    for i in range(0, rows, 100):
        for j in range(0, cols, 100):
            sheet.set_cell(i, j, CellData(value=f"R{i}C{j}"))

    # Generate bitmap
    img_data, metadata = generator.generate(sheet)

    print(f"\nSheet {rows}x{cols} ({rows*cols:,} cells):")
    print(f"  Compression: {metadata.compression.value}")
    print(f"  Image size: {metadata.width}x{metadata.height}")
    print(f"  File size: {metadata.file_size_bytes/1024:.1f}KB")
    print(f"  Cell size: {metadata.cell_width}x{metadata.cell_height}")
```

**Expected**: Different compression modes based on size, all files < 20MB

### Test 3.2: 2-bit Compression Testing
```python
# Force 2-bit compression
sparse_sheet = generator.create_sparse_table(5000, 100, density=0.1)

# Generate with different cell types
for row in range(10):
    sparse_sheet.set_cell(row, 0, CellData(value=f"Header{row}", is_bold=True))
    sparse_sheet.set_cell(row, 1, CellData(value=f"=SUM(A{row}:Z{row})", has_formula=True))
    sparse_sheet.set_cell(row, 2, CellData(value="Merged", is_merged=True))

img_data, metadata = generator._generate_compressed_2bit(
    sparse_sheet, None, 5000, 100
)

print("2-bit compression test:")
print(f"  Original cells: {5000*100:,}")
print(f"  Image dimensions: {metadata.width}x{metadata.height}")
print(f"  Compression ratio: {(5000*100)/(metadata.width*metadata.height):.1f}x")
print(f"  File size: {metadata.file_size_bytes/1024:.1f}KB")

# Save for visual inspection
with open("test_2bit.png", "wb") as f:
    f.write(img_data)
print("  Saved to test_2bit.png for inspection")
```

**Expected**: 4 distinct gray levels visible, significant size reduction

### Test 3.3: Sampled Mode for Huge Sheets
```python
# Simulate maximum Excel sheet
from gridporter.vision.quadtree import QuadTreeBounds

# Create bounds for huge region
huge_bounds = QuadTreeBounds(
    min_row=0, min_col=0,
    max_row=100000, max_col=1000
)

# Create sparse data in the huge region
huge_sheet = SheetData(name="HugeSheet")
huge_sheet.max_row = huge_bounds.max_row
huge_sheet.max_column = huge_bounds.max_col

# Add sparse data (don't fill everything!)
for i in range(0, 100000, 1000):
    for j in range(0, 1000, 50):
        if random.random() < 0.1:
            huge_sheet.set_cell(i, j, CellData(value="X"))

# Generate sampled bitmap
img_data, metadata = generator._generate_sampled(
    huge_sheet, huge_bounds,
    huge_bounds.height, huge_bounds.width
)

print("Sampled mode for 100M cells:")
print(f"  Sampling rate: ~{1/metadata.scale_factor:.0f}x")
print(f"  Final image: {metadata.width}x{metadata.height}")
print(f"  File size: {metadata.file_size_bytes/1024:.1f}KB")
```

**Expected**: Tiny image (< 1MB) representing huge region

### Test 3.4: Region-Based Generation
```python
# Use quadtree regions for bitmap generation
regions = analyzer.plan_visualization(quadtree_aware, max_regions=5)

# Generate bitmaps for each region
bitmap_results = generator.generate_from_visualization_plan(large_sheet, regions)

print(f"\nGenerated {len(bitmap_results)} bitmaps:")
total_size = 0
for i, result in enumerate(bitmap_results):
    print(f"Bitmap {i+1}:")
    print(f"  Region: {result.metadata.region_bounds.min_row}-"
          f"{result.metadata.region_bounds.max_row} x "
          f"{result.metadata.region_bounds.min_col}-"
          f"{result.metadata.region_bounds.max_col}")
    print(f"  Compression: {result.compression_info['mode']}")
    print(f"  Size: {result.compression_info['file_size_mb']:.2f}MB")
    total_size += result.compression_info['file_size_mb']

print(f"\nTotal size: {total_size:.2f}MB (limit: 20MB)")
```

**Expected**: Multiple bitmaps, each < 20MB, total < 20MB

## Section 4: Integration Testing

### Test 4.1: End-to-End Pipeline
```python
from gridporter.vision.bitmap_generator import BitmapGenerator
from gridporter.vision.pattern_detector import SparsePatternDetector
from gridporter.vision.quadtree import QuadtreeAnalyzer

# Create a complex test sheet
complex_sheet = SheetData(name="ComplexSheet")

# Add multiple tables with different patterns
# Table 1: Dense header-data table
for col in range(5):
    complex_sheet.set_cell(0, col, CellData(value=f"Col{col+1}", is_bold=True))
for row in range(1, 20):
    for col in range(5):
        complex_sheet.set_cell(row, col, CellData(value=f"T1_R{row}C{col}"))

# Table 2: Sparse matrix (offset)
for i in range(10):
    complex_sheet.set_cell(25+i, 10, CellData(value=f"Row{i}", is_bold=True))
    complex_sheet.set_cell(25, 10+i, CellData(value=f"Col{i}", is_bold=True))
for row in range(1, 10):
    for col in range(1, 10):
        if random.random() < 0.3:
            complex_sheet.set_cell(25+row, 10+col, CellData(value="X"))

# Table 3: Form pattern (offset)
complex_sheet.set_cell(40, 0, CellData(value="Task", is_bold=True))
complex_sheet.set_cell(40, 1, CellData(value="Done", is_bold=True))
for i in range(1, 15):
    complex_sheet.set_cell(40+i, 0, CellData(value=f"Task {i}"))
    if random.random() < 0.4:
        complex_sheet.set_cell(40+i, 1, CellData(value="✓"))

complex_sheet.max_row = 55
complex_sheet.max_column = 20

# Run full pipeline
print("Running end-to-end pipeline...")

# Step 1: Pattern detection
detector = SparsePatternDetector()
patterns = detector.detect_patterns(complex_sheet)
print(f"Step 1: Detected {len(patterns)} patterns")

# Step 2: Quadtree analysis
analyzer = QuadtreeAnalyzer(pattern_aware=True)
quadtree = analyzer.analyze(complex_sheet, patterns)
print(f"Step 2: Built quadtree with max depth {analyzer.get_coverage_stats(quadtree)['max_depth']}")

# Step 3: Visualization planning
regions = analyzer.plan_visualization(quadtree, max_regions=5)
print(f"Step 3: Planned {len(regions)} regions")

# Step 4: Bitmap generation
generator = BitmapGenerator(auto_compress=True)
bitmaps = generator.generate_from_visualization_plan(complex_sheet, regions)
print(f"Step 4: Generated {len(bitmaps)} bitmaps")

# Verify results
for i, (region, bitmap) in enumerate(zip(regions, bitmaps)):
    print(f"\nRegion {i+1} -> Bitmap {i+1}:")
    print(f"  Patterns in region: {len(region.patterns)}")
    print(f"  Bitmap size: {bitmap.compression_info['file_size_mb']:.2f}MB")
    print(f"  Compression used: {bitmap.compression_info['mode']}")
```

**Expected**: 3 patterns detected, 3-5 regions planned, all bitmaps generated

### Test 4.2: Performance Benchmarks
```python
import time

def benchmark_sheet_processing(rows: int, cols: int, density: float):
    """Benchmark processing time for different sheet sizes."""
    # Create sheet
    start = time.time()
    sheet = TestDataGenerator.create_sparse_table(rows, cols, density)
    create_time = time.time() - start

    # Pattern detection
    start = time.time()
    detector = SparsePatternDetector()
    patterns = detector.detect_patterns(sheet)
    detect_time = time.time() - start

    # Quadtree analysis
    start = time.time()
    analyzer = QuadtreeAnalyzer()
    quadtree = analyzer.analyze(sheet, patterns)
    quadtree_time = time.time() - start

    # Visualization planning
    start = time.time()
    regions = analyzer.plan_visualization(quadtree)
    plan_time = time.time() - start

    # Bitmap generation
    start = time.time()
    generator = BitmapGenerator(auto_compress=True)
    if regions:
        bitmaps = generator.generate_from_visualization_plan(sheet, regions[:3])
        bitmap_time = time.time() - start
    else:
        bitmap_time = 0

    total_time = detect_time + quadtree_time + plan_time + bitmap_time

    return {
        'size': f"{rows}x{cols}",
        'cells': rows * cols,
        'create_time': create_time,
        'detect_time': detect_time,
        'quadtree_time': quadtree_time,
        'plan_time': plan_time,
        'bitmap_time': bitmap_time,
        'total_time': total_time,
        'patterns': len(patterns),
        'regions': len(regions)
    }

# Run benchmarks
benchmarks = [
    (100, 50, 0.3),      # Small dense
    (1000, 100, 0.1),    # Medium sparse
    (10000, 500, 0.01),  # Large very sparse
]

print("Performance Benchmarks:")
print("-" * 80)
print(f"{'Size':<12} {'Cells':<10} {'Create':<8} {'Detect':<8} {'Quadtree':<8} {'Plan':<8} {'Bitmap':<8} {'Total':<8}")
print("-" * 80)

for rows, cols, density in benchmarks:
    result = benchmark_sheet_processing(rows, cols, density)
    print(f"{result['size']:<12} {result['cells']:<10,} "
          f"{result['create_time']:<8.3f} {result['detect_time']:<8.3f} "
          f"{result['quadtree_time']:<8.3f} {result['plan_time']:<8.3f} "
          f"{result['bitmap_time']:<8.3f} {result['total_time']:<8.3f}")
```

**Expected**: Processing time should scale sub-linearly with sheet size

### Test 4.3: Memory Usage Testing
```python
import psutil
import os

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Test memory usage for large sheets
print("Memory Usage Testing:")
print("-" * 50)

initial_memory = get_memory_usage()
print(f"Initial memory: {initial_memory:.1f}MB")

# Create large sheet
sheet = TestDataGenerator.create_sparse_table(50000, 200, density=0.01)
after_create = get_memory_usage()
print(f"After creating 10M cell sheet: {after_create:.1f}MB (+{after_create-initial_memory:.1f}MB)")

# Run detection
detector = SparsePatternDetector()
patterns = detector.detect_patterns(sheet)
after_detect = get_memory_usage()
print(f"After pattern detection: {after_detect:.1f}MB (+{after_detect-after_create:.1f}MB)")

# Build quadtree
analyzer = QuadtreeAnalyzer()
quadtree = analyzer.analyze(sheet, patterns)
after_quadtree = get_memory_usage()
print(f"After quadtree analysis: {after_quadtree:.1f}MB (+{after_quadtree-after_detect:.1f}MB)")

# Generate bitmaps
regions = analyzer.plan_visualization(quadtree, max_regions=3)
generator = BitmapGenerator(auto_compress=True)
bitmaps = generator.generate_from_visualization_plan(sheet, regions)
after_bitmaps = get_memory_usage()
print(f"After bitmap generation: {after_bitmaps:.1f}MB (+{after_bitmaps-after_quadtree:.1f}MB)")

print(f"\nTotal memory increase: {after_bitmaps-initial_memory:.1f}MB")
```

**Expected**: Memory usage should be reasonable (<1GB for 10M cells)

## Section 5: Edge Cases and Stress Testing

### Test 5.1: Empty and Nearly Empty Sheets
```python
# Completely empty sheet
empty_sheet = SheetData(name="Empty")
empty_sheet.max_row = 1000
empty_sheet.max_column = 100

patterns = detector.detect_patterns(empty_sheet)
print(f"Empty sheet patterns: {len(patterns)}")

# Sheet with single cell
single_cell = SheetData(name="SingleCell")
single_cell.set_cell(500, 50, CellData(value="Lonely"))
single_cell.max_row = 1000
single_cell.max_column = 100

patterns = detector.detect_patterns(single_cell)
print(f"Single cell patterns: {len(patterns)}")

# Generate bitmap for empty sheet
img_data, metadata = generator.generate(empty_sheet)
print(f"Empty sheet bitmap size: {metadata.file_size_bytes} bytes")
```

**Expected**: No patterns for empty, possibly 1 for single cell, minimal bitmap

### Test 5.2: Maximum Size Handling
```python
# Test Excel limits (don't actually create full sheet!)
from gridporter.vision.bitmap_generator import BitmapGenerator

# Check size calculations
xlsx_cells = 1048576 * 16384  # ~17 billion
xls_cells = 65536 * 256       # ~17 million

print(f"XLSX max cells: {xlsx_cells:,}")
print(f"XLS max cells: {xls_cells:,}")

# Calculate bitmap sizes for different compressions
gen = BitmapGenerator()
for bits in [8, 4, 2]:
    size_mb = (xlsx_cells * bits / 8) / (1024 * 1024)
    print(f"{bits}-bit representation: {size_mb:,.0f}MB")

# Test compression mode selection
mode = gen._determine_compression_mode(1048576, 16384)
print(f"Auto-selected mode for XLSX max: {mode.value}")
```

**Expected**: SAMPLED mode for maximum size, reasonable memory estimates

### Test 5.3: Malformed Data Handling
```python
# Create sheet with problematic data
problem_sheet = SheetData(name="Problems")

# Add various edge cases
problem_sheet.set_cell(0, 0, CellData(value="", is_bold=True))  # Empty bold
problem_sheet.set_cell(1, 0, CellData(value=None))  # None value
problem_sheet.set_cell(2, 0, CellData(value="  \n  "))  # Whitespace

# Circular references in characteristics
problem_sheet.max_row = 10
problem_sheet.max_column = 10

try:
    patterns = detector.detect_patterns(problem_sheet)
    print(f"Handled problematic data: {len(patterns)} patterns")
except Exception as e:
    print(f"Error with problematic data: {e}")
```

**Expected**: Graceful handling, no crashes

### Test 5.4: Pattern Detection in Noise
```python
# Create very noisy sheet
noisy_sheet = SheetData(name="Noisy")

# Add random noise
for _ in range(1000):
    row = random.randint(0, 999)
    col = random.randint(0, 99)
    noisy_sheet.set_cell(row, col, CellData(value="noise"))

# Add a clear pattern in the noise
for col in range(10, 20):
    noisy_sheet.set_cell(100, col, CellData(value=f"Header{col}", is_bold=True))
for row in range(101, 120):
    for col in range(10, 20):
        if random.random() < 0.7:  # 70% filled
            noisy_sheet.set_cell(row, col, CellData(value=f"Data{row}{col}"))

noisy_sheet.max_row = 999
noisy_sheet.max_column = 99

# Detect patterns
patterns = detector.detect_patterns(noisy_sheet)
print(f"Patterns found in noise: {len(patterns)}")

# Check if the real pattern was found
for pattern in patterns:
    if (pattern.bounds.start_row >= 100 and pattern.bounds.start_row <= 120 and
        pattern.bounds.start_col >= 10 and pattern.bounds.start_col <= 20):
        print(f"Found embedded pattern! Confidence: {pattern.confidence:.2f}")
```

**Expected**: Should find the embedded pattern despite noise

## Performance Guidelines

### Expected Performance Metrics

| Sheet Size | Cells | Pattern Detection | Quadtree | Bitmap Gen | Total |
|------------|-------|------------------|----------|------------|-------|
| 100×50 | 5K | <10ms | <5ms | <20ms | <50ms |
| 1,000×100 | 100K | <50ms | <20ms | <100ms | <200ms |
| 10,000×500 | 5M | <500ms | <200ms | <500ms | <2s |
| 100,000×1,000 | 100M | <5s | <2s | <2s | <10s |

### Memory Usage Guidelines

- Small sheets (<100K cells): ~50-100MB
- Medium sheets (100K-10M cells): ~200-500MB
- Large sheets (10M-100M cells): ~500MB-2GB
- Maximum sheets (1B+ cells): ~2-4GB with sampling

### Optimization Tips

1. **Pattern Detection**:
   - Increase `min_filled_ratio` for faster detection
   - Reduce `header_density_threshold` for sparser data

2. **Quadtree Analysis**:
   - Reduce `max_depth` for faster processing
   - Increase `min_node_size` to create fewer nodes

3. **Bitmap Generation**:
   - Use `auto_compress=True` for automatic optimization
   - Force `SAMPLED` mode for very large sheets
   - Increase `compression_level` for smaller files

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   ```python
   # Use region-based processing
   regions = analyzer.plan_visualization(quadtree, max_regions=1)
   # Process one region at a time
   ```

2. **Slow Pattern Detection**:
   ```python
   # Increase thresholds
   detector = SparsePatternDetector(
       min_filled_ratio=0.2,  # Higher threshold
       min_table_size=(5, 5)  # Larger minimum
   )
   ```

3. **Large Bitmap Files**:
   ```python
   # Force maximum compression
   generator = BitmapGenerator(
       compression_level=9,
       auto_compress=True
   )
   ```

4. **Missing Patterns**:
   ```python
   # Lower thresholds for sparse data
   detector = SparsePatternDetector(
       min_filled_ratio=0.05,
       header_density_threshold=0.3
   )
   ```

### Debug Visualization

```python
# Save intermediate results for debugging
def save_debug_visualization(sheet, patterns, quadtree, regions):
    """Save debug information for analysis."""
    import json

    debug_info = {
        'sheet_size': f"{sheet.max_row+1}x{sheet.max_column+1}",
        'patterns': [
            {
                'type': p.pattern_type.value,
                'bounds': f"{p.bounds.start_row},{p.bounds.start_col} to {p.bounds.end_row},{p.bounds.end_col}",
                'confidence': p.confidence
            }
            for p in patterns
        ],
        'quadtree_stats': analyzer.get_coverage_stats(quadtree),
        'regions': [
            {
                'bounds': f"{r.bounds.min_row}-{r.bounds.max_row} x {r.bounds.min_col}-{r.bounds.max_col}",
                'priority': r.priority,
                'size_mb': r.estimated_size_mb
            }
            for r in regions
        ]
    }

    with open('debug_vision_pipeline.json', 'w') as f:
        json.dump(debug_info, f, indent=2)

    print("Debug info saved to debug_vision_pipeline.json")
```

## Next Steps

After completing these tests:

1. **Integration with Main Pipeline**: Connect these components to the vision orchestrator
2. **Caching Implementation**: Add caching for pattern detection and quadtree results
3. **Batch Processing**: Handle multiple sheets efficiently
4. **Performance Profiling**: Identify and optimize bottlenecks
5. **Production Hardening**: Add comprehensive error handling and logging
